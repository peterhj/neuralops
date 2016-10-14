//use common::{CommonResources, CommonOperatorOutput, ActivationKind, ParamInitKind};
use common::*;
use kernels::activate::{ActivateKernel};

use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array2d};
use densearray::linalg::{Transpose};
use operator::prelude::*;
use operator::io::{IoBuffer};
use operator::rw::{ReadBuffer, WriteBuffer, ReadAccumulateBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{max};
use std::marker::{PhantomData};
use std::rc::{Rc};

#[derive(Clone, Copy)]
pub struct AffineOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   usize,
  pub out_dim:  usize,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

pub struct AffineOperator {
  cfg:      AffineOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  weights:  Array2d<f32>,
  w_grad:   Array2d<f32>,
  bias:     Array1d<f32>,
  b_grad:   Array1d<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  act_kern: ActivateKernel,
  out:      CommonOperatorOutput<f32>,
}

impl AffineOperator {
  pub fn new(cfg: AffineOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> AffineOperator {
    let out_len = cfg.batch_sz * cfg.out_dim;
    let mut tmp_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_buf.push(0.0);
    }
    let mut tmp_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_grad.push(0.0);
    }
    AffineOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      weights:  Array2d::zeros((cfg.in_dim, cfg.out_dim)),
      w_grad:   Array2d::zeros((cfg.in_dim, cfg.out_dim)),
      bias:     Array1d::zeros(cfg.out_dim),
      b_grad:   Array1d::zeros(cfg.out_dim),
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim, cfg.act_kind),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim, cap),
    }
  }
}

impl DiffOperator<f32> for AffineOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn diff_param_sz(&self) -> usize {
    self.cfg.in_dim * self.cfg.out_dim + self.cfg.out_dim
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    match self.cfg.w_init {
      ParamInitKind::Disabled => {
        panic!("parameter initialization explicitly disabled");
      }
      ParamInitKind::Uniform{lo, hi} => {
        let dist = Range::new(lo, hi);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Xavier => {
        let half_range = (6.0 / (self.cfg.in_dim + self.cfg.out_dim) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim, self.cfg.out_dim) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    for e in self.bias.as_mut_slice().iter_mut() {
      *e = 0.0;
    }
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read(offset, self.weights.as_mut_slice());
    offset += param_reader.read(offset, self.bias.as_mut_slice());
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write(offset, self.weights.as_slice());
    offset += param_writer.write(offset, self.bias.as_slice());
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.weights.as_mut_slice());
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.bias.as_mut_slice());
    offset - init_offset
  }

  fn reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0);
    self.b_grad.as_view_mut().set_constant(0.0);
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        let weights_len = self.weights.dim().flat_len();
        let weights = self.weights.as_mut_slice();
        let w_grad = self.w_grad.as_mut_slice();
        let bias_len = self.bias.dim().flat_len();
        let bias = self.bias.as_mut_slice();
        let b_grad = self.b_grad.as_mut_slice();
        for j in 0 .. weights_len {
          w_grad[j] += lambda * weights[j];
        }
        for j in 0 .. bias_len {
          b_grad[j] += lambda * bias[j];
        }
      }
    }
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write(offset, self.w_grad.as_slice());
    offset += grad_writer.write(offset, self.b_grad.as_slice());
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.w_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, self.b_grad.as_slice());
    offset - init_offset
  }

  fn forward(&mut self, _phase: OpPhase) {
    let batch_size = *self.in_.batch_size.borrow();
    *self.out.batch_size.borrow_mut() = batch_size;
    assert!(batch_size <= self.cfg.batch_sz);

    self.tmp_buf.reshape_mut((self.cfg.out_dim, batch_size))
      .matrix_prod(
          1.0,
          self.weights.as_view(), Transpose::T,
          self.in_.out_buf.borrow().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          0.0,
      );
    for j in 0 .. batch_size {
      self.tmp_buf.reshape_mut((self.cfg.out_dim, batch_size))
        .view_mut((0, j), (self.cfg.out_dim, j+1))
        .matrix_add(
            1.0,
            self.bias.as_view().reshape((self.cfg.out_dim, 1)),
        );
    }

    //activate_fwd(self.cfg.act_kind, &self.tmp_buf, &mut *self.out.out_buf.borrow_mut());
    self.act_kern.forward(batch_size, &self.tmp_buf, &mut *self.out.out_buf.borrow_mut());

    let in_loss = *self.in_.out_loss.borrow();
    *self.out.out_loss.borrow_mut() = in_loss;
  }

  /*fn fwd_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        // FIXME(20161002): regularize the bias too?
        let w_norm = self.weights.as_view().reshape(self.cfg.in_dim * self.cfg.out_dim).l2_norm();
        *self.out.out_loss.borrow_mut() = 0.5 * lambda * w_norm * w_norm;
      }
    }
  }*/

  fn backward(&mut self) {
    //assert!(self.in_.batch_size <= self.cfg.batch_sz);
    //assert_eq!(self.out.batch_size, self.in_.batch_size);
    let batch_size = *self.out.batch_size.borrow();

    //activate_bwd(self.cfg.act_kind, &self.tmp_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);
    self.act_kern.backward(batch_size, &self.tmp_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    self.w_grad.as_view_mut()
      .matrix_prod(
          1.0,
          self.in_.out_buf.borrow().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          self.tmp_grad.reshape((self.cfg.out_dim, batch_size)), Transpose::T,
          1.0,
      );
    for j in 0 .. batch_size {
      self.b_grad.as_view_mut().reshape_mut((self.cfg.out_dim, 1))
        .matrix_add(
            1.0,
            self.tmp_grad.reshape((self.cfg.out_dim, batch_size))
              .view((0, j), (self.cfg.out_dim, j+1)),
        );
    }

    if let Some(in_grad) = self.in_.out_grad.as_ref() {
      in_grad.borrow_mut().reshape_mut((self.cfg.in_dim, batch_size))
        .matrix_prod(
            1.0,
            self.weights.as_view(), Transpose::N,
            self.tmp_grad.reshape((self.cfg.out_dim, batch_size)), Transpose::N,
            0.0,
        );
    }
  }

  /*fn bwd_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(_lambda) => {
        // FIXME(20161002)
        unimplemented!();
      }
    }
  }*/
}

pub struct NewAffineOperator<S> {
  cfg:      AffineOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  weights:  Array2d<f32>,
  w_grad:   Array2d<f32>,
  bias:     Array1d<f32>,
  b_grad:   Array1d<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  act_kern: ActivateKernel,
  _marker:  PhantomData<S>,
}

impl<S> NewAffineOperator<S> {
  pub fn new<InOp>(cfg: AffineOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewAffineOperator<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let out_len = cfg.batch_sz * cfg.out_dim;
    let mut tmp_buf = Vec::with_capacity(out_len);
    tmp_buf.resize(out_len, 0.0);
    let mut tmp_grad = Vec::with_capacity(out_len);
    tmp_grad.resize(out_len, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(NewAffineOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim, cap),
      weights:  Array2d::zeros((cfg.in_dim, cfg.out_dim)),
      w_grad:   Array2d::zeros((cfg.in_dim, cfg.out_dim)),
      bias:     Array1d::zeros(cfg.out_dim),
      b_grad:   Array1d::zeros(cfg.out_dim),
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim, cfg.act_kind),
      _marker:  PhantomData,
    }))
  }
}

impl<S> Operator for NewAffineOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for NewAffineOperator<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for NewAffineOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
  }

  fn _diff_param_sz(&self) -> usize {
    self.cfg.in_dim * self.cfg.out_dim + self.cfg.out_dim
  }

  fn _init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    match self.cfg.w_init {
      ParamInitKind::Disabled => {
        panic!("parameter initialization explicitly disabled");
      }
      ParamInitKind::Uniform{lo, hi} => {
        let dist = Range::new(lo, hi);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Xavier => {
        let half_range = (6.0 / (self.cfg.in_dim + self.cfg.out_dim) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim, self.cfg.out_dim) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    for e in self.bias.as_mut_slice().iter_mut() {
      *e = 0.0;
    }
  }

  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read(offset, self.weights.as_mut_slice());
    offset += param_reader.read(offset, self.bias.as_mut_slice());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write(offset, self.weights.as_slice());
    offset += param_writer.write(offset, self.bias.as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write(offset, self.w_grad.as_slice());
    offset += grad_writer.write(offset, self.b_grad.as_slice());
    offset - init_offset
  }

  fn _reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0);
    self.b_grad.as_view_mut().set_constant(0.0);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    self.tmp_buf.reshape_mut((self.cfg.out_dim, batch_size))
      .matrix_prod(
          1.0,
          self.weights.as_view(), Transpose::T,
          self.in_.buf.borrow().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          0.0,
      );
    for j in 0 .. batch_size {
      self.tmp_buf.reshape_mut((self.cfg.out_dim, batch_size))
        .view_mut((0, j), (self.cfg.out_dim, j+1))
        .matrix_add(
            1.0,
            self.bias.as_view().reshape((self.cfg.out_dim, 1)),
        );
    }

    self.act_kern.forward(batch_size, &self.tmp_buf, &mut *self.out.buf.borrow_mut());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    self.act_kern.backward(batch_size, &self.tmp_buf, &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    self.w_grad.as_view_mut()
      .matrix_prod(
          1.0,
          self.in_.buf.borrow().reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          self.tmp_grad.reshape((self.cfg.out_dim, batch_size)), Transpose::T,
          1.0,
      );
    for j in 0 .. batch_size {
      self.b_grad.as_view_mut().reshape_mut((self.cfg.out_dim, 1))
        .matrix_add(
            1.0,
            self.tmp_grad.reshape((self.cfg.out_dim, batch_size))
              .view((0, j), (self.cfg.out_dim, j+1)),
        );
    }

    if let Some(in_grad) = self.in_.grad.as_ref() {
      in_grad.borrow_mut().reshape_mut((self.cfg.in_dim, batch_size))
        .matrix_prod(
            1.0,
            self.weights.as_view(), Transpose::N,
            self.tmp_grad.reshape((self.cfg.out_dim, batch_size)), Transpose::N,
            0.0,
        );
    }
  }
}
