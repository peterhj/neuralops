use prelude::*;
//use common::*;
use kernels::activate::*;

//use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array2d};
//use densearray::linalg::{Transpose};
use densearray::prelude::*;
use operator::prelude::*;
use operator::io::{IoBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
//use std::cmp::{max};
//use std::marker::{PhantomData};
use std::rc::{Rc};

#[derive(Clone, Copy)]
pub struct AffineOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   usize,
  pub out_dim:  usize,
  pub bias:     bool,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

pub struct NewAffineOperator<S, IoBuf: ?Sized> {
  cfg:      AffineOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  weights:  Rc<RefCell<ParamBlock<Array2d<f32>>>>,
  //w_grad:   Array2d<f32>,
  bias:     Rc<RefCell<ParamBlock<Array1d<f32>>>>,
  //b_grad:   Array1d<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  act_kern: ActivateKernel,
  //_marker:  PhantomData<S>,
}

impl<S, IoBuf: ?Sized> NewAffineOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: AffineOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewAffineOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
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
      weights:  ParamBlock::new(cap, || Array2d::zeros((cfg.out_dim, cfg.in_dim))),
      //w_grad:   Array2d::zeros((cfg.out_dim, cfg.in_dim)),
      bias:     ParamBlock::new(cap, || Array1d::zeros(cfg.out_dim)),
      //b_grad:   Array1d::zeros(cfg.out_dim),
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim, cfg.act_kind),
      //_marker:  PhantomData,
    }))
  }

  /*pub fn diff_op(&mut self) -> &mut DiffOperator<S, IoBuf> {
    self
  }*/
}

impl<S, IoBuf: ?Sized> Operator for NewAffineOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for NewAffineOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for NewAffineOperator<S, IoBuf> {
  default fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }
}

impl<S> DiffOperatorIo<[f32]> for NewAffineOperator<S, [f32]> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.weights.borrow_mut().as_mut_slice());
    offset += param_reader.read_buf(offset, self.bias.borrow_mut().as_mut_slice());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write_buf(offset, self.weights.borrow().as_slice());
    offset += param_writer.write_buf(offset, self.bias.borrow().as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write_buf(offset, self.weights.borrow().grad().as_slice());
    offset += grad_writer.write_buf(offset, self.bias.borrow().grad().as_slice());
    offset - init_offset
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for NewAffineOperator<S, IoBuf> {
  //type IoBuf = [f32];
  //type Op = Rc<RefCell<CommonOperator>>;

  //fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf, Op=CommonOperator>)) {
  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  //fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf, Op=CommonOperator>)) {
  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
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
        for e in self.weights.borrow_mut().as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for e in self.weights.borrow_mut().as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Xavier => {
        let half_range = (6.0 / (self.cfg.in_dim + self.cfg.out_dim) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.borrow_mut().as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim, self.cfg.out_dim) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.borrow_mut().as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    for e in self.bias.borrow_mut().as_mut_slice().iter_mut() {
      *e = 0.0;
    }
  }

  fn _reset_grad(&mut self) {
    self.weights.borrow_mut().grad_mut().as_view_mut().set_constant(0.0);
    self.bias.borrow_mut().grad_mut().as_view_mut().set_constant(0.0);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    let in_buf = self.in_.buf.borrow();
    //println!("DEBUG: affine: input: {:?}", &in_buf[ .. self.cfg.in_dim]);
    self.tmp_buf.reshape_mut((self.cfg.out_dim, batch_size))
      .matrix_prod(
          1.0,
          self.weights.borrow().as_view(), Transpose::N,
          in_buf.reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          0.0,
      );
    for j in 0 .. batch_size {
      self.tmp_buf.reshape_mut((self.cfg.out_dim, batch_size))
        .view_mut((0, j), (self.cfg.out_dim, j+1))
        .matrix_add(
            1.0,
            self.bias.borrow().as_view().reshape((self.cfg.out_dim, 1)),
        );
    }

    let mut out_buf = self.out.buf.borrow_mut();
    self.act_kern.forward(batch_size, &self.tmp_buf, &mut *out_buf);
    //println!("DEBUG: affine: output: {:?}", &out_buf[ .. self.cfg.out_dim]);
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    self.act_kern.backward(batch_size, &self.out.buf.borrow(), &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    self.weights.borrow_mut().grad_mut().as_view_mut()
      .matrix_prod(
          1.0,
          self.tmp_grad.reshape((self.cfg.out_dim, batch_size)), Transpose::N,
          self.in_.buf.borrow().reshape((self.cfg.in_dim, batch_size)), Transpose::T,
          1.0,
      );
    for j in 0 .. batch_size {
      self.bias.borrow_mut().grad_mut().as_view_mut().reshape_mut((self.cfg.out_dim, 1))
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
            self.weights.borrow().as_view(), Transpose::T,
            self.tmp_grad.reshape((self.cfg.out_dim, batch_size)), Transpose::N,
            0.0,
        );
    }
  }
}

pub struct ParallelAffineOperator<S, IoBuf: ?Sized> {
  cfg:      AffineOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  weights:  Rc<RefCell<ParamBlock<Array2d<f32>>>>,
  bias:     Rc<RefCell<ParamBlock<Array1d<f32>>>>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  act_kern: ParallelActivateKernel,
  watch:    Stopwatch,
}

impl<S, IoBuf: ?Sized> ParallelAffineOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: AffineOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<ParallelAffineOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let out_len = cfg.batch_sz * cfg.out_dim;
    let mut tmp_buf = Vec::with_capacity(out_len);
    tmp_buf.resize(out_len, 0.0);
    let mut tmp_grad = Vec::with_capacity(out_len);
    tmp_grad.resize(out_len, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(ParallelAffineOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim, cap),
      weights:  ParamBlock::new(cap, || Array2d::zeros((cfg.out_dim, cfg.in_dim))),
      bias:     ParamBlock::new(cap, || Array1d::zeros(cfg.out_dim)),
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      act_kern: ParallelActivateKernel::new(cfg.batch_sz, cfg.out_dim, cfg.act_kind),
      watch:    Stopwatch::new(),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for ParallelAffineOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for ParallelAffineOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for ParallelAffineOperator<S, IoBuf> {
  default fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }
}

impl<S> DiffOperatorIo<[f32]> for ParallelAffineOperator<S, [f32]> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.weights.borrow_mut().as_mut_slice());
    offset += param_reader.read_buf(offset, self.bias.borrow_mut().as_mut_slice());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write_buf(offset, self.weights.borrow().as_slice());
    offset += param_writer.write_buf(offset, self.bias.borrow().as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write_buf(offset, self.weights.borrow().grad().as_slice());
    offset += grad_writer.write_buf(offset, self.bias.borrow().grad().as_slice());
    offset - init_offset
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for ParallelAffineOperator<S, IoBuf> {
  //type IoBuf = [f32];
  //type Op = Rc<RefCell<CommonOperator>>;

  //fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf, Op=CommonOperator>)) {
  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  //fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf, Op=CommonOperator>)) {
  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
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
        for e in self.weights.borrow_mut().as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for e in self.weights.borrow_mut().as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Xavier => {
        let half_range = (6.0 / (self.cfg.in_dim + self.cfg.out_dim) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.borrow_mut().as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim, self.cfg.out_dim) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.borrow_mut().as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    for e in self.bias.borrow_mut().as_mut_slice().iter_mut() {
      *e = 0.0;
    }
  }

  fn _reset_grad(&mut self) {
    self.weights.borrow_mut().grad_mut().as_view_mut().parallel_set_constant(0.0);
    self.bias.borrow_mut().grad_mut().as_view_mut().parallel_set_constant(0.0);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    self.watch.lap();

    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    let in_buf = self.in_.buf.borrow();
    //println!("DEBUG: affine: input: {:?}", &in_buf[ .. self.cfg.in_dim]);
    self.tmp_buf.reshape_mut((self.cfg.out_dim, batch_size))
      .parallel_matrix_prod(
          1.0,
          self.weights.borrow().as_view(), Transpose::N,
          in_buf.reshape((self.cfg.in_dim, batch_size)), Transpose::N,
          0.0,
      );

    if self.cfg.bias {
      for j in 0 .. batch_size {
        self.tmp_buf.reshape_mut((self.cfg.out_dim, batch_size))
          .view_mut((0, j), (self.cfg.out_dim, j+1))
          .matrix_add(
              1.0,
              self.bias.borrow().as_view().reshape((self.cfg.out_dim, 1)),
          );
      }
    }

    let mut out_buf = self.out.buf.borrow_mut();
    self.act_kern.forward(batch_size, &self.tmp_buf, &mut *out_buf);
    //println!("DEBUG: affine: output: {:?}", &out_buf[ .. self.cfg.out_dim]);

    self.watch.lap();
    //println!("DEBUG: affine: fwd: {:.6}", self.watch.elapsed());
  }

  fn _backward(&mut self) {
    self.watch.lap();

    let batch_size = self.out.batch_sz.get();

    self.act_kern.backward(batch_size, &self.out.buf.borrow(), &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    if self.cfg.bias {
      for j in 0 .. batch_size {
        self.bias.borrow_mut().grad_mut().as_view_mut().reshape_mut((self.cfg.out_dim, 1))
          .matrix_add(
              1.0,
              self.tmp_grad.reshape((self.cfg.out_dim, batch_size))
                .view((0, j), (self.cfg.out_dim, j+1)),
          );
      }
    }

    self.weights.borrow_mut().grad_mut().as_view_mut()
      .parallel_matrix_prod(
          1.0,
          self.tmp_grad.reshape((self.cfg.out_dim, batch_size)), Transpose::N,
          self.in_.buf.borrow().reshape((self.cfg.in_dim, batch_size)), Transpose::T,
          1.0,
      );

    if let Some(in_grad) = self.in_.grad.as_ref() {
      in_grad.borrow_mut().reshape_mut((self.cfg.in_dim, batch_size))
        .parallel_matrix_prod(
            1.0,
            self.weights.borrow().as_view(), Transpose::T,
            self.tmp_grad.reshape((self.cfg.out_dim, batch_size)), Transpose::N,
            0.0,
        );
    }

    self.watch.lap();
    //println!("DEBUG: affine: bwd: {:.6}", self.watch.elapsed());
  }
}

#[derive(Clone, Copy)]
pub struct BatchNormAffineOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   usize,
  pub out_dim:  usize,
  pub avg_rate: f32,
  pub epsilon:  f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

