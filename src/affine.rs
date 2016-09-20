use common::{CommonOperatorOutput, ActivationKind, ParamInitKind};
use kernels::{activate_fwd, activate_bwd};

use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array2d};
use densearray::linalg::{Transpose};
use operator::prelude::*;
use operator::rw::{ReadBuffer, WriteBuffer, ReadAccumulateBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

//use rand::{Rng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cmp::{max, min};

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
  bias:     Vec<f32>,
  b_grad:   Vec<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  out:      CommonOperatorOutput<f32>,
}

impl AffineOperator {
  pub fn new(cfg: AffineOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>>, prev_arm: usize) -> AffineOperator {
    let mut bias = Vec::with_capacity(cfg.out_dim);
    unsafe { bias.set_len(cfg.out_dim) };
    let mut b_grad = Vec::with_capacity(cfg.out_dim);
    unsafe { b_grad.set_len(cfg.out_dim) };
    let mut tmp_buf = Vec::with_capacity(cfg.batch_sz * cfg.out_dim);
    unsafe { tmp_buf.set_len(cfg.batch_sz * cfg.out_dim) };
    let mut tmp_grad = Vec::with_capacity(cfg.batch_sz * cfg.out_dim);
    unsafe { tmp_grad.set_len(cfg.batch_sz * cfg.out_dim) };
    AffineOperator{
      cfg:      cfg,
      in_:      prev_op.output(prev_arm),
      weights:  Array2d::zeros((cfg.in_dim, cfg.out_dim)),
      w_grad:   Array2d::zeros((cfg.in_dim, cfg.out_dim)),
      bias:     bias,
      b_grad:   b_grad,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim, cap),
    }
  }
}

impl DiffOperator<f32> for AffineOperator {
  type Output = CommonOperatorOutput<f32>;

  fn output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn param_len(&self) -> usize {
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
        let std = (2.0 / max(self.cfg.in_dim, self.cfg.out_dim) as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    for e in self.bias.iter_mut() {
      *e = 0.0;
    }
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read(offset, self.weights.as_mut_slice());
    offset += param_reader.read(offset, &mut self.bias);
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write(offset, self.weights.as_slice());
    offset += param_writer.write(offset, &self.bias);
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.weights.as_mut_slice());
    offset += grad_reader.read_accumulate(alpha, beta, offset, &mut self.bias);
    offset - init_offset
  }

  fn reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0);
    for e in self.b_grad.iter_mut() {
      *e = 0.0;
    }
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        let weights_len = self.weights.dim().flat_len();
        let weights = self.weights.as_mut_slice();
        let w_grad = self.w_grad.as_mut_slice();
        for j in 0 .. weights_len {
          w_grad[j] += lambda * weights[j];
        }
        for j in 0 .. self.bias.len() {
          self.b_grad[j] += lambda * self.bias[j];
        }
      }
    }
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write(offset, self.w_grad.as_slice());
    offset += grad_writer.write(offset, &self.b_grad);
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.w_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, &self.b_grad);
    offset - init_offset
  }

  fn forward(&mut self, _phase: OpPhase) {
    assert!(self.in_.batch_size <= self.cfg.batch_sz);
    self.out.batch_size = self.in_.batch_size;

    self.tmp_buf.reshape_mut((self.cfg.out_dim, self.out.batch_size))
      .matrix_prod(
          1.0,
          self.weights.as_view(), Transpose::T,
          self.in_.out_buf.borrow().reshape((self.cfg.in_dim, self.in_.batch_size)), Transpose::N,
          0.0,
      );
    for j in 0 .. self.out.batch_size {
      self.tmp_buf.reshape_mut((self.cfg.out_dim, self.out.batch_size))
        .view_mut((0, j), (self.cfg.out_dim, j+1))
        .matrix_add(
            1.0,
            self.bias.reshape((self.cfg.out_dim, 1)),
        );
    }

    activate_fwd(self.cfg.act_kind, &self.tmp_buf, &mut *self.out.out_buf.borrow_mut());
  }

  fn backward(&mut self) {
    assert!(self.in_.batch_size <= self.cfg.batch_sz);
    assert_eq!(self.out.batch_size, self.in_.batch_size);

    activate_bwd(self.cfg.act_kind, &self.tmp_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    self.w_grad.as_view_mut()
      .matrix_prod(
          1.0,
          self.in_.out_buf.borrow().reshape((self.cfg.in_dim, self.in_.batch_size)), Transpose::N,
          self.tmp_grad.reshape((self.cfg.out_dim, self.out.batch_size)), Transpose::T,
          1.0,
      );
    for j in 0 .. self.out.batch_size {
      self.b_grad.reshape_mut((self.cfg.out_dim, 1))
        .matrix_add(
            1.0,
            self.tmp_grad.reshape((self.cfg.out_dim, self.out.batch_size))
              .view((0, j), (self.cfg.out_dim, j+1)),
        );
    }

    if let Some(in_grad) = self.in_.out_grad.as_ref() {
      in_grad.borrow_mut().reshape_mut((self.cfg.in_dim, self.in_.batch_size))
        .matrix_prod(
            1.0,
            self.weights.as_view(), Transpose::N,
            self.tmp_grad.reshape((self.cfg.out_dim, self.out.batch_size)), Transpose::N,
            0.0,
        );
    }
  }
}
