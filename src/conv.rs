use super::{OpCapability};
use common::{CommonOperatorOutput, ActivationKind, ParamInitKind};
use kernels::{activate_fwd, activate_bwd};

use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array4d};
use densearray::linalg::{Transpose};
use nnpack::{NnpackPthreadPool};
use nnpack::ffi::*;
use operator::{InternalOperator, OpPhase, Regularization};
use operator::rw::{ReadAccumulateBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cmp::{min};

#[derive(Clone, Copy)]
pub struct Conv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub kernel_w: usize,
  pub kernel_h: usize,
  pub stride_w: usize,
  pub stride_h: usize,
  pub pad_left: usize,
  pub pad_right: usize,
  pub pad_bot:  usize,
  pub pad_top:  usize,
  pub out_chan: usize,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl Conv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, in_chan) = self.in_dim;
    let out_w = (self.pad_left  + self.pad_right  + in_w - self.kernel_w + 1) / self.stride_w;
    let out_h = (self.pad_bot   + self.pad_top    + in_h - self.kernel_h + 1) / self.stride_h;
    (out_w, out_h, self.out_chan)
  }
}

pub struct Conv2dOperator {
  cfg:      Conv2dOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  weights:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  bias:     Vec<f32>,
  b_grad:   Vec<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  out:      CommonOperatorOutput<f32>,
  pool:     NnpackPthreadPool,
}

impl Conv2dOperator {
  pub fn new(cfg: Conv2dOperatorConfig, cap: OpCapability, prev_op: &InternalOperator<f32, Output=CommonOperatorOutput<f32>>, prev_arm: usize) -> Conv2dOperator {
    let mut bias = Vec::with_capacity(cfg.out_chan);
    unsafe { bias.set_len(cfg.out_chan) };
    let mut b_grad = Vec::with_capacity(cfg.out_chan);
    unsafe { b_grad.set_len(cfg.out_chan) };
    // FIXME
    let mut tmp_buf = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp_buf.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    let mut tmp_grad = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp_grad.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    Conv2dOperator{
      cfg:      cfg,
      in_:      prev_op.output(prev_arm),
      weights:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      bias:     bias,
      b_grad:   b_grad,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      pool:     NnpackPthreadPool::new(1),
    }
  }
}

impl InternalOperator<f32> for Conv2dOperator {
  type Output = CommonOperatorOutput<f32>;

  fn output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn param_len(&self) -> usize {
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan + self.cfg.out_chan
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
        let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        let std = (2.0 / min(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
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
      Regularization::L2{lambda} => {
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

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.w_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, &self.b_grad);
    offset - init_offset
  }

  fn forward(&mut self, _phase: OpPhase) {
    assert!(self.in_.batch_size <= self.cfg.batch_sz);
    self.out.batch_size = self.in_.batch_size;

    // TODO

    activate_fwd(self.cfg.act_kind, &self.tmp_buf, &mut *self.out.out_buf.borrow_mut());
  }

  fn backward(&mut self) {
    assert!(self.in_.batch_size <= self.cfg.batch_sz);
    assert_eq!(self.out.batch_size, self.in_.batch_size);

    activate_bwd(self.cfg.act_kind, &self.tmp_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    // TODO

    if let Some(in_grad) = self.in_.out_grad.as_mut() {
    }
  }
}
