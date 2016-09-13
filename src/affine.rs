use common::{CommonOperatorOutput, ActivationKind, ParamInitKind};
use kernels::{activate_fwd, activate_bwd};

use densearray::{Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array2d};
use densearray::linalg::{Transpose};
use operator::{Operator, InternalOperator, OpPhase};
use operator::rw::{ReadAccumulateBuffer, AccumulateBuffer};

use rand::{Rng};

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

impl InternalOperator<f32> for AffineOperator {
  type Output = CommonOperatorOutput<f32>;

  fn output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn init_param<R>(&mut self, rng: &mut R) where R: Rng {
    match self.cfg.w_init {
      ParamInitKind::Disabled => {
        panic!();
      }
      ParamInitKind::Uniform{lo, hi} => {
        for e in self.weights.as_mut_slice().iter_mut() {
          // TODO
        }
      }
      ParamInitKind::Normal{mean, std} => {
        for e in self.weights.as_mut_slice().iter_mut() {
          // TODO
        }
      }
      _ => unimplemented!(),
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

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.weights.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, &self.bias);
    offset - init_offset
  }

  fn forward(&mut self, _phase: OpPhase) {
    assert!(self.in_.batch_size <= self.cfg.batch_sz);
    self.out.batch_size = self.in_.batch_size;

    self.tmp_buf.reshape_mut((self.cfg.out_dim, self.in_.batch_size))
      .matrix_prod(
          1.0,
          self.weights.as_view(), Transpose::T,
          self.in_.out_buf.borrow().reshape((self.cfg.in_dim, self.in_.batch_size)), Transpose::N,
          0.0,
      );
    for j in 0 .. self.out.batch_size {
      self.tmp_buf.reshape_mut((self.cfg.out_dim, self.in_.batch_size))
        .view_mut((0, j), (self.cfg.out_dim, j+1))
        .matrix_sum(
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
        .matrix_sum(
            1.0,
            self.tmp_grad.reshape((self.cfg.out_dim, self.out.batch_size))
              .view((0, j), (self.cfg.out_dim, j+1)),
        );
    }

    if let Some(in_grad) = self.in_.out_grad.as_mut() {
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
