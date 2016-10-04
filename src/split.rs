use common::{CommonResources, CommonOperatorOutput};

use densearray::{Reshape, ReshapeMut};
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

#[derive(Clone, Copy, Debug)]
pub struct SplitOperatorConfig {
  pub batch_sz: usize,
  pub out_arms: usize,
  pub dim:      usize,
}

pub struct CopySplitOperator {
  cfg:  SplitOperatorConfig,
  in_:  CommonOperatorOutput<f32>,
  out:  Vec<CommonOperatorOutput<f32>>,
}

impl CopySplitOperator {
  pub fn new(cfg: SplitOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> CopySplitOperator {
    let mut out = Vec::with_capacity(cfg.out_arms);
    for arm in 0 .. cfg.out_arms {
      out.push(CommonOperatorOutput::new(cfg.batch_sz, cfg.dim, cap));
    }
    CopySplitOperator{
      cfg:  cfg,
      in_:  prev_op._output(prev_arm),
      out:  out,
    }
  }
}

impl DiffOperator<f32> for CopySplitOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, arm: usize) -> CommonOperatorOutput<f32> {
    self.out[arm].clone()
  }

  fn forward(&mut self, _phase: OpPhase) {
    let batch_size = *self.in_.batch_size.borrow();
    assert!(batch_size <= self.cfg.batch_sz);
    let batch_loss = *self.in_.out_loss.borrow();
    for arm in 0 .. self.cfg.out_arms {
      *self.out[arm].batch_size.borrow_mut() = batch_size;
      *self.out[arm].out_loss.borrow_mut() = batch_loss;
      self.out[arm].out_buf.borrow_mut()[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.in_.out_buf.borrow()[ .. batch_size * self.cfg.dim]);
    }
  }

  fn backward(&mut self) {
    if let Some(in_grad) = self.in_.out_grad.as_ref() {
      let batch_size = *self.out[0].batch_size.borrow();
      let mut in_grad = in_grad.borrow_mut();
      in_grad[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.out[0].out_grad.as_ref().unwrap().borrow()[ .. batch_size * self.cfg.dim]);
      for arm in 1 .. self.cfg.out_arms {
        let arm_batch_size = *self.out[arm].batch_size.borrow();
        assert_eq!(batch_size, arm_batch_size);
        in_grad.reshape_mut(batch_size * self.cfg.dim)
          .vector_add(1.0, self.out[arm].out_grad.as_ref().unwrap().borrow().reshape(batch_size * self.cfg.dim));
      }
    }
  }
}
