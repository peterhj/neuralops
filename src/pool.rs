use common::{CommonResources, CommonOperatorOutput, PoolKind};
use kernels::pool::*;

use densearray::{ArrayIndex};
//use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array4d};
//use densearray::linalg::{Transpose};
//use nnpack::{NnpackHandle, NnpackPthreadPool};
//use nnpack::ffi::*;
use operator::prelude::*;
//use operator::rw::{ReadAccumulateBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

//use std::cmp::{max, min};
//use std::ptr::{null_mut};
//use std::rc::{Rc};

/*#[derive(Clone, Copy, Debug)]
pub enum PoolKind {
  Average,
  Max,
}*/

#[derive(Clone, Copy, Debug)]
pub struct Pool2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub pool_w:   usize,
  pub pool_h:   usize,
  pub stride_w: usize,
  pub stride_h: usize,
  pub pad_w:    usize,
  pub pad_h:    usize,
  pub kind:     PoolKind,
}

impl Pool2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    unimplemented!();
  }
}

pub struct Pool2dOperator {
  cfg:      Pool2dOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  out:      CommonOperatorOutput<f32>,
  //nnp_h:    NnpackHandle,
  //nnp_pool: Rc<NnpackPthreadPool>,
}

impl Pool2dOperator {
  pub fn new(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> Pool2dOperator {
    Pool2dOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
    }
  }
}

impl DiffOperator<f32> for Pool2dOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn forward(&mut self, _phase: OpPhase) {
    match self.cfg.kind {
      PoolKind::Average => {
        if self.cfg.pad_w == 0 && self.cfg.pad_h == 0 {
          if self.cfg.pool_w == 2 && self.cfg.pool_h == 2 &&
              self.cfg.stride_w == 2 && self.cfg.stride_h == 2
          {
            unimplemented!();
          } else if self.cfg.pool_w == self.cfg.stride_w &&
              self.cfg.pool_h == self.cfg.stride_h
          {
            unimplemented!();
          } else {
            unimplemented!();
          }
        } else {
          unimplemented!();
        }
      }
      _ => unimplemented!(),
    }
  }

  fn backward(&mut self) {
    unimplemented!();
  }
}
