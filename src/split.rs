use prelude::*;
use common::{CommonResources, CommonOperatorOutput};

use densearray::{Reshape, ReshapeMut};
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{Cell, RefCell};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct SplitOperatorConfig {
  pub batch_sz: usize,
  pub out_arms: usize,
  pub dim:      usize,
}

pub struct NewCopySplitOperator<S> {
  cfg:  SplitOperatorConfig,
  node: OperatorNode,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      CommonOutput,
  out:  Vec<CommonOutput>,
}

impl<S> NewCopySplitOperator<S> {
  pub fn new<InOp>(cfg: SplitOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewCopySplitOperator<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let prev_out = prev_op.borrow()._output(prev_arm);
    let mut out = Vec::with_capacity(cfg.out_arms);
    for arm in 0 .. cfg.out_arms {
      out.push(CommonOutput::new(cfg.batch_sz, cfg.dim, cap));
    }
    Rc::new(RefCell::new(NewCopySplitOperator{
      cfg:  cfg,
      node: OperatorNode::default(),
      in_op:    prev_op,
      in_:      prev_out,
      out:  out,
    }))
  }
}

impl<S> Operator for NewCopySplitOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for NewCopySplitOperator<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    self.out[arm].clone()
  }
}

impl<S> NewDiffOperator<S> for NewCopySplitOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(self.cfg.out_arms as _));
    if self.node.count() == 1 {
      self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
      apply(self);
    } else if self.node.count() == self.cfg.out_arms as _ {
      for _ in 0 .. self.cfg.out_arms {
        self.node.pop(epoch);
      }
    }
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(self.cfg.out_arms as _));
    if self.node.count() == self.cfg.out_arms as _ {
      apply(self);
      self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
      for _ in 0 .. self.cfg.out_arms {
        self.node.pop(epoch);
      }
    }
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.batch_sz);
    for arm in 0 .. self.cfg.out_arms {
      self.out[arm].batch_sz.set(batch_size);
      self.out[arm].buf.borrow_mut()[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.in_.buf.borrow()[ .. batch_size * self.cfg.dim]);
    }
  }

  fn _backward(&mut self) {
    if let Some(in_grad) = self.in_.grad.as_ref() {
      let batch_size = self.out[0].batch_sz.get();
      let mut in_grad = in_grad.borrow_mut();
      in_grad[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.out[0].grad.as_ref().unwrap().borrow()[ .. batch_size * self.cfg.dim]);
      for arm in 1 .. self.cfg.out_arms {
        let arm_batch_size = self.out[arm].batch_sz.get();
        assert_eq!(batch_size, arm_batch_size);
        in_grad.reshape_mut(batch_size * self.cfg.dim)
          .vector_add(1.0, self.out[arm].grad.as_ref().unwrap().borrow().reshape(batch_size * self.cfg.dim));
      }
    }
  }
}
