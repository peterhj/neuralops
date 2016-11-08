use prelude::*;

use densearray::prelude::*;
use operator::prelude::*;

use std::cell::{RefCell};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct JoinOperatorConfig {
  pub batch_sz: usize,
  pub in_arms:  usize,
  pub dim:      usize,
}

pub struct NewAddJoinOperator<S> {
  cfg:  JoinOperatorConfig,
  node: OperatorNode,
  in_ops:   Vec<Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>>,
  in_:  Vec<CommonOutput>,
  out:  CommonOutput,
}

impl<S> NewAddJoinOperator<S> {
  pub fn new(cfg: JoinOperatorConfig, cap: OpCapability) -> Rc<RefCell<NewAddJoinOperator<S>>> {
    let in_ops = Vec::with_capacity(cfg.in_arms);
    let in_ = Vec::with_capacity(cfg.in_arms);
    Rc::new(RefCell::new(NewAddJoinOperator{
      cfg:  cfg,
      node: OperatorNode::default(),
      in_ops:   in_ops,
      in_:      in_,
      out:  CommonOutput::new(cfg.batch_sz, cfg.dim, cap),
    }))
  }

  pub fn append_input<InOp>(&mut self, in_op: Rc<RefCell<InOp>>, in_arm: usize) where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    assert!(self.in_ops.len() < self.cfg.in_arms);
    assert_eq!(self.in_ops.len(), self.in_.len());
    let out = in_op.borrow()._output(in_arm);
    self.in_ops.push(in_op);
    self.in_.push(out);
  }
}

impl<S> Operator for NewAddJoinOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for NewAddJoinOperator<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for NewAddJoinOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    for in_op in self.in_ops.iter() {
      in_op.borrow_mut()._traverse_fwd(epoch, apply);
    }
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    for in_op in self.in_ops.iter() {
      in_op.borrow_mut()._traverse_bwd(epoch, apply);
    }
    self.node.pop(epoch);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_[0].batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);
    self.out.buf.borrow_mut()[ .. batch_size * self.cfg.dim]
      .copy_from_slice(&self.in_[0].buf.borrow()[ .. batch_size * self.cfg.dim]);
    for arm in 1 .. self.cfg.in_arms {
      let arm_batch_size = self.in_[arm].batch_sz.get();
      assert_eq!(batch_size, arm_batch_size);
      self.out.buf.borrow_mut().reshape_mut(batch_size * self.cfg.dim)
        .vector_add(1.0, self.in_[arm].buf.borrow().reshape(batch_size * self.cfg.dim));
    }
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    for arm in 0 .. self.cfg.in_arms {
      if let Some(in_grad) = self.in_[arm].grad.as_ref() {
        let mut in_grad = in_grad.borrow_mut();
        in_grad[ .. batch_size * self.cfg.dim]
          .copy_from_slice(&self.out.grad.as_ref().unwrap().borrow()[ .. batch_size * self.cfg.dim]);
      }
    }
  }
}
