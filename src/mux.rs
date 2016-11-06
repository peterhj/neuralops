use prelude::*;

//use densearray::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct BatchMuxOperatorConfig {
  pub in_batch_sz:  usize,
  pub out_batch_sz: usize,
  //pub num_batches:  usize,
  pub dim:          usize,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchDemuxOperatorConfig {
  pub in_batch_sz:  usize,
  pub out_batch_sz: usize,
  //pub num_batches:  usize,
  pub dim:          usize,
}

pub struct BatchMuxOperator<S> {
  cfg:      BatchMuxOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
}

impl<S> BatchMuxOperator<S> {
  pub fn new<InOp>(cfg: BatchMuxOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<BatchMuxOperator<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let prev_out = prev_op.borrow()._output(prev_arm);
    let out = CommonOutput::new(cfg.out_batch_sz, cfg.dim, cap);
    Rc::new(RefCell::new(BatchMuxOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      prev_out,
      out:      out,
    }))
  }
}

impl<S> Operator for BatchMuxOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for BatchMuxOperator<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for BatchMuxOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    unimplemented!();
    /*assert!(self.node.limit(self.cfg.out_arms as _));
    if self.node.count() == 1 {
      self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
      apply(self);
    }*/
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    unimplemented!();
    /*assert!(self.node.limit(self.cfg.out_arms as _));
    if self.node.count() == self.cfg.out_arms as _ {
      apply(self);
      self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    }*/
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.in_batch_sz);
    unimplemented!();
    /*for arm in 0 .. self.cfg.out_arms {
      self.out[arm].batch_sz.set(batch_size);
      self.out[arm].buf.borrow_mut()[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.in_.buf.borrow()[ .. batch_size * self.cfg.dim]);
    }*/
  }

  fn _backward(&mut self) {
    if let Some(in_grad) = self.in_.grad.as_ref() {
      unimplemented!();
      /*let batch_size = self.out[0].batch_sz.get();
      let mut in_grad = in_grad.borrow_mut();
      in_grad[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.out[0].grad.as_ref().unwrap().borrow()[ .. batch_size * self.cfg.dim]);
      for arm in 1 .. self.cfg.out_arms {
        let arm_batch_size = self.out[arm].batch_sz.get();
        assert_eq!(batch_size, arm_batch_size);
        in_grad.reshape_mut(batch_size * self.cfg.dim)
          .vector_add(1.0, self.out[arm].grad.as_ref().unwrap().borrow().reshape(batch_size * self.cfg.dim));
      }*/
    }
  }
}

pub struct BatchDemuxOperator<S> {
  cfg:      BatchDemuxOperatorConfig,
  node:     OperatorNode,
  mux_op:   Rc<RefCell<BatchMuxOperator<S>>>,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  batchszs: Vec<usize>,
  offset:   usize,
}

impl<S> BatchDemuxOperator<S> {
  pub fn new<InOp>(cfg: BatchDemuxOperatorConfig, cap: OpCapability, mux_op: Rc<RefCell<BatchMuxOperator<S>>>, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<BatchDemuxOperator<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    assert_eq!(cfg.out_batch_sz, mux_op.borrow().cfg.out_batch_sz);
    let prev_out = prev_op.borrow()._output(prev_arm);
    let out = CommonOutput::new(cfg.out_batch_sz, cfg.dim, cap);
    let num_in_batches = (cfg.out_batch_sz + cfg.in_batch_sz - 1) / cfg.in_batch_sz;
    let mut batchszs = Vec::with_capacity(num_in_batches);
    batchszs.resize(num_in_batches, 0);
    Rc::new(RefCell::new(BatchDemuxOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      mux_op:   mux_op,
      in_op:    prev_op,
      in_:      prev_out,
      out:      out,
      batchszs: batchszs,
      offset:   0,
    }))
  }
}

impl<S> Operator for BatchDemuxOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for BatchDemuxOperator<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for BatchDemuxOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    let num_in_batches = (self.cfg.out_batch_sz + self.cfg.in_batch_sz - 1) / self.cfg.in_batch_sz;
    self.offset = 0;
    for batch in 0 .. num_in_batches {
      let batch_size = self.in_.batch_sz.get();
      assert!(batch_size <= self.cfg.in_batch_sz);
      self.batchszs[batch] = batch_size;
      self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
      apply(self);
      self.offset += batch_size;
    }
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    let num_in_batches = (self.cfg.out_batch_sz + self.cfg.in_batch_sz - 1) / self.cfg.in_batch_sz;
    self.offset = 0;
    for batch in 0 .. num_in_batches {
      let batch_size = self.batchszs[batch];
      apply(self);
      self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
      self.offset += batch_size;
    }
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.in_batch_sz);
    unimplemented!();
    /*for arm in 0 .. self.cfg.out_arms {
      self.out[arm].batch_sz.set(batch_size);
      self.out[arm].buf.borrow_mut()[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.in_.buf.borrow()[ .. batch_size * self.cfg.dim]);
    }*/
  }

  fn _backward(&mut self) {
    if let Some(in_grad) = self.in_.grad.as_ref() {
      unimplemented!();
      /*let batch_size = self.out[0].batch_sz.get();
      let mut in_grad = in_grad.borrow_mut();
      in_grad[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.out[0].grad.as_ref().unwrap().borrow()[ .. batch_size * self.cfg.dim]);
      for arm in 1 .. self.cfg.out_arms {
        let arm_batch_size = self.out[arm].batch_sz.get();
        assert_eq!(batch_size, arm_batch_size);
        in_grad.reshape_mut(batch_size * self.cfg.dim)
          .vector_add(1.0, self.out[arm].grad.as_ref().unwrap().borrow().reshape(batch_size * self.cfg.dim));
      }*/
    }
  }
}

// FIXME(20161105)
pub struct BatchDemuxLoss<S, Loss> {
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  loss:     Rc<RefCell<Loss>>,
}
