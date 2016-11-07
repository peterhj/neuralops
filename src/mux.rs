use prelude::*;

//use densearray::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
use std::cmp::{min};
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
  batchszs: Vec<usize>,
  batch:    usize,
  offset:   usize,
  fwdprop:  bool,
}

impl<S> BatchMuxOperator<S> {
  pub fn new<InOp>(cfg: BatchMuxOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<BatchMuxOperator<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let prev_out = prev_op.borrow()._output(prev_arm);
    let out = CommonOutput::new(cfg.out_batch_sz, cfg.dim, cap);
    let num_out_batches = (cfg.in_batch_sz + cfg.out_batch_sz - 1) / cfg.out_batch_sz;
    let mut batchszs = Vec::with_capacity(num_out_batches);
    batchszs.resize(num_out_batches, 0);
    Rc::new(RefCell::new(BatchMuxOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      prev_out,
      out:      out,
      batchszs: batchszs,
      batch:    0,
      offset:   0,
      fwdprop:  true,
    }))
  }

  fn _reset_fwdprop(&mut self) {
    self.fwdprop = true;
  }

  fn _set_fwdprop(&mut self, fwdprop: bool) {
    self.fwdprop = fwdprop;
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
    self.node.push(epoch);
    let num_out_batches = (self.cfg.in_batch_sz + self.cfg.out_batch_sz - 1) / self.cfg.out_batch_sz;
    assert!(self.node.limit(num_out_batches as _));
    if self.node.count() == 1 {
      if self.fwdprop {
        self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
      }
      self.batch = 0;
      self.offset = 0;
    }
    let in_batch_size = self.in_.batch_sz.get();
    let out_batch_size = min(in_batch_size, (self.batch + 1) * self.cfg.out_batch_sz) - min(in_batch_size, self.batch * self.cfg.out_batch_sz);
    self.batchszs[self.batch] = out_batch_size;
    apply(self);
    self.batch += 1;
    self.offset += out_batch_size;
    if self.node.count() == num_out_batches as _ {
      for _ in 0 .. num_out_batches {
        self.node.pop(epoch);
      }
    }
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    let num_out_batches = (self.cfg.in_batch_sz + self.cfg.out_batch_sz - 1) / self.cfg.out_batch_sz;
    assert!(self.node.limit(num_out_batches as _));
    if self.node.count() == 1 {
      self.batch = 0;
      self.offset = 0;
    }
    let batch_size = self.batchszs[self.batch];
    apply(self);
    self.batch += 1;
    self.offset += batch_size;
    if self.node.count() == num_out_batches as _ {
      self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
      for _ in 0 .. num_out_batches {
        self.node.pop(epoch);
      }
    }
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let in_batch_size = self.in_.batch_sz.get();
    assert!(in_batch_size <= self.cfg.in_batch_sz);
    let batch_size = self.batchszs[self.batch];
    self.out.buf.borrow_mut()[ .. batch_size * self.cfg.dim]
      .copy_from_slice(&self.in_.buf.borrow()[self.offset * self.cfg.dim .. (self.offset + batch_size) * self.cfg.dim]);
  }

  fn _backward(&mut self) {
    if let Some(in_grad) = self.in_.grad.as_ref() {
      let batch_size = self.batchszs[self.batch];
      in_grad.borrow_mut()[self.offset * self.cfg.dim .. (self.offset + batch_size) * self.cfg.dim]
        .copy_from_slice(&self.out.grad.as_ref().unwrap().borrow()[ .. batch_size * self.cfg.dim]);
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
  batch:    usize,
  offset:   usize,
  fwdepoch: u64,
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
      batch:    0,
      offset:   0,
      fwdepoch: 0,
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
    self.node.push(epoch);
    assert!(self.node.limit(1));
    let num_in_batches = (self.cfg.out_batch_sz + self.cfg.in_batch_sz - 1) / self.cfg.in_batch_sz;
    self.offset = 0;
    for batch in 0 .. num_in_batches {
      let batch_size = self.in_.batch_sz.get();
      assert!(batch_size <= self.cfg.in_batch_sz);
      self.batchszs[batch] = batch_size;
      self.batch = batch;
      self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
      apply(self);
      self.offset += batch_size;
    }
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    let num_in_batches = (self.cfg.out_batch_sz + self.cfg.in_batch_sz - 1) / self.cfg.in_batch_sz;
    self.offset = 0;
    for batch in 0 .. num_in_batches {
      let batch_size = self.batchszs[batch];
      self.batch = batch;
      apply(self);
      self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
      self.offset += batch_size;
    }
    self.node.pop(epoch);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.in_batch_sz);
    self.out.buf.borrow_mut()[self.offset * self.cfg.dim .. (self.offset + batch_size) * self.cfg.dim]
      .copy_from_slice(&self.in_.buf.borrow()[ .. batch_size * self.cfg.dim]);
  }

  fn _backward(&mut self) {
    if let Some(in_grad) = self.in_.grad.as_ref() {
      if self.offset == 0 {
        self.mux_op.borrow_mut()._set_fwdprop(false);
        self.fwdepoch = self.node._next();
      }
      self.in_op.borrow_mut()._traverse_fwd(self.fwdepoch, &mut |op| op._forward(OpPhase::Learning));
      let batch_size = self.batchszs[self.batch];
      in_grad.borrow_mut()[ .. batch_size * self.cfg.dim]
        .copy_from_slice(&self.out.grad.as_ref().unwrap().borrow()[self.offset * self.cfg.dim .. (self.offset + batch_size) * self.cfg.dim]);
      let num_in_batches = (self.cfg.out_batch_sz + self.cfg.in_batch_sz - 1) / self.cfg.in_batch_sz;
      if self.batch == num_in_batches - 1 {
        self.mux_op.borrow_mut()._reset_fwdprop();
      }
    }
  }
}

// FIXME(20161105)
pub struct BatchDemuxLoss<S, Loss> {
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  loss:     Rc<RefCell<Loss>>,
}
