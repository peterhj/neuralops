use prelude::*;
use kernels::ffi::*;

use densearray::prelude::*;
use operator::prelude::*;

use std::cell::{RefCell};
use std::cmp::{max};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct DummyOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub out_dim:  (usize, usize, usize),
}

pub struct DummyOperator<S, IoBuf: ?Sized> {
  cfg:      DummyOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
}

impl<S, IoBuf: ?Sized> DummyOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: DummyOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<DummyOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(DummyOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim.flat_len(), cap),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DummyOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for DummyOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DummyOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DummyOperator<S, IoBuf> {
  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.batch_sz);
    self.out.batch_sz.set(batch_size);
  }

  fn _backward(&mut self) {
  }
}
