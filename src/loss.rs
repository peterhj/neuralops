use prelude::*;

use densearray::prelude::*;
use operator::prelude::*;

use std::cell::{RefCell};
use std::rc::{Rc};

pub struct L2RegOperator<A> {
  lambda:   f32,
  node:     OperatorNode,
  out:      CommonOutput,
  param:    Rc<ParamBlock<A>>,
}

impl<A> L2RegOperator<A> {
  pub fn new(lambda: f32, param: Rc<ParamBlock<A>>) -> Rc<RefCell<L2RegOperator<A>>> {
    Rc::new(RefCell::new(L2RegOperator{
      lambda:   lambda,
      node:     OperatorNode::default(),
      out:      CommonOutput::new(1, 1, OpCapability::Forward),
      param:    param,
    }))
  }
}

impl<A> Operator for L2RegOperator<A> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<A> CommonOperator for L2RegOperator<A> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<A, S> DiffOperatorData<S> for L2RegOperator<A> {
}

impl<A, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for L2RegOperator<A> {
}

impl<A, S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for L2RegOperator<A> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    // FIXME(20161123): pass through to parameter.
    //self.param.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    // FIXME(20161123): pass through to parameter.
    //self.param.borrow_mut()._traverse_fwd(epoch, apply);
    self.node.pop(epoch);
  }

  default fn _forward(&mut self, _phase: OpPhase) {
    unimplemented!();
  }

  default fn _backward(&mut self) {
    unimplemented!();
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for L2RegOperator<Array1d<f32>> {
  fn _forward(&mut self, phase: OpPhase) {
    let param_norm = self.param.val.as_ref().as_view().l2_norm();
    let reg_loss = 0.5 * self.lambda * param_norm * param_norm;
    self.out.buf.borrow_mut()[0] = reg_loss;
  }

  fn _backward(&mut self) {
    self.param.grad.as_mut().as_view_mut()
      .vector_add(self.lambda, self.param.val.as_ref().as_view());
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for L2RegOperator<Array2d<f32>> {
  fn _forward(&mut self, phase: OpPhase) {
    let param_sz = self.param.val.as_ref().dim().flat_len();
    let param_norm = self.param.val.as_ref().as_view().reshape(param_sz).l2_norm();
    let reg_loss = 0.5 * self.lambda * param_norm * param_norm;
    self.out.buf.borrow_mut()[0] = reg_loss;
  }

  fn _backward(&mut self) {
    let param_sz = self.param.val.as_ref().dim().flat_len();
    self.param.grad.as_mut().as_view_mut().reshape_mut(param_sz)
      .vector_add(self.lambda, self.param.val.as_ref().as_view().reshape(param_sz));
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for L2RegOperator<Array4d<f32>> {
  fn _forward(&mut self, phase: OpPhase) {
    let param_sz = self.param.val.as_ref().dim().flat_len();
    let param_norm = self.param.val.as_ref().as_view().reshape(param_sz).l2_norm();
    let reg_loss = 0.5 * self.lambda * param_norm * param_norm;
    self.out.buf.borrow_mut()[0] = reg_loss;
  }

  fn _backward(&mut self) {
    let param_sz = self.param.val.as_ref().dim().flat_len();
    self.param.grad.as_mut().as_view_mut().reshape_mut(param_sz)
      .vector_add(self.lambda, self.param.val.as_ref().as_view().reshape(param_sz));
  }
}
