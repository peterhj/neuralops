use common::*;
//use kernels::softmax::{SoftmaxKernel};

use densearray::{AsView, AsViewMut, Array1d, Array2d, Array4d};
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
use std::rc::{Rc};

pub struct ParamBlock<A> {
  node:     OperatorNode,
  pub val:  A,
  pub grad: Option<A>,
}

impl ParamBlock<Array1d<f32>> {
  pub fn new(dim: usize, cap: OpCapability) -> Rc<RefCell<ParamBlock<Array1d<f32>>>> {
    Rc::new(RefCell::new(ParamBlock{
      node:     OperatorNode::default(),
      val:      Array1d::zeros(dim),
      grad:     match cap {
        OpCapability::Forward   => None,
        OpCapability::Backward  => Some(Array1d::zeros(dim)),
        _ => unimplemented!(),
      },
    }))
  }
}

impl ParamBlock<Array2d<f32>> {
  pub fn new(dim: (usize, usize), cap: OpCapability) -> Rc<RefCell<ParamBlock<Array2d<f32>>>> {
    Rc::new(RefCell::new(ParamBlock{
      node:     OperatorNode::default(),
      val:      Array2d::zeros(dim),
      grad:     match cap {
        OpCapability::Forward   => None,
        OpCapability::Backward  => Some(Array2d::zeros(dim)),
        _ => unimplemented!(),
      },
    }))
  }
}

impl ParamBlock<Array4d<f32>> {
  pub fn new(dim: (usize, usize, usize, usize), cap: OpCapability) -> Rc<RefCell<ParamBlock<Array4d<f32>>>> {
    Rc::new(RefCell::new(ParamBlock{
      node:     OperatorNode::default(),
      val:      Array4d::zeros(dim),
      grad:     match cap {
        OpCapability::Forward   => None,
        OpCapability::Backward  => Some(Array4d::zeros(dim)),
        _ => unimplemented!(),
      },
    }))
  }
}

impl<A> Operator for ParamBlock<A> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<A, S> NewDiffOperator<S> for ParamBlock<A> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    // Do nothing.
  }

  fn _backward(&mut self) {
    // Do nothing.
  }
}

pub struct L2RegOperator<A> {
  lambda:   f32,
  node:     OperatorNode,
  out:      CommonOutput,
  param:    Rc<RefCell<ParamBlock<A>>>,
}

impl<A> L2RegOperator<A> {
  pub fn new(lambda: f32, param: Rc<RefCell<ParamBlock<A>>>) -> Rc<RefCell<L2RegOperator<A>>> {
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

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<A> CommonOperator for L2RegOperator<A> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<A, S> NewDiffOperator<S> for L2RegOperator<A> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    self.param.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.param.borrow_mut()._traverse_fwd(epoch, apply);
  }

  default fn _forward(&mut self, _phase: OpPhase) {
    unimplemented!();
  }

  default fn _backward(&mut self) {
    unimplemented!();
  }
}

impl<S> NewDiffOperator<S> for L2RegOperator<Array1d<f32>> {
  fn _forward(&mut self, phase: OpPhase) {
    let param = self.param.borrow();
    let param_norm = param.val.as_view().l2_norm();
    let reg_loss = 0.5 * self.lambda * param_norm * param_norm;
    self.out.buf.borrow_mut()[0] = reg_loss;
  }

  fn _backward(&mut self) {
    let mut param = &mut *self.param.borrow_mut();
    param.grad.as_mut().unwrap().as_view_mut()
      .vector_add(self.lambda, param.val.as_view());
  }
}
