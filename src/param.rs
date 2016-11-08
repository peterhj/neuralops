//use prelude::*;

//use densearray::prelude::*;
use operator::prelude::*;

use std::cell::{RefCell};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};

pub struct ParamBlock<A> {
  node:         OperatorNode,
  pub val:      A,
  pub grad:     Option<A>,
  pub r_dir:    Option<A>,
  //pub r_grad:   Option<A>,
}

impl<A> Deref for ParamBlock<A> {
  type Target = A;

  fn deref(&self) -> &A {
    &self.val
  }
}

impl<A> DerefMut for ParamBlock<A> {
  fn deref_mut(&mut self) -> &mut A {
    &mut self.val
  }
}

impl<A> ParamBlock<A> {
  pub fn new<F>(cap: OpCapability, mut builder: F) -> Rc<RefCell<ParamBlock<A>>> where F: FnMut() -> A {
    Rc::new(RefCell::new(ParamBlock{
      node:     OperatorNode::default(),
      val:      builder(),
      grad:     if cap.enable_backward() {
        Some(builder())
      } else {
        None
      },
      r_dir:    if cap.enable_r_forward() {
        Some(builder())
      } else {
        None
      },
    }))
  }

  pub fn grad(&self) -> &A {
    self.grad.as_ref().unwrap()
  }

  pub fn grad_mut(&mut self) -> &mut A {
    self.grad.as_mut().unwrap()
  }

  pub fn r_dir(&self) -> &A {
    self.r_dir.as_ref().unwrap()
  }

  pub fn r_dir_mut(&mut self) -> &mut A {
    self.r_dir.as_mut().unwrap()
  }
}

/*impl ParamBlock<Array1d<f32>> {
  pub fn new(dim: usize, cap: OpCapability) -> Rc<RefCell<ParamBlock<Array1d<f32>>>> {
    Rc::new(RefCell::new(ParamBlock{
      node:     OperatorNode::default(),
      val:      Array1d::zeros(dim),
      grad:     match cap {
        OpCapability::Forward   => None,
        OpCapability::Backward  => Some(Array1d::zeros(dim)),
        _ => unimplemented!(),
      },
      r_dir:    match cap {
        OpCapability::Forward   => None,
        OpCapability::Backward  => None,
        OpCapability::RForward  => Some(Array1d::zeros(dim)),
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
      r_dir:    match cap {
        OpCapability::Forward   => None,
        OpCapability::Backward  => None,
        OpCapability::RForward  => Some(Array2d::zeros(dim)),
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
      r_dir:    match cap {
        OpCapability::Forward   => None,
        OpCapability::Backward  => None,
        OpCapability::RForward  => Some(Array4d::zeros(dim)),
        _ => unimplemented!(),
      },
    }))
  }
}*/

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
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    // Do nothing.
  }

  fn _backward(&mut self) {
    // Do nothing.
  }
}
