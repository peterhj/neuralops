#![feature(iter_arith_traits)]

extern crate densearray;
extern crate float;
extern crate iter_utils;
extern crate nnpack;
extern crate operator;
extern crate rng;

extern crate byteorder;
extern crate memmap;
extern crate rand;

/*use affine::{AffineOperatorConfig};
use input::{SimpleInputOperatorConfig};
use loss::{ClassLossOperatorConfig};*/
use prelude::*;

pub mod affine;
pub mod common;
pub mod conv;
pub mod data;
pub mod input;
pub mod kernels;
pub mod loss;
pub mod prelude;
pub mod seq;

#[derive(Clone, Copy)]
pub enum OpCapability {
  Forward,
  Backward,
  RForward,
  RBackward,
}

impl OpCapability {
  pub fn enable_backward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      _ => true,
    }
  }

  pub fn enable_r_forward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      OpCapability::Backward => false,
      _ => true,
    }
  }

  pub fn enable_r_backward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      OpCapability::Backward => false,
      OpCapability::RForward => false,
      _ => true,
    }
  }
}

pub enum OperatorConfig {
  SimpleInput(SimpleInputOperatorConfig),
  Affine(AffineOperatorConfig),
  Conv2d(Conv2dOperatorConfig),
  SoftmaxNLLClassLoss(ClassLossOperatorConfig),
}
