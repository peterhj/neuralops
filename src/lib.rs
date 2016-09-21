#![feature(iter_arith_traits)]

extern crate densearray;
extern crate float;
extern crate iter_utils;
extern crate nnpack;
extern crate operator;
extern crate rng;

extern crate byteorder;
extern crate rand;

use prelude::*;

pub mod affine;
pub mod common;
pub mod conv;
pub mod data;
pub mod graph;
pub mod input;
pub mod kernels;
pub mod loss;
pub mod prelude;
pub mod seq;

pub enum OperatorConfig {
  SimpleInput(SimpleInputOperatorConfig),
  Affine(AffineOperatorConfig),
  Conv2d(Conv2dOperatorConfig),
  SoftmaxNLLClassLoss(ClassLossOperatorConfig),
}
