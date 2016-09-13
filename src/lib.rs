extern crate densearray;
extern crate float;
extern crate iter_utils;
extern crate nnpack;
extern crate operator;
extern crate rng;

extern crate byteorder;
extern crate memmap;
extern crate rand;

use affine::{AffineOperatorConfig};
use input::{SimpleInputOperatorConfig};
use loss::{ClassLossOperatorConfig};

pub mod affine;
pub mod common;
pub mod conv;
pub mod data;
pub mod input;
pub mod kernels;
pub mod loss;
pub mod seq;

pub enum OpConfig {
  SimpleInput(SimpleInputOperatorConfig),
  Affine(AffineOperatorConfig),
  SoftmaxNLLClassLoss(ClassLossOperatorConfig),
}
