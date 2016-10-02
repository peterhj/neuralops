#![feature(iter_arith_traits)]

extern crate densearray;
extern crate float;
extern crate iter_utils;
extern crate nnpack;
extern crate operator;
extern crate rng;
extern crate sharedmem;

extern crate byteorder;
extern crate rand;

use prelude::*;

pub mod affine;
pub mod checkpoint;
//pub mod class_loss;
pub mod common;
pub mod conv;
pub mod data;
pub mod graph;
pub mod input;
pub mod kernels;
pub mod loss;
pub mod prelude;
pub mod regress_loss;
pub mod seq;
pub mod softmax;

mod ops;
