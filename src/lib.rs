//#![feature(iter_arith_traits)]
//#![feature(conservative_impl_trait)]
//#![feature(reflect_marker)]
#![feature(specialization)]

extern crate densearray;
extern crate float;
extern crate iter_utils;
#[cfg(feature = "mkl")]
extern crate mkl_dnn;
//extern crate neuralops_kernels;
//extern crate neuralops_omp_kernels;
//extern crate nnpack;
extern crate operator;
extern crate rng;
extern crate sharedmem;
extern crate stb_image;
extern crate turbojpeg;
//extern crate typemap;
extern crate varraydb;

extern crate byteorder;
extern crate libc;
extern crate rand;

pub mod affine;
pub mod archs;
pub mod checkpoint;
pub mod class_loss;
pub mod common;
pub mod conv;
#[cfg(feature = "mkl")]
pub mod conv_mkl;
#[cfg(not(feature = "mkl"))]
pub mod conv_nnpack;
pub mod data;
pub mod deconv;
pub mod input;
pub mod join;
pub mod kernels;
pub mod loss;
pub mod mux;
pub mod param;
pub mod pool;
pub mod prelude;
pub mod regress_loss;
pub mod split;
pub mod util;
