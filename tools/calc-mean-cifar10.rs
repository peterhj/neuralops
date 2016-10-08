extern crate densearray;
extern crate neuralops;
extern crate operator;
extern crate rand;
extern crate rng;

use densearray::{Reshape, ReshapeMut};
use neuralops::prelude::*;
use neuralops::archs::*;
use neuralops::data::{CyclicDataIter};
use neuralops::data::cifar::{CifarFlavor, CifarDataShard};
use operator::prelude::*;
use operator::data::{SampleExtractInput};

use rand::{thread_rng};
use std::path::{PathBuf};

fn main() {
  let mut train_data =
      CyclicDataIter::new(
      CifarDataShard::new(
          CifarFlavor::Cifar10,
          PathBuf::from("datasets/cifar10/train.bin"),
      ));
  let mut mean = Vec::with_capacity(32 * 32 * 3);
  mean.resize(32 * 32 * 3, 0.0_f32);
  let mut buf = Vec::with_capacity(32 * 32 * 3);
  buf.resize(32 * 32 * 3, 0.0_f32);
  let num_samples = train_data.len();
  for (idx, sample) in train_data.take(num_samples).enumerate() {
    sample.extract_input(&mut buf);
    mean.reshape_mut(32 * 32 * 3).vector_add(1.0, buf.reshape(32 * 32 * 3));
    if (idx+1) % 100 == 0 {
      println!("DEBUG: processed {} samples", idx+1);
    }
  }
  for m in mean.iter_mut() {
    *m /= num_samples as f32;
  }
  let mut pixel_mean = vec![0.0, 0.0, 0.0];
  for xy in 0 .. 32 * 32 {
    pixel_mean[0] += mean[xy];
    pixel_mean[1] += mean[xy+1024];
    pixel_mean[2] += mean[xy+2048];
  }
  pixel_mean[0] /= 1024.0;
  pixel_mean[1] /= 1024.0;
  pixel_mean[2] /= 1024.0;
  println!("DEBUG: {:e} {:e} {:e}", pixel_mean[0], pixel_mean[1], pixel_mean[2]);
}
