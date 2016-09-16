use densearray::{Array3d};
use operator::data::{WeightedSample};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::marker::{PhantomData};

pub mod cifar;
pub mod mnist;

#[derive(Clone, Copy)]
pub enum Layout {
  Dim(usize),
  Width,
  Height,
  Depth,
}

#[derive(Clone)]
pub struct ClassSample2d<T> where T: Copy {
  pub input:    Array3d<T>,
  pub layout:   (Layout, Layout, Layout),
  pub label:    Option<i32>,
  pub weight:   Option<f32>,
}

impl<T> WeightedSample for ClassSample2d<T> where T: Copy {
  fn set_weight(&mut self, weight: f32) {
    self.weight = Some(weight);
  }

  fn multiply_weight(&mut self, w2: f32) {
    self.weight = self.weight.map_or(Some(w2), |w1| Some(w1 * w2));
  }
}

pub trait IndexedDataShard<S> {
  fn len(&self) -> usize;
  fn get(&mut self, idx: usize) -> S;
}

pub struct CyclicSamplingDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  rng:      Xorshiftplus128Rng,
  inner:    Shard,
  counter:  usize,
  _marker:  PhantomData<S>,
}

impl<S, Shard> CyclicSamplingDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  pub fn new(inner: Shard) -> CyclicSamplingDataIter<S, Shard> {
    CyclicSamplingDataIter{
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      inner:    inner,
      counter:  0,
      _marker:  PhantomData,
    }
  }

  pub fn len(&self) -> usize {
    self.inner.len()
  }
}

impl<S, Shard> Iterator for CyclicSamplingDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  type Item = S;

  fn next(&mut self) -> Option<S> {
    if self.counter >= self.len() {
      self.counter = 0;
    }
    let idx = self.counter;
    let sample = self.inner.get(idx);
    self.counter += 1;
    Some(sample)
  }
}

pub struct RandomSamplingDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  rng:      Xorshiftplus128Rng,
  inner:    Shard,
  _marker:  PhantomData<S>,
}

impl<S, Shard> RandomSamplingDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  pub fn new(inner: Shard) -> RandomSamplingDataIter<S, Shard> {
    RandomSamplingDataIter{
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      inner:    inner,
      _marker:  PhantomData,
    }
  }

  pub fn len(&self) -> usize {
    self.inner.len()
  }
}

impl<S, Shard> Iterator for RandomSamplingDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  type Item = S;

  fn next(&mut self) -> Option<S> {
    let idx = self.rng.gen_range(0, self.inner.len());
    let sample = self.inner.get(idx);
    Some(sample)
  }
}
