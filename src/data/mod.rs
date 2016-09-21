use densearray::{Array3d};
use operator::data::{SampleExtractInput, SampleClass, SampleWeight};
use operator::memory::{SharedSlice};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::marker::{PhantomData};

pub mod cifar;
pub mod mnist;

#[derive(Clone, Copy)]
pub enum Shape {
  Dim(usize, usize),
  Width(usize),
  Height(usize),
  Depth(usize),
}

#[derive(Clone)]
pub struct ClassSample2d<T> where T: Copy {
  pub input:    Array3d<T>,
  pub layout:   (Shape, Shape, Shape),
  pub label:    Option<u32>,
  pub weight:   Option<f32>,
}

impl<T> SampleClass for ClassSample2d<T> where T: Copy {
  fn class(&self) -> Option<u32> {
    self.label
  }
}

impl<T> SampleWeight for ClassSample2d<T> where T: Copy {
  fn weight(&self) -> Option<f32> {
    self.weight
  }

  fn mix_weight(&mut self, w: f32) {
    self.weight = Some(self.weight.map_or(w, |w0| w0 * w));
  }
}

#[derive(Clone)]
pub struct SharedClassSample<T> where T: Copy {
  pub input:    SharedSlice<T>,
  pub label:    Option<u32>,
  pub weight:   Option<f32>,
}

impl SampleExtractInput<u8> for SharedClassSample<u8> {
  fn extract_input(&self, output: &mut [u8]) {
    let input: &[u8] = &*self.input;
    output.copy_from_slice(input);
  }
}

impl SampleExtractInput<f32> for SharedClassSample<u8> {
  fn extract_input(&self, output: &mut [f32]) {
    let input: &[u8] = &*self.input;
    for i in 0 .. input.len() {
      output[i] = input[i] as f32;
    }
  }
}

impl<T> SampleClass for SharedClassSample<T> where T: Copy {
  fn class(&self) -> Option<u32> {
    self.label
  }
}

impl<T> SampleWeight for SharedClassSample<T> where T: Copy {
  fn weight(&self) -> Option<f32> {
    self.weight
  }

  fn mix_weight(&mut self, w: f32) {
    self.weight = Some(self.weight.map_or(w, |w0| w0 * w));
  }
}

#[derive(Clone)]
pub struct SharedClassSample2d<T> where T: Copy {
  pub input:    Array3d<T, SharedSlice<T>>,
  pub layout:   (Shape, Shape, Shape),
  pub label:    Option<u32>,
  pub weight:   Option<f32>,
}

impl SampleExtractInput<u8> for SharedClassSample2d<u8> {
  fn extract_input(&self, output: &mut [u8]) {
    let input: &[u8] = self.input.as_slice();
    output.copy_from_slice(input);
  }
}

impl SampleExtractInput<f32> for SharedClassSample2d<u8> {
  fn extract_input(&self, output: &mut [f32]) {
    let input: &[u8] = self.input.as_slice();
    for i in 0 .. input.len() {
      output[i] = input[i] as f32;
    }
  }
}

impl<T> SampleClass for SharedClassSample2d<T> where T: Copy {
  fn class(&self) -> Option<u32> {
    self.label
  }
}

impl<T> SampleWeight for SharedClassSample2d<T> where T: Copy {
  fn weight(&self) -> Option<f32> {
    self.weight
  }

  fn mix_weight(&mut self, w: f32) {
    self.weight = Some(self.weight.map_or(w, |w0| w0 * w));
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
