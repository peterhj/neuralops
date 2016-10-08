use densearray::{Array3d};
use operator::data::{SampleExtractInput, SampleClass, SampleWeight};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{SharedSlice};

use rand::{Rng, thread_rng};
use std::cmp::{min};
use std::collections::{HashSet};
use std::marker::{PhantomData};

pub mod cifar;
pub mod mnist;

#[derive(Clone, Copy)]
pub enum Shape {
  Dim(usize, usize),
  Time(usize),
  Frequency(usize),
  Width(usize),
  Height(usize),
  Depth(usize),
}

#[derive(Clone)]
pub struct ClassSample2d<T> where T: Copy {
  pub input:    Array3d<T>,
  pub shape:    (Shape, Shape, Shape),
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
  pub shape:    (Shape, Shape, Shape),
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

pub struct CyclicDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  rng:      Xorshiftplus128Rng,
  inner:    Shard,
  counter:  usize,
  _marker:  PhantomData<S>,
}

impl<S, Shard> CyclicDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  pub fn new(inner: Shard) -> CyclicDataIter<S, Shard> {
    CyclicDataIter{
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

impl<S, Shard> Iterator for CyclicDataIter<S, Shard> where Shard: IndexedDataShard<S> {
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

pub struct RandomSampleDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  rng:      Xorshiftplus128Rng,
  inner:    Shard,
  _marker:  PhantomData<S>,
}

impl<S, Shard> RandomSampleDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  pub fn new(inner: Shard) -> RandomSampleDataIter<S, Shard> {
    RandomSampleDataIter{
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      inner:    inner,
      _marker:  PhantomData,
    }
  }

  pub fn len(&self) -> usize {
    self.inner.len()
  }
}

impl<S, Shard> Iterator for RandomSampleDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  type Item = S;

  fn next(&mut self) -> Option<S> {
    let idx = self.rng.gen_range(0, self.inner.len());
    let sample = self.inner.get(idx);
    Some(sample)
  }
}

pub struct SubsampleDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  batch_sz: usize,
  idxs_set: HashSet<usize>,
  rng:      Xorshiftplus128Rng,
  inner:    Shard,
  _marker:  PhantomData<S>,
}

impl<S, Shard> SubsampleDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  pub fn new(batch_sz: usize, inner: Shard) -> SubsampleDataIter<S, Shard> {
    assert!(batch_sz <= inner.len());
    SubsampleDataIter{
      batch_sz: batch_sz,
      idxs_set: HashSet::with_capacity(batch_sz),
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      inner:    inner,
      _marker:  PhantomData,
    }
  }

  pub fn len(&self) -> usize {
    self.inner.len()
  }
}

impl<S, Shard> Iterator for SubsampleDataIter<S, Shard> where Shard: IndexedDataShard<S> {
  type Item = S;

  fn next(&mut self) -> Option<S> {
    if self.idxs_set.len() >= self.batch_sz {
      self.idxs_set.clear();
    }
    loop {
      let idx = self.rng.gen_range(0, self.inner.len());
      if self.idxs_set.contains(&idx) {
        continue;
      }
      self.idxs_set.insert(idx);
      let sample = self.inner.get(idx);
      return Some(sample);
    }
  }
}

pub struct PartitionDataShard<S, Shard> where Shard: IndexedDataShard<S> {
  part_offset:  usize,
  part_len:     usize,
  inner:        Shard,
  _marker:      PhantomData<S>,
}

impl<S, Shard> PartitionDataShard<S, Shard> where Shard: IndexedDataShard<S> {
  pub fn new(part_idx: usize, num_parts: usize, inner: Shard) -> PartitionDataShard<S, Shard> {
    let inner_len = inner.len();
    let part_max_len = (inner_len + num_parts - 1) / num_parts;
    let part_offset = part_idx * part_max_len;
    let part_end = min((part_idx+1) * part_max_len, inner_len);
    let part_len = part_end - part_offset;
    PartitionDataShard{
      part_offset:  part_offset,
      part_len:     part_len,
      inner:        inner,
      _marker:      PhantomData,
    }
  }
}

impl<S, Shard> IndexedDataShard<S> for PartitionDataShard<S, Shard> where Shard: IndexedDataShard<S> {
  fn len(&self) -> usize {
    self.part_len
  }

  fn get(&mut self, idx: usize) -> S {
    assert!(idx < self.part_len);
    self.inner.get(self.part_offset + idx)
  }
}
