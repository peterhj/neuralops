use densearray::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{SharedSlice};

use byteorder::{ReadBytesExt, LittleEndian};
use typemap::{TypeMap, Key};

use rand::{Rng, thread_rng};
use std::cmp::{min};
use std::collections::{HashSet};
use std::io::{Read, Cursor};
use std::marker::{PhantomData, Reflect};

pub mod cifar;
pub mod mnist;
pub mod varraydb;

pub fn partition_range(upper_bound: usize, parts: usize) -> Vec<(usize, usize)> {
  let mut ranges = Vec::with_capacity(parts);
  let mut offset = 0;
  for p in 0 .. parts {
    let rem_parts = parts - p;
    let span = (upper_bound - offset + rem_parts - 1) / rem_parts;
    ranges.push((offset, offset + span));
    offset += span;
  }
  assert_eq!(offset, upper_bound);
  ranges
}

pub struct SampleSharedSliceDataKey<T> where T: 'static + Copy + Reflect {
  _marker:  PhantomData<T>,
}

impl<T> Key for SampleSharedSliceDataKey<T> where T: 'static + Copy + Reflect {
  type Value = SharedSlice<T>;
}

pub trait ExtractInput<U> {
  fn extract_input(&self, output: &mut U) -> Result<(), ()>;
}

pub struct SampleExtractInputKey<U> where U: 'static + Reflect {
  _marker:  PhantomData<U>,
}

impl<U> Key for SampleExtractInputKey<U> where U: 'static + Reflect {
  type Value = Box<ExtractInput<U>>;
}

pub struct SampleClassLabelKey {}

impl Key for SampleClassLabelKey {
  type Value = u32;
}

pub struct SampleRegressTargetKey {}

impl Key for SampleRegressTargetKey {
  type Value = f32;
}

pub struct SampleWeightKey {}

impl Key for SampleWeightKey {
  type Value = f32;
}

pub struct SampleItem {
  pub kvs:  TypeMap,
}

#[derive(Clone)]
pub struct OwnedSample<T> where T: Copy {
  pub input:    Vec<T>,
}

impl SampleDatum<[u8]> for OwnedSample<u8> {
  fn extract_input(&self, output: &mut [u8]) -> Result<(), ()> {
    output.copy_from_slice(&self.input);
    Ok(())
  }
}

impl SampleDatum<[f32]> for OwnedSample<u8> {
  fn extract_input(&self, output: &mut [f32]) -> Result<(), ()> {
    for (&x, y) in self.input.iter().zip(output.iter_mut()) {
      *y = x as f32;
    }
    Ok(())
  }

  fn len(&self) -> Option<usize> {
    Some(self.input.len())
  }

  fn shape(&self) -> Option<Shape> {
    None
  }
}

#[derive(Clone)]
pub struct OwnedClassSample<T> where T: Copy {
  pub input:    Vec<T>,
  pub label:    Option<u32>,
}

impl SampleDatum<[u8]> for OwnedClassSample<u8> {
  fn extract_input(&self, output: &mut [u8]) -> Result<(), ()> {
    output.copy_from_slice(&self.input);
    Ok(())
  }
}

impl SampleDatum<[f32]> for OwnedClassSample<u8> {
  fn extract_input(&self, output: &mut [f32]) -> Result<(), ()> {
    for (&x, y) in self.input.iter().zip(output.iter_mut()) {
      *y = x as f32;
    }
    Ok(())
  }

  fn len(&self) -> Option<usize> {
    Some(self.input.len())
  }

  fn shape(&self) -> Option<Shape> {
    None
  }
}

impl SampleLabel for OwnedClassSample<u8> {
  fn class(&self) -> Option<u32> {
    self.label
  }
}

#[derive(Clone)]
pub struct SharedSample<T> where T: Copy {
  pub input:    SharedSlice<T>,
}

impl SampleDatum<[u8]> for SharedSample<u8> {
  fn extract_input(&self, output: &mut [u8]) -> Result<(), ()> {
    //let input: &[u8] = self.input.as_slice();
    output.copy_from_slice(&*self.input);
    Ok(())
  }
}

impl SampleDatum<[f32]> for SharedSample<u8> {
  fn extract_input(&self, output: &mut [f32]) -> Result<(), ()> {
    //let input: &[u8] = self.input.as_slice();
    let input: &[u8] = &*self.input;
    for i in 0 .. input.len() {
      output[i] = input[i] as f32;
    }
    Ok(())
  }

  fn len(&self) -> Option<usize> {
    Some(self.input.len())
  }

  fn shape(&self) -> Option<Shape> {
    None
  }
}

#[derive(Clone)]
pub struct SharedClassSample<T> where T: Copy {
  pub input:    SharedSlice<T>,
  pub shape:    Option<Shape>,
  pub label:    Option<u32>,
  pub weight:   Option<f32>,
}

impl SampleDatum<[u8]> for SharedClassSample<u8> {
  fn extract_input(&self, output: &mut [u8]) -> Result<(), ()> {
    //let input: &[u8] = self.input.as_slice();
    output.copy_from_slice(&*self.input);
    Ok(())
  }
}

impl SampleDatum<[f32]> for SharedClassSample<u8> {
  fn extract_input(&self, output: &mut [f32]) -> Result<(), ()> {
    //let input: &[u8] = self.input.as_slice();
    let input: &[u8] = &*self.input;
    for i in 0 .. input.len() {
      output[i] = input[i] as f32;
    }
    Ok(())
  }

  fn len(&self) -> Option<usize> {
    Some(self.input.len())
  }

  fn shape(&self) -> Option<Shape> {
    self.shape.clone()
  }
}

impl SampleLabel for SharedClassSample<u8> {
  fn class(&self) -> Option<u32> {
    self.label
  }
}

impl SampleLossWeight<ClassLoss> for SharedClassSample<u8> {
  fn weight(&self) -> Option<f32> {
    self.weight
  }

  fn mix_weight(&mut self, w: f32) -> Result<(), ()> {
    self.weight = Some(self.weight.map_or(w, |w0| w0 * w));
    Ok(())
  }
}

#[derive(Clone)]
pub struct SharedClassSample2d<T> where T: Copy {
  pub input:    Array3d<T, SharedSlice<T>>,
  pub shape:    Option<Shape>,
  pub label:    Option<u32>,
  pub weight:   Option<f32>,
}

impl SampleDatum<[u8]> for SharedClassSample2d<u8> {
  fn extract_input(&self, output: &mut [u8]) -> Result<(), ()> {
    let input: &[u8] = self.input.as_slice();
    output.copy_from_slice(input);
    Ok(())
  }
}

impl SampleDatum<[f32]> for SharedClassSample2d<u8> {
  fn extract_input(&self, output: &mut [f32]) -> Result<(), ()> {
    let input: &[u8] = self.input.as_slice();
    for i in 0 .. input.len() {
      output[i] = input[i] as f32;
    }
    Ok(())
  }

  fn len(&self) -> Option<usize> {
    Some(self.input.dim().flat_len())
  }

  fn shape(&self) -> Option<Shape> {
    self.shape.clone()
  }
}

impl SampleLabel for SharedClassSample2d<u8> {
  fn class(&self) -> Option<u32> {
    self.label
  }
}

impl SampleLossWeight<ClassLoss> for SharedClassSample2d<u8> {
  fn weight(&self) -> Option<f32> {
    self.weight
  }

  fn mix_weight(&mut self, w: f32) -> Result<(), ()> {
    self.weight = Some(self.weight.map_or(w, |w0| w0 * w));
    Ok(())
  }
}

/*impl SampleExtractInput<u8> for SharedClassSample2d<u8> {
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
}*/

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

pub struct SimpleLabelCodec<S, Iter> /*where Iter: Iterator<Item=OwnedSample<u8>>*/ {
  inner:    Iter,
  _marker:  PhantomData<S>,
}

impl<S, Iter> SimpleLabelCodec<S, Iter> /*where Iter: Iterator<Item=OwnedSample<u8>>*/ {
  pub fn new(inner: Iter) -> SimpleLabelCodec<S, Iter> {
    SimpleLabelCodec{
      inner:    inner,
      _marker:  PhantomData,
    }
  }
}

impl<Iter> Iterator for SimpleLabelCodec<OwnedClassSample<u8>, Iter> where Iter: Iterator<Item=OwnedSample<u8>> {
  type Item = OwnedClassSample<u8>;

  fn next(&mut self) -> Option<OwnedClassSample<u8>> {
    let value = match self.inner.next() {
      None => return None,
      Some(x) => x,
    };
    assert!(value.input.len() >= 4);
    let label = Cursor::new(&value.input).read_u32::<LittleEndian>().unwrap();
    //unimplemented!();
    Some(OwnedClassSample{
      input:    value.input[4 ..].to_owned(),
      label:    Some(label),
    })
  }
}

pub struct EasyLabelCodec<Iter> /*where Iter: Iterator<Item=OwnedSample<u8>>*/ {
  inner:    Iter,
  //_marker:  PhantomData<S>,
}

impl<Iter> EasyLabelCodec<Iter> /*where Iter: Iterator<Item=OwnedSample<u8>>*/ {
  pub fn new(inner: Iter) -> EasyLabelCodec<Iter> {
    EasyLabelCodec{
      inner:    inner,
      //_marker:  PhantomData,
    }
  }
}

impl<Iter> Iterator for EasyLabelCodec<Iter> where Iter: Iterator<Item=SampleItem> {
  type Item = SampleItem;

  fn next(&mut self) -> Option<SampleItem> {
    let mut item = match self.inner.next() {
      None => return None,
      Some(x) => x,
    };
    let data = if let Some(data_val) = item.kvs.get::<SampleSharedSliceDataKey<u8>>() {
      data_val.clone()
    } else {
      panic!();
    };
    let len = data.len();
    assert!(len >= 4);
    let label = Cursor::new(&*data as &[u8]).read_u32::<LittleEndian>().unwrap();
    let new_data = data.slice(4, len);
    item.kvs.insert::<SampleSharedSliceDataKey<u8>>(new_data);
    item.kvs.insert::<SampleClassLabelKey>(label);
    Some(item)
  }
}
