use densearray::prelude::*;
use operator::prelude::*;
use sharedmem::{SharedMem};

use std::io::{Cursor};
use std::rc::{Rc};
use std::sync::{Arc};

pub struct DecodeArray3dData<Iter> {
  inner:    Iter,
}

impl<Iter> DecodeArray3dData<Iter> {
  pub fn new(inner: Iter) -> DecodeArray3dData<Iter> {
    DecodeArray3dData{
      inner:    inner,
    }
  }
}

impl<Iter> Iterator for DecodeArray3dData<Iter> where Iter: Iterator<Item=SampleItem> {
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

    let mut cursor = Cursor::new(&*data as &[u8]);
    let arr = Array3d::<u8>::deserialize(&mut cursor).unwrap();
    let dim = arr.dim();

    let new_mem = SharedMem::new(arr.into_storage());
    let new_buf = new_mem.as_slice();
    item.kvs.insert::<SampleSharedSliceDataKey<u8>>(new_buf.clone());
    item.kvs.insert::<SampleSharedExtractInputKey<[f32]>>(Arc::new(new_buf));
    item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(Rc::new(dim));

    Some(item)
  }
}


pub struct SharedDecodeArray3dData<Iter> {
  inner:    Iter,
}

impl<Iter> SharedDecodeArray3dData<Iter> {
  pub fn new(inner: Iter) -> SharedDecodeArray3dData<Iter> {
    SharedDecodeArray3dData{
      inner:    inner,
    }
  }
}

impl<Iter> Iterator for SharedDecodeArray3dData<Iter> where Iter: Iterator<Item=SharedSampleItem> {
  type Item = SharedSampleItem;

  fn next(&mut self) -> Option<SharedSampleItem> {
    let mut item = match self.inner.next() {
      None => return None,
      Some(x) => x,
    };
    let data = if let Some(data_val) = item.kvs.get::<SampleSharedSliceDataKey<u8>>() {
      data_val.clone()
    } else {
      panic!();
    };

    let mut cursor = Cursor::new(&*data as &[u8]);
    let arr = Array3d::<u8>::deserialize(&mut cursor).unwrap();
    let dim = arr.dim();

    let new_mem = SharedMem::new(arr.into_storage());
    let new_buf = new_mem.as_slice();
    item.kvs.insert::<SampleSharedSliceDataKey<u8>>(new_buf.clone());
    item.kvs.insert::<SampleSharedExtractInputKey<[f32]>>(Arc::new(new_buf));
    item.kvs.insert::<SharedSampleInputShapeKey<(usize, usize, usize)>>(Arc::new(dim));

    Some(item)
  }
}
