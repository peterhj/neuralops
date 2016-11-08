use densearray::prelude::*;
use operator::prelude::*;
use sharedmem::{SharedMem};

use std::io::{Cursor};
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
    item.kvs.insert::<SampleInputShape3dKey>(dim);

    Some(item)
  }
}
