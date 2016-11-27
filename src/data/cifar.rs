use data::{IndexedDataShard};

use operator::prelude::*;
use sharedmem::{MemoryMap, SharedMem};

use std::fs::{File};
use std::path::{PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

#[derive(Clone, Copy, Debug)]
pub enum CifarFlavor {
  Cifar10,
  Cifar100,
}

pub struct CifarDataShard {
  flavor:   CifarFlavor,
  len:      usize,
  frame_sz: usize,
  input_d:  (usize, usize, usize),
  data_m:   SharedMem<u8>,
}

impl CifarDataShard {
  pub fn new(flavor: CifarFlavor, data_path: PathBuf) -> CifarDataShard {
    let data_file = File::open(&data_path).unwrap();
    let file_meta = data_file.metadata().unwrap();
    let file_sz = file_meta.len() as usize;
    let frame_sz = match flavor {
      CifarFlavor::Cifar10  => 3073,
      CifarFlavor::Cifar100 => 3074,
    };
    assert_eq!(0, file_sz % frame_sz);
    let n = file_sz / frame_sz;
    let map = match MemoryMap::open_with_offset(data_file, 0, frame_sz * n) {
      Ok(map) => map,
      Err(e) => panic!("failed to mmap cifar batch file: {:?}", e),
    };
    CifarDataShard{
      flavor:   flavor,
      len:      n,
      frame_sz: frame_sz,
      input_d:  (32, 32, 3),
      data_m:   SharedMem::new(map),
    }
  }
}

/*impl IndexedDataShard<SharedClassSample2d<u8>> for CifarDataShard {
  fn len(&self) -> usize {
    self.len
  }

  fn get(&mut self, idx: usize) -> SharedClassSample2d<u8> {
    assert!(idx < self.len);
    let (input_buf, label) = match self.flavor {
      CifarFlavor::Cifar10 => {
        ( self.data_m.slice(idx * self.frame_sz + 1, (idx+1) * self.frame_sz),
          self.data_m.as_slice()[idx * self.frame_sz] as u32)
      }
      CifarFlavor::Cifar100 => {
        ( self.data_m.slice(idx * self.frame_sz + 2, (idx+1) * self.frame_sz),
          self.data_m.as_slice()[idx * self.frame_sz + 1] as u32)
      }
    };
    SharedClassSample2d{
      input:    Array3d::from_storage(self.input_d, input_buf),
      shape:    Some(Shape3d(self.input_d)),
      label:    Some(label),
      weight:   None,
    }
  }
}*/

impl IndexedDataShard<SampleItem> for CifarDataShard {
  fn len(&self) -> usize {
    self.len
  }

  fn get(&mut self, idx: usize) -> SampleItem {
    assert!(idx < self.len);
    let (input_buf, label) = match self.flavor {
      CifarFlavor::Cifar10 => {
        ( self.data_m.slice(idx * self.frame_sz + 1, (idx+1) * self.frame_sz),
          self.data_m.as_slice()[idx * self.frame_sz] as u32)
      }
      CifarFlavor::Cifar100 => {
        ( self.data_m.slice(idx * self.frame_sz + 2, (idx+1) * self.frame_sz),
          self.data_m.as_slice()[idx * self.frame_sz + 1] as u32)
      }
    };
    let mut item = SampleItem::new();
    item.kvs.insert::<SampleSharedExtractInputKey<[u8]>>(Arc::new(input_buf.clone()));
    item.kvs.insert::<SampleSharedExtractInputKey<[f32]>>(Arc::new(input_buf));
    item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(Rc::new((32, 32, 3)));
    item.kvs.insert::<SampleClassLabelKey>(label);
    item
  }
}
