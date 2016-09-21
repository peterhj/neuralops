use data::{IndexedDataShard, Shape, SharedClassSample2d};

use densearray::{ArrayIndex, Array3d};
use sharedmem::{MemoryMap, SharedMem};

use byteorder::{ReadBytesExt, BigEndian};

use std::fs::{File};
use std::path::{PathBuf};

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

impl IndexedDataShard<SharedClassSample2d<u8>> for CifarDataShard {
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
      shape:    (Shape::Width(32), Shape::Height(32), Shape::Dim(0, 3)),
      label:    Some(label),
      weight:   None,
    }
  }
}
