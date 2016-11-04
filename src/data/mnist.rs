use prelude::*;
use data::{IndexedDataShard, SharedClassSample2d};

use densearray::{ArrayIndex, Array3d};
use operator::prelude::*;
use sharedmem::{MemoryMap, SharedMem};

use byteorder::{ReadBytesExt, BigEndian};

use std::fs::{File};
use std::path::{PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

fn mmap_idx_file(mut file: File) -> (usize, Option<(usize, usize, usize)>, MemoryMap<u8>) {
  let magic: u32 = file.read_u32::<BigEndian>().unwrap();
  let magic2 = (magic >> 8) as u8;
  let magic3 = (magic >> 0) as u8;
  assert_eq!(magic2, 0x08);
  let ndims = magic3 as usize;
  let mut dims = vec![];
  for _ in 0 .. ndims {
    dims.push(file.read_u32::<BigEndian>().unwrap() as usize);
  }
  let n = dims[0] as usize;
  let mut frame_size = 1;
  for d in 1 .. ndims {
    frame_size *= dims[d] as usize;
  }
  /*let buf = match Mmap::open_with_offset(file, Protection::Read, (1 + ndims) * 4, frame_size * n) {
    Ok(buf) => buf,
    Err(e) => panic!("failed to mmap buffer: {:?}", e),
  };
  assert_eq!(buf.len(), n * frame_size);*/
  let buf = match MemoryMap::open_with_offset(file, (1 + ndims) * 4, frame_size * n) {
    Ok(buf) => buf,
    Err(e) => panic!("failed to mmap buffer: {:?}", e),
  };
  if ndims == 3 {
    (n, Some((dims[2], dims[1], 1)), buf)
  } else if ndims == 1 {
    (n, None, buf)
  } else {
    unimplemented!();
  }
}

pub struct MnistDataShard {
  len:      usize,
  frame_sz: usize,
  frame_d:  (usize, usize, usize),
  frames_m: SharedMem<u8>,
  labels_m: SharedMem<u8>,
}

impl MnistDataShard {
  pub fn new(data_path: PathBuf, labels_path: PathBuf) -> MnistDataShard {
    let mut frames_file = File::open(&data_path).unwrap();
    let mut labels_file = File::open(&labels_path).unwrap();
    let (f_n, frame_dim, frames_mmap) = mmap_idx_file(frames_file);
    let (l_n, _, labels_mmap) = mmap_idx_file(labels_file);
    assert_eq!(f_n, l_n);
    MnistDataShard{
      len:      f_n,
      frame_sz: frame_dim.unwrap().flat_len(),
      frame_d:  frame_dim.unwrap(),
      frames_m: SharedMem::new(frames_mmap),
      labels_m: SharedMem::new(labels_mmap),
    }
  }
}

/*impl IndexedDataShard<SharedClassSample2d<u8>> for MnistDataShard {
  fn len(&self) -> usize {
    self.len
  }

  fn get(&mut self, idx: usize) -> SharedClassSample2d<u8> {
    assert_eq!(784, self.frame_sz);
    assert_eq!((28, 28, 1), self.frame_d);
    assert!(idx < self.len);
    /*let mut input_buf = Vec::with_capacity(self.frame_sz);
    input_buf.extend_from_slice(&self.frames_m[idx * self.frame_sz .. (idx+1) * self.frame_sz]);
    assert_eq!(self.frame_sz, input_buf.len());*/
    let input_buf = self.frames_m.slice(idx * self.frame_sz, (idx+1) * self.frame_sz);
    let label = self.labels_m.as_slice()[idx] as u32;
    SharedClassSample2d{
      input:    Array3d::from_storage(self.frame_d, input_buf),
      //shape:    (Shape::Width(28), Shape::Height(28), Shape::Dim(0, 1)),
      shape:    Some(Shape3d((28, 28, 1))),
      label:    Some(label),
      weight:   None,
    }
  }
}*/

impl IndexedDataShard<SampleItem> for MnistDataShard {
  fn len(&self) -> usize {
    self.len
  }

  fn get(&mut self, idx: usize) -> SampleItem {
    assert_eq!(784, self.frame_sz);
    assert_eq!((28, 28, 1), self.frame_d);
    assert!(idx < self.len);
    let input_buf = self.frames_m.slice(idx * self.frame_sz, (idx+1) * self.frame_sz);
    let label = self.labels_m.as_slice()[idx] as u32;
    let mut item = SampleItem::new();
    item.kvs.insert::<SampleSharedExtractInputKey<[f32]>>(Arc::new(input_buf));
    item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(Rc::new(self.frame_d));
    item.kvs.insert::<SampleClassLabelKey>(label);
    item
  }
}
