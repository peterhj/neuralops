use data::{IndexedDataShard, Layout, ClassSample2d};

use densearray::{ArrayIndex, Array3d};

use byteorder::{ReadBytesExt, LittleEndian, BigEndian};
use memmap::{Mmap, Protection};

use std::fs::{File};
use std::path::{PathBuf};

fn mmap_idx_file(file: &mut File) -> (usize, Option<(usize, usize, usize)>, Mmap) {
  let magic: u32 = file.read_u32::<BigEndian>().unwrap();
  let magic2 = (magic >> 8) as u8;
  let magic3 = (magic >> 0) as u8;
  assert_eq!(magic2, 0x08);
  let ndims = magic3 as usize;
  let mut dims = vec![];
  for d in 0 .. ndims {
    dims.push(file.read_u32::<BigEndian>().unwrap() as usize);
  }
  let n = dims[0] as usize;
  let mut frame_size = 1;
  for d in 1 .. ndims {
    frame_size *= dims[d] as usize;
  }
  let buf = match Mmap::open_with_offset(file, Protection::Read, (1 + ndims) * 4, frame_size * n) {
    Ok(buf) => buf,
    Err(e) => panic!("failed to mmap buffer: {:?}", e),
  };
  assert_eq!(buf.len(), n * frame_size);
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
  frames_f: File,
  frames_m: Mmap,
  labels_f: File,
  labels_m: Mmap,
}

impl MnistDataShard {
  pub fn new(data_path: PathBuf, labels_path: PathBuf) -> MnistDataShard {
    let mut frames_file = File::open(&data_path).unwrap();
    let mut labels_file = File::open(&labels_path).unwrap();
    let (f_n, frame_dim, frames_mmap) = mmap_idx_file(&mut frames_file);
    let (l_n, _, labels_mmap) = mmap_idx_file(&mut labels_file);
    assert_eq!(f_n, l_n);
    MnistDataShard{
      len:      f_n,
      frame_sz: frame_dim.unwrap().flat_len(),
      frame_d:  frame_dim.unwrap(),
      frames_f: frames_file,
      frames_m: frames_mmap,
      labels_f: labels_file,
      labels_m: labels_mmap,
    }
  }
}

impl IndexedDataShard<ClassSample2d<u8>> for MnistDataShard {
  fn len(&self) -> usize {
    self.len
  }

  fn get(&mut self, idx: usize) -> ClassSample2d<u8> {
    assert!(idx < self.len);
    let mut input_buf = Vec::with_capacity(self.frame_sz);
    input_buf.extend_from_slice(&unsafe { self.frames_m.as_slice() }[idx * self.frame_sz .. (idx+1) * self.frame_sz]);
    let label = unsafe { self.labels_m.as_slice() }[idx] as i32;
    ClassSample2d{
      input:    Array3d::from_storage(self.frame_d, input_buf),
      layout:   (Layout::Width, Layout::Height, Layout::Dim(0)),
      label:    Some(label),
      weight:   None,
    }
  }
}
