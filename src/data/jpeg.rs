use operator::prelude::*;
use sharedmem::{SharedMem};
use stb_image::image::{Image, LoadResult, load_from_memory};
use turbojpeg::{TurbojpegDecoder};

use std::rc::{Rc};
use std::sync::{Arc};

pub struct JpegDecoder {
  turbo:    TurbojpegDecoder,
}

impl JpegDecoder {
  pub fn new() -> JpegDecoder {
    JpegDecoder{
      turbo:    TurbojpegDecoder::create().unwrap(),
    }
  }

  pub fn _decode(&mut self, data: &[u8]) -> (Vec<u8>, usize, usize) {
    let (pixels, width, height) = match self.turbo.decode_rgb8(data) {
      Ok((head, pixels)) => {
        //println!("DEBUG: JpegDecoderData: decoded jpeg");
        (pixels, head.width, head.height)
      }
      Err(_) => {
        match load_from_memory(data) {
          LoadResult::ImageU8(mut im) => {
            if im.depth != 3 && im.depth != 1 {
              panic!("jpeg codec: unsupported depth: {}", im.depth);
            }
            assert_eq!(im.depth * im.width * im.height, im.data.len());

            if im.depth == 1 {
              let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
              assert_eq!(im.width * im.height, im.data.len());
              for i in 0 .. im.data.len() {
                rgb_data.push(im.data[i]);
                rgb_data.push(im.data[i]);
                rgb_data.push(im.data[i]);
              }
              assert_eq!(3 * im.width * im.height, rgb_data.len());
              im = Image::new(im.width, im.height, 3, rgb_data);
            }
            assert_eq!(3, im.depth);

            (im.data, im.width, im.height)
          }
          LoadResult::Error(_) |
          LoadResult::ImageF32(_) => {
            panic!("jpeg codec: backup stb_image decoder failed");
          }
        }
      }
    };

    // FIXME(20161018): transposing should be optional!
    let mut transp_pixels = Vec::with_capacity(pixels.len());
    for c in 0 .. 3 {
      for y in 0 .. height {
        for x in 0 .. width {
          transp_pixels.push(pixels[c + x * 3 + y * 3 * width]);
        }
      }
    }
    assert_eq!(pixels.len(), transp_pixels.len());

    (transp_pixels, width, height)
  }
}

pub struct DecodeJpegData<Iter> {
  //turbo:    TurbojpegDecoder,
  decoder:  JpegDecoder,
  inner:    Iter,
}

impl<Iter> DecodeJpegData<Iter> {
  pub fn new(inner: Iter) -> DecodeJpegData<Iter> {
    DecodeJpegData{
      //turbo:    TurbojpegDecoder::create().unwrap(),
      decoder:  JpegDecoder::new(),
      inner:    inner,
    }
  }
}

impl<Iter> Iterator for DecodeJpegData<Iter> where Iter: Iterator<Item=SampleItem> {
  type Item = SampleItem;

  fn next(&mut self) -> Option<SampleItem> {
    //println!("DEBUG: DecodeJpegData: received item");
    loop {
      let mut item = match self.inner.next() {
        None => return None,
        Some(x) => x,
      };
      let data = if let Some(data_val) = item.kvs.get::<SampleSharedSliceDataKey<u8>>() {
        data_val.clone()
      } else {
        panic!();
      };

      let (transp_pixels, width, height) = self.decoder._decode(&*data);

      let new_mem = SharedMem::new(transp_pixels);
      let new_buf = new_mem.as_slice();
      item.kvs.insert::<SampleSharedSliceDataKey<u8>>(new_buf.clone());
      item.kvs.insert::<SampleSharedExtractInputKey<[u8]>>(Arc::new(new_buf.clone()));
      item.kvs.insert::<SampleSharedExtractInputKey<[f32]>>(Arc::new(new_buf));
      let dim: (usize, usize, usize) = (width, height, 3);
      item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(Rc::new(dim));
      //item.kvs.insert::<SampleInputShapeKey<(usize, usize, usize)>>(Arc::new(dim));

      return Some(item);
    }
  }
}

pub struct SharedDecodeJpegData<Iter> {
  //turbo:    TurbojpegDecoder,
  decoder:  JpegDecoder,
  inner:    Iter,
}

impl<Iter> SharedDecodeJpegData<Iter> {
  pub fn new(inner: Iter) -> SharedDecodeJpegData<Iter> {
    SharedDecodeJpegData{
      //turbo:    TurbojpegDecoder::create().unwrap(),
      decoder:  JpegDecoder::new(),
      inner:    inner,
    }
  }
}

impl<Iter> Iterator for SharedDecodeJpegData<Iter> where Iter: Iterator<Item=SharedSampleItem> {
  type Item = SharedSampleItem;

  fn next(&mut self) -> Option<SharedSampleItem> {
    //println!("DEBUG: SharedDecodeJpegData: received item");
    loop {
      let mut item = match self.inner.next() {
        None => return None,
        Some(x) => x,
      };
      let data = if let Some(data_val) = item.kvs.get::<SampleSharedSliceDataKey<u8>>() {
        data_val.clone()
      } else {
        panic!();
      };

      let (transp_pixels, width, height) = self.decoder._decode(&*data);

      let new_mem = SharedMem::new(transp_pixels);
      let new_buf = new_mem.as_slice();
      item.kvs.insert::<SampleSharedSliceDataKey<u8>>(new_buf.clone());
      item.kvs.insert::<SampleSharedExtractInputKey<[u8]>>(Arc::new(new_buf.clone()));
      item.kvs.insert::<SampleSharedExtractInputKey<[f32]>>(Arc::new(new_buf));
      let dim: (usize, usize, usize) = (width, height, 3);
      item.kvs.insert::<SharedSampleInputShapeKey<(usize, usize, usize)>>(Arc::new(dim));

      return Some(item);
    }
  }
}
