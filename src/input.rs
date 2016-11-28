use prelude::*;
use kernels::ffi::*;

use densearray::{ReshapeMut, ArrayIndex};
use operator::prelude::*;
use rng::{RngState};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{RwSlice};

use rand::{Rng, thread_rng};
use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::rc::{Rc};

#[derive(Clone)]
pub enum VarInputPreproc {
  Scale{scale: f32},
  DivScale{scale: f32},
  ChannelShift{shift: Vec<f32>},
  RandomResize2d{lo: usize, hi: usize, phases: Vec<OpPhase>},
  RandomCrop2d{crop_w: usize, crop_h: usize, pad_w: usize, pad_h: usize, phases: Vec<OpPhase>},
  OffsetCrop2d{crop_w: usize, crop_h: usize, offset_x: isize, offset_y: isize, phases: Vec<OpPhase>},
  CenterCrop2d{crop_w: usize, crop_h: usize, phases: Vec<OpPhase>},
  RandomFlipX{phases: Vec<OpPhase>},
  Dummy,
}

#[derive(Clone)]
pub struct VarInputOperatorConfig {
  pub batch_sz:     usize,
  pub max_stride:   usize,
  pub out_dim:      (usize, usize, usize),
  pub in_dtype:     Dtype,
  pub preprocs:     Vec<VarInputPreproc>,
}

pub struct NewVarInputOperator<S, IoBuf: ?Sized> {
  cfg:      VarInputOperatorConfig,
  node:     OperatorNode,
  out:      CommonOutput,
  rng:      Xorshiftplus128Rng,
  r_state:  Vec<u64>,
  in_dims:  Vec<(usize, usize, usize)>,
  tmp_dims: Vec<(usize, usize, usize)>,
  tmp_buf:  Vec<f32>,
  watch:    Stopwatch,
  _marker:  PhantomData<fn (S, IoBuf)>,
}

impl<S, IoBuf: ?Sized> NewVarInputOperator<S, IoBuf> {
  pub fn new(cfg: VarInputOperatorConfig, cap: OpCapability) -> Rc<RefCell<NewVarInputOperator<S, IoBuf>>> {
    let batch_sz = cfg.batch_sz;
    let mut tmp_buf = Vec::with_capacity(batch_sz * cfg.max_stride);
    tmp_buf.resize(batch_sz * cfg.max_stride, 0.0);
    let out = CommonOutput::new(batch_sz, cfg.max_stride, OpCapability::Forward);
    Rc::new(RefCell::new(NewVarInputOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      out:      out,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      r_state:  vec![],
      in_dims:  Vec::with_capacity(batch_sz),
      tmp_dims: Vec::with_capacity(batch_sz),
      tmp_buf:  tmp_buf,
      watch:    Stopwatch::new(),
      _marker:  PhantomData,
    }))
  }

  /*pub fn _get_output(&self) -> RwSlice<f32> {
    self.out.buf.as_slice()
  }*/
}

impl<S, IoBuf: ?Sized> Operator for NewVarInputOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for NewVarInputOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }

  /*fn diff_op(&mut self) -> &mut DiffOperator<SampleItem, IoBuf=[f32]> {
    self
  }*/
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for NewVarInputOperator<S, IoBuf> {
}

impl<IoBuf: ?Sized> DiffOperator<SampleItem, IoBuf> for NewVarInputOperator<SampleItem, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<SampleItem, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<SampleItem, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.node.pop(epoch);
  }

  fn _save_rng_state(&mut self) {
    self.r_state.clear();
    self.r_state.resize(self.rng.state_size(), 0);
    self.rng.extract_state(&mut self.r_state);
  }

  fn _restore_rng_state(&mut self) {
    self.rng.set_state(&self.r_state);
  }

  fn _load_batch(&mut self, samples: &[SampleItem]) {
    self.watch.lap();

    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    self.out.batch_sz.set(batch_size);

    let mut out_buf = self.out.buf.borrow_mut();
    out_buf.reshape_mut(self.cfg.batch_sz * self.cfg.max_stride).set_constant(0.0);
    self.in_dims.clear();
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleSharedExtractInputKey<[f32]>>() {
        let data = sample.kvs.get::<SampleSharedExtractInputKey<[f32]>>().unwrap();
        data.extract_input(&mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]).unwrap();
      } else if sample.kvs.contains::<SampleExtractInputKey<[f32]>>() {
        let data = sample.kvs.get::<SampleExtractInputKey<[f32]>>().unwrap();
        data.extract_input(&mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]).unwrap();
      } else {
        panic!();
      }
      if sample.kvs.contains::<SampleInputShapeKey<(usize, usize, usize)>>() {
        let data = sample.kvs.get::<SampleInputShapeKey<(usize, usize, usize)>>().unwrap();
        let in_dim = data.input_shape().unwrap();
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }
      /*if sample.kvs.contains::<SampleInputShape3dKey>() {
        let in_dim = *sample.kvs.get::<SampleInputShape3dKey>().unwrap();
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }*/
    }

    self.watch.lap();
    println!("DEBUG: varinput: load batch: {:.6}", self.watch.elapsed());
  }

  fn _forward(&mut self, phase: OpPhase) {
    self.watch.lap();

    let batch_size = self.out.batch_sz.get();
    self.tmp_dims.clear();
    for idx in 0 .. batch_size {
      self.tmp_dims.push(self.in_dims[idx]);
    }
    let mut out_buf = self.out.buf.borrow_mut();
    for preproc in self.cfg.preprocs.iter() {
      match preproc {
        &VarInputPreproc::Scale{scale} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let mut out = &mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
            out.reshape_mut(dim.flat_len()).scale(scale);
          }
        }
        &VarInputPreproc::DivScale{scale} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let mut out = &mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
            out.reshape_mut(dim.flat_len()).div_scalar(scale);
          }
        }
        &VarInputPreproc::ChannelShift{ref shift} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let space_len = dim.0 * dim.1;
            for a in 0 .. self.cfg.out_dim.2 {
              let mut out = &mut (&mut *out_buf)[a * space_len + idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
              out.reshape_mut(space_len).add_scalar(-shift[a]);
            }
          }
        }
        &VarInputPreproc::RandomResize2d{lo, hi, ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              let resized_out_d = self.rng.gen_range(lo, hi+1);
              let (out_w, out_h) = if in_dim.0 >= in_dim.1 {
                let sy = resized_out_d as f64 / in_dim.1 as f64;
                ((sy * in_dim.0 as f64).round() as usize, resized_out_d)
              } else {
                let sx = resized_out_d as f64 / in_dim.0 as f64;
                (resized_out_d, (sx * in_dim.1 as f64).round() as usize)
              };
              let out_dim = (out_w, out_h, in_dim.2);
              let out_len = out_dim.flat_len();
              {
                let out = &(&*out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
                let mut tmp = &mut self.tmp_buf[idx * out_len .. (idx+1) * out_len];
                unsafe { neuralops_interpolate2d_catmullrom(
                    in_dim.0, in_dim.1, in_dim.2,
                    out_dim.0, out_dim.1,
                    out.as_ptr(),
                    tmp.as_mut_ptr(),
                ) };
              }
              let tmp = &self.tmp_buf[idx * out_len .. (idx+1) * out_len];
              let mut out = &mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
              out[ .. out_len].copy_from_slice(&tmp);
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::RandomCrop2d{crop_w, crop_h, pad_w, pad_h, ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              assert!(crop_w <= in_dim.0 + 2 * pad_w);
              assert!(crop_h <= in_dim.1 + 2 * pad_h);
              let out_dim = (crop_w, crop_h, in_dim.2);
              let out_len = out_dim.flat_len();
              let offset_x = self.rng.gen_range(0, in_dim.0 + 2 * pad_w - crop_w + 1) as isize - pad_w as isize;
              let offset_y = self.rng.gen_range(0, in_dim.1 + 2 * pad_h - crop_h + 1) as isize - pad_h as isize;
              {
                let out = &(&*out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
                let mut tmp = &mut self.tmp_buf[idx * out_len .. (idx+1) * out_len];
                unsafe { neuralops_image_crop(
                    in_dim.0, in_dim.1, in_dim.2,
                    out_dim.0, out_dim.1,
                    offset_x, offset_y,
                    out.as_ptr(),
                    tmp.as_mut_ptr(),
                ) };
              }
              let tmp = &self.tmp_buf[idx * out_len .. (idx+1) * out_len];
              let mut out = &mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
              out[ .. out_len].copy_from_slice(&tmp);
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::OffsetCrop2d{crop_w, crop_h, offset_x, offset_y, ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              let out_dim = (crop_w, crop_h, in_dim.2);
              let out_len = out_dim.flat_len();
              {
                let out = &(&*out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
                let mut tmp = &mut self.tmp_buf[idx * out_len .. (idx+1) * out_len];
                unsafe { neuralops_image_crop(
                    in_dim.0, in_dim.1, in_dim.2,
                    out_dim.0, out_dim.1,
                    offset_x, offset_y,
                    out.as_ptr(),
                    tmp.as_mut_ptr(),
                ) };
              }
              let tmp = &self.tmp_buf[idx * out_len .. (idx+1) * out_len];
              let mut out = &mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
              out[ .. out_len].copy_from_slice(&tmp);
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::CenterCrop2d{crop_w, crop_h, ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let in_dim = self.tmp_dims[idx];
              assert!(crop_w <= in_dim.0);
              assert!(crop_h <= in_dim.1);
              let out_dim = (crop_w, crop_h, in_dim.2);
              let out_len = out_dim.flat_len();
              let offset_x = ((in_dim.0 - crop_w) / 2) as isize;
              let offset_y = ((in_dim.1 - crop_h) / 2) as isize;
              {
                let out = &(&*out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
                let mut tmp = &mut self.tmp_buf[idx * out_len .. (idx+1) * out_len];
                unsafe { neuralops_image_crop(
                    in_dim.0, in_dim.1, in_dim.2,
                    out_dim.0, out_dim.1,
                    offset_x, offset_y,
                    out.as_ptr(),
                    tmp.as_mut_ptr(),
                ) };
              }
              let tmp = &self.tmp_buf[idx * out_len .. (idx+1) * out_len];
              let mut out = &mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
              out[ .. out_len].copy_from_slice(&tmp);
              self.tmp_dims[idx] = out_dim;
            }
          }
        }
        &VarInputPreproc::RandomFlipX{ref phases} => {
          if phases.contains(&phase) {
            for idx in 0 .. batch_size {
              let out_dim = self.tmp_dims[idx];
              let out_len = out_dim.flat_len();
              let bernoulli = self.rng.gen_range(0, 2);
              match bernoulli {
                0 => {}
                1 => {
                  {
                    let out = &(&*out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
                    let mut tmp = &mut self.tmp_buf[idx * out_len .. (idx+1) * out_len];
                    unsafe { neuralops_image_flip(
                        out_dim.0, out_dim.1, out_dim.2,
                        out.as_ptr(),
                        tmp.as_mut_ptr(),
                    ) };
                  }
                  let tmp = &self.tmp_buf[idx * out_len .. (idx+1) * out_len];
                  let mut out = &mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
                  out[ .. out_len].copy_from_slice(&tmp);
                }
                _ => unreachable!(),
              }
            }
          }
        }
        _ => unimplemented!(),
      }
    }
    let out_len = self.cfg.out_dim.flat_len();
    for idx in 0 .. batch_size {
      assert_eq!(self.cfg.out_dim, self.tmp_dims[idx]);
      let out = &(&*out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
      let mut tmp = &mut self.tmp_buf[idx * out_len .. (idx+1) * out_len];
      tmp.copy_from_slice(&out[ .. out_len]);
    }
    out_buf[ .. batch_size * out_len].copy_from_slice(&self.tmp_buf[ .. batch_size * out_len]);
    //println!("DEBUG: varinput: output: {:?}", &out_buf[ .. out_len]);

    self.watch.lap();
    println!("DEBUG: varinput: fwd: {:.6}", self.watch.elapsed());
  }

  fn _backward(&mut self) {
  }
}
