use prelude::*;
use common::{CommonOperatorOutput, CommonResources};
use kernels::image::*;
use kernels::interpolate::*;

use densearray::{ReshapeMut, ArrayIndex};
use operator::prelude::*;
use rng::{RngState};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::rc::{Rc};

#[derive(Clone)]
pub enum InputPreproc {
  ShiftScale{shift: Option<f32>, scale: Option<f32>},
  ExperimentGammaCorrect{gamma: f32},
  ExperimentPerturb2d{width: usize, height: usize, chan: usize, scale: f32},
}

#[derive(Clone)]
pub struct SimpleInputOperatorConfig {
  pub batch_sz: usize,
  pub stride:   usize,
  pub preprocs: Vec<InputPreproc>,
}

pub struct SimpleInputOperator {
  cfg:  SimpleInputOperatorConfig,
  out:  CommonOperatorOutput<f32>,
  tmp:  Vec<f32>,
  rng:  Xorshiftplus128Rng,
  //rng_state:    Vec<u8>,
}

impl SimpleInputOperator {
  pub fn new(cfg: SimpleInputOperatorConfig, _cap: OpCapability, _res: CommonResources) -> SimpleInputOperator {
    let mut tmp_buf = Vec::with_capacity(cfg.batch_sz * cfg.stride);
    for _ in 0 .. cfg.batch_sz * cfg.stride {
      tmp_buf.push(0.0);
    }
    let out = CommonOperatorOutput::new(cfg.batch_sz, cfg.stride, OpCapability::Forward);
    SimpleInputOperator{
      cfg:  cfg,
      out:  out,
      tmp:  tmp_buf,
      rng:  Xorshiftplus128Rng::new(&mut thread_rng()),
      //rng_state:    vec![],
    }
  }
}

impl<S> DiffOperatorInput<f32, S> for SimpleInputOperator where S: SampleDatum<[f32]> {
  fn as_op(&self) -> &DiffOperator<f32, Output=Self::Output, Rng=Self::Rng> {
    self
  }

  fn load_data(&mut self, samples: &[S]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    let mut out_buf = self.out.out_buf.borrow_mut();
    for (idx, sample) in samples.iter().enumerate() {
      // FIXME(20160920): check input shape.
      /*assert_eq!(self.cfg.stride, sample.input.dim().flat_len());
      assert_eq!(sample.input.stride(), sample.input.dim().least_stride());*/
      sample.extract_input(&mut (&mut *out_buf)[idx * self.cfg.stride .. (idx+1) * self.cfg.stride]);
    }
    for preproc in self.cfg.preprocs.iter() {
      match preproc {
        &InputPreproc::ShiftScale{shift, scale} => {
          //let mut out_buf = self.out.out_buf.borrow_mut();
          //let mut out = &mut (&mut *out_buf)[idx * self.cfg.stride .. (idx+1) * self.cfg.stride];
          let mut out = &mut (&mut *out_buf)[ .. batch_size * self.cfg.stride];
          if let Some(shift) = shift {
            out.reshape_mut(batch_size * self.cfg.stride).vector_add_scalar(shift);
          }
          if let Some(scale) = scale {
            out.reshape_mut(batch_size * self.cfg.stride).vector_scale(scale);
          }
        }
        &InputPreproc::ExperimentGammaCorrect{gamma} => {
          let mut out = &mut (&mut *out_buf)[ .. batch_size * self.cfg.stride];
          for i in 0 .. batch_size * self.cfg.stride {
            out[i] = out[i].powf(gamma);
          }
        }
        &InputPreproc::ExperimentPerturb2d{width, height, chan, scale} => {
          //let mut out_buf = self.out.out_buf.borrow_mut();
          for idx in 0 .. batch_size {
            let mut out = &mut (&mut *out_buf)[idx * self.cfg.stride .. (idx+1) * self.cfg.stride];
            // FIXME(20160927)
            let offset_x = self.rng.gen_range(0, width);
            let offset_y = self.rng.gen_range(0, height);
            for a in 0 .. chan {
              out[offset_x + width * (offset_y + chan * a)] *= scale;
            }
          }
        }
      }
    }
    *self.out.batch_size.borrow_mut() = batch_size;
  }
}

impl DiffOperator<f32> for SimpleInputOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn forward(&mut self, _phase: OpPhase) {
    *self.out.out_loss.borrow_mut() = 0.0;
  }

  fn backward(&mut self) {
  }
}

#[derive(Clone)]
pub enum VarInputPreproc {
  Scale{scale: f32},
  ChannelShift{shift: Vec<f32>},
  RandomResize2d{lo: usize, hi: usize, phases: Vec<OpPhase>},
  RandomCrop2d{crop_w: usize, crop_h: usize, pad_w: usize, pad_h: usize, phases: Vec<OpPhase>},
  CenterCrop2d{crop_w: usize, crop_h: usize, phases: Vec<OpPhase>},
  RandomFlipX{phases: Vec<OpPhase>},
  Dummy,
}

#[derive(Clone)]
pub struct VarInputOperatorConfig {
  pub batch_sz:     usize,
  pub max_stride:   usize,
  pub out_dim:      (usize, usize, usize),
  pub preprocs:     Vec<VarInputPreproc>,
}

pub struct VarInputOperator {
  cfg:      VarInputOperatorConfig,
  rng:      Xorshiftplus128Rng,
  r_state:  Vec<u64>,
  in_dims:  Vec<(usize, usize, usize)>,
  tmp_dims: Vec<(usize, usize, usize)>,
  tmp_buf:  Vec<f32>,
  out:      CommonOperatorOutput<f32>,
}

impl VarInputOperator {
  pub fn new(cfg: VarInputOperatorConfig, cap: OpCapability, _res: CommonResources) -> VarInputOperator {
    let batch_sz = cfg.batch_sz;
    let mut tmp_buf = Vec::with_capacity(batch_sz * cfg.max_stride);
    tmp_buf.resize(batch_sz * cfg.max_stride, 0.0);
    let out = CommonOperatorOutput::new(batch_sz, cfg.max_stride, cap);
    VarInputOperator{
      cfg:      cfg,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      r_state:  vec![],
      in_dims:  Vec::with_capacity(batch_sz),
      tmp_dims: Vec::with_capacity(batch_sz),
      tmp_buf:  tmp_buf,
      out:      out,
    }
  }
}

impl<S> DiffOperatorInput<f32, S> for VarInputOperator where S: SampleDatum<[f32]> {
  fn as_op(&self) -> &DiffOperator<f32, Output=Self::Output, Rng=Self::Rng> {
    self
  }

  fn load_data(&mut self, samples: &[S]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    *self.out.batch_size.borrow_mut() = batch_size;

    let mut out_buf = self.out.out_buf.borrow_mut();
    out_buf.reshape_mut(self.cfg.batch_sz * self.cfg.max_stride).set_constant(0.0);
    self.in_dims.clear();
    for (idx, sample) in samples.iter().enumerate() {
      // FIXME(20160920): check input shape.
      /*assert_eq!(self.cfg.max_stride, sample.input.dim().flat_len());
      assert_eq!(sample.input.stride(), sample.input.dim().least_stride());*/
      sample.extract_input(&mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]);
      if let Some(Shape3d(in_dim)) = sample.shape() {
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }
    }
  }
}

impl DiffOperator<f32> for VarInputOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn save_rng_state(&mut self) {
    self.r_state.clear();
    self.r_state.resize(self.rng.state_size(), 0);
    self.rng.extract_state(&mut self.r_state);
  }

  fn restore_rng_state(&mut self) {
    self.rng.set_state(&self.r_state);
  }

  fn forward(&mut self, phase: OpPhase) {
    let batch_size = *self.out.batch_size.borrow();
    self.tmp_dims.clear();
    for idx in 0 .. batch_size {
      self.tmp_dims.push(self.in_dims[idx]);
    }
    let mut out_buf = self.out.out_buf.borrow_mut();
    for preproc in self.cfg.preprocs.iter() {
      match preproc {
        &VarInputPreproc::Scale{scale} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let mut out = &mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
            out.reshape_mut(dim.flat_len()).vector_scale(scale);
          }
        }
        &VarInputPreproc::ChannelShift{ref shift} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let space_len = dim.0 * dim.1;
            for a in 0 .. self.cfg.out_dim.2 {
              let mut out = &mut (&mut *out_buf)[a * space_len + idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
              out.reshape_mut(space_len).vector_add_scalar(-shift[a]);
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
    //*self.out.out_loss.borrow_mut() = 0.0;
  }

  fn backward(&mut self) {
  }
}

pub struct NewVarInputOperator<S> {
  cfg:      VarInputOperatorConfig,
  node:     OperatorNode,
  //out:      CommonOperatorOutput<f32>,
  out:      CommonOutput,
  rng:      Xorshiftplus128Rng,
  r_state:  Vec<u64>,
  in_dims:  Vec<(usize, usize, usize)>,
  tmp_dims: Vec<(usize, usize, usize)>,
  tmp_buf:  Vec<f32>,
  _marker:  PhantomData<S>,
}

impl<S> NewVarInputOperator<S> /*where S: SampleDatum<[f32]>*/ {
  pub fn new(cfg: VarInputOperatorConfig, cap: OpCapability) -> Rc<RefCell<NewVarInputOperator<S>>> {
    let batch_sz = cfg.batch_sz;
    let mut tmp_buf = Vec::with_capacity(batch_sz * cfg.max_stride);
    tmp_buf.resize(batch_sz * cfg.max_stride, 0.0);
    //let out = CommonOperatorOutput::new(batch_sz, cfg.max_stride, cap);
    let out = CommonOutput::new(batch_sz, cfg.max_stride, cap);
    Rc::new(RefCell::new(NewVarInputOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      out:      out,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      r_state:  vec![],
      in_dims:  Vec::with_capacity(batch_sz),
      tmp_dims: Vec::with_capacity(batch_sz),
      tmp_buf:  tmp_buf,
      _marker:  PhantomData,
    }))
  }
}

impl<S> Operator for NewVarInputOperator<S> /*where S: SampleDatum<[f32]>*/ {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for NewVarInputOperator<S> /*where S: SampleDatum<[f32]>*/ {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for NewVarInputOperator<S> where S: SampleDatum<[f32]> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _save_rng_state(&mut self) {
    self.r_state.clear();
    self.r_state.resize(self.rng.state_size(), 0);
    self.rng.extract_state(&mut self.r_state);
  }

  fn _restore_rng_state(&mut self) {
    self.rng.set_state(&self.r_state);
  }

  fn _load_batch(&mut self, samples: &[S]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    self.out.batch_sz.set(batch_size);

    let mut out_buf = self.out.buf.borrow_mut();
    out_buf.reshape_mut(self.cfg.batch_sz * self.cfg.max_stride).set_constant(0.0);
    self.in_dims.clear();
    for (idx, sample) in samples.iter().enumerate() {
      // FIXME(20160920): check input shape.
      /*assert_eq!(self.cfg.max_stride, sample.input.dim().flat_len());
      assert_eq!(sample.input.stride(), sample.input.dim().least_stride());*/
      sample.extract_input(&mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]);
      if let Some(Shape3d(in_dim)) = sample.shape() {
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }
    }
  }

  fn _forward(&mut self, phase: OpPhase) {
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
            out.reshape_mut(dim.flat_len()).vector_scale(scale);
          }
        }
        &VarInputPreproc::ChannelShift{ref shift} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let space_len = dim.0 * dim.1;
            for a in 0 .. self.cfg.out_dim.2 {
              let mut out = &mut (&mut *out_buf)[a * space_len + idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
              out.reshape_mut(space_len).vector_add_scalar(-shift[a]);
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
  }

  fn _backward(&mut self) {
  }
}

impl NewDiffOperator<SampleItem> for NewVarInputOperator<SampleItem> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
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
      if sample.kvs.contains::<SampleInputShape3dKey>() {
        let in_dim = *sample.kvs.get::<SampleInputShape3dKey>().unwrap();
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }
      /*sample.extract_input(&mut (&mut *out_buf)[idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride]);
      if let Some(Shape3d(in_dim)) = sample.shape() {
        self.in_dims.push(in_dim);
      } else {
        panic!();
      }*/
    }
  }

  fn _forward(&mut self, phase: OpPhase) {
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
            out.reshape_mut(dim.flat_len()).vector_scale(scale);
          }
        }
        &VarInputPreproc::ChannelShift{ref shift} => {
          for idx in 0 .. batch_size {
            let dim = self.tmp_dims[idx];
            let space_len = dim.0 * dim.1;
            for a in 0 .. self.cfg.out_dim.2 {
              let mut out = &mut (&mut *out_buf)[a * space_len + idx * self.cfg.max_stride .. (idx+1) * self.cfg.max_stride];
              out.reshape_mut(space_len).vector_add_scalar(-shift[a]);
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
  }

  fn _backward(&mut self) {
  }
}
