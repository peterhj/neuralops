use common::{CommonResources, CommonOperatorOutput};

use densearray::{ReshapeMut, ArrayIndex};
use operator::prelude::*;
use rng::{RngState};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};

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
  Crop2d{pad_w: usize, pad_h: usize, learning_only: bool},
  Flip{learning_only: bool},
  Dummy,
}

#[derive(Clone)]
pub struct VarInputOperatorConfig {
  pub batch_sz:     usize,
  pub stride:       usize,
  //pub max_stride:   usize,
  pub out_dim:      (usize, usize, usize),
  pub preprocs:     Vec<VarInputPreproc>,
}

pub struct VarInputOperator {
  cfg:  VarInputOperatorConfig,
  out:  CommonOperatorOutput<f32>,
  tmp:  Vec<f32>,
  rng:  Xorshiftplus128Rng,
  rng_state:    Vec<u64>,
}

impl VarInputOperator {
  pub fn new(cfg: VarInputOperatorConfig, _cap: OpCapability, _res: CommonResources) -> VarInputOperator {
    let mut tmp_buf = Vec::with_capacity(cfg.batch_sz * cfg.stride);
    for _ in 0 .. cfg.batch_sz * cfg.stride {
      tmp_buf.push(0.0);
    }
    let out = CommonOperatorOutput::new(cfg.batch_sz, cfg.stride, OpCapability::Forward);
    VarInputOperator{
      cfg:  cfg,
      out:  out,
      tmp:  tmp_buf,
      rng:  Xorshiftplus128Rng::new(&mut thread_rng()),
      rng_state:    vec![],
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
    for (idx, sample) in samples.iter().enumerate() {
      // FIXME(20160920): check input shape.
      /*assert_eq!(self.cfg.stride, sample.input.dim().flat_len());
      assert_eq!(sample.input.stride(), sample.input.dim().least_stride());*/
      sample.extract_input(&mut (&mut *out_buf)[idx * self.cfg.stride .. (idx+1) * self.cfg.stride]);
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
    self.rng_state.clear();
    self.rng.extract_state(&mut self.rng_state);
  }

  fn restore_rng_state(&mut self) {
    self.rng.set_state(&self.rng_state);
  }

  fn forward(&mut self, phase: OpPhase) {
    let batch_size = *self.out.batch_size.borrow();
    let mut out_buf = self.out.out_buf.borrow_mut();
    for preproc in self.cfg.preprocs.iter() {
      match preproc {
        &VarInputPreproc::Scale{scale} => {
          //let mut out_buf = self.out.out_buf.borrow_mut();
          //let mut out = &mut (&mut *out_buf)[idx * self.cfg.stride .. (idx+1) * self.cfg.stride];
          let mut out = &mut (&mut *out_buf)[ .. batch_size * self.cfg.stride];
          out.reshape_mut(batch_size * self.cfg.stride).vector_scale(scale);
        }
        &VarInputPreproc::ChannelShift{ref shift} => {
          let space_len = self.cfg.out_dim.0 * self.cfg.out_dim.1;
          for idx in 0 .. batch_size {
            for a in 0 .. self.cfg.out_dim.2 {
              let mut out = &mut (&mut *out_buf)[a * space_len + idx * self.cfg.stride .. a * space_len + (idx+1) * self.cfg.stride];
              out.reshape_mut(space_len).vector_add_scalar(-shift[a]);
            }
          }
        }
        &VarInputPreproc::Crop2d{pad_w, pad_h, learning_only} => {
          if !learning_only || phase == OpPhase::Learning {
            for idx in 0 .. batch_size {
            }
          }
          unimplemented!();
        }
        &VarInputPreproc::Flip{learning_only} => {
          if !learning_only || phase == OpPhase::Learning {
            for idx in 0 .. batch_size {
            }
          }
          unimplemented!();
        }
        _ => unimplemented!(),
      }
    }
    *self.out.out_loss.borrow_mut() = 0.0;
  }

  fn backward(&mut self) {
  }
}
