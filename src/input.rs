use common::{CommonResources, CommonOperatorOutput};

use densearray::{ReshapeMut, ArrayIndex};
use operator::prelude::*;
use operator::data::{SampleExtractInput};
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

impl<S> DiffOperatorInput<f32, S> for SimpleInputOperator where S: SampleExtractInput<f32> {
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
