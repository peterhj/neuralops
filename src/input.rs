use super::{OpCapability};
use common::{CommonOperatorOutput};
use data::{ClassSample2d};

use densearray::{ArrayIndex};
use operator::{Operator, InternalOperator, OpPhase};
use operator::data::{ClassSample};

#[derive(Clone, Copy)]
pub struct SimpleInputOperatorConfig {
  pub batch_sz: usize,
  pub frame_sz: usize,
}

pub struct SimpleInputOperator {
  cfg:      SimpleInputOperatorConfig,
  in_buf:   Vec<f32>,
  out:      CommonOperatorOutput<f32>,
}

impl SimpleInputOperator {
  pub fn new(cfg: SimpleInputOperatorConfig, _cap: OpCapability) -> SimpleInputOperator {
    let mut in_buf = Vec::with_capacity(cfg.batch_sz * cfg.frame_sz);
    unsafe { in_buf.set_len(cfg.batch_sz * cfg.frame_sz) };
    SimpleInputOperator{
      cfg:      cfg,
      in_buf:   in_buf,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.frame_sz, OpCapability::Forward),
    }
  }
}

impl Operator<f32, ClassSample<f32>> for SimpleInputOperator {
  fn load_data(&mut self, samples: &[ClassSample<f32>]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      assert_eq!(self.cfg.frame_sz, sample.input.len());
      self.in_buf[idx * self.cfg.frame_sz .. (idx+1) * self.cfg.frame_sz].copy_from_slice(&sample.input);
    }
    self.out.batch_size = batch_size;
  }
}

impl Operator<f32, ClassSample<u8>> for SimpleInputOperator {
  fn load_data(&mut self, samples: &[ClassSample<u8>]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      assert_eq!(self.cfg.frame_sz, sample.input.len());
      for j in 0 .. self.cfg.frame_sz {
        self.in_buf[idx * self.cfg.frame_sz + j] = sample.input[j] as f32 / 255.0;
      }
    }
    self.out.batch_size = batch_size;
  }
}

impl Operator<f32, ClassSample2d<u8>> for SimpleInputOperator {
  fn load_data(&mut self, samples: &[ClassSample2d<u8>]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      //println!("DEBUG: input op: loading {}/{}", idx, batch_size);
      assert_eq!(self.cfg.frame_sz, sample.input.dim().flat_len());
      assert_eq!(sample.input.stride(), sample.input.dim().least_stride());
      let input = sample.input.as_slice();
      for i in 0 .. self.cfg.frame_sz {
        self.in_buf[idx * self.cfg.frame_sz + i] = input[i] as f32 / 255.0;
      }
    }
    self.out.batch_size = batch_size;
  }
}

impl InternalOperator<f32> for SimpleInputOperator {
  type Output = CommonOperatorOutput<f32>;

  fn output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn forward(&mut self, _phase: OpPhase) {
    self.out.out_buf.borrow_mut().copy_from_slice(&self.in_buf);
  }

  fn backward(&mut self) {
  }
}
