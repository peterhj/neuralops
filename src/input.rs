use common::{CommonOperatorOutput};
use data::{SharedClassSample2d};

use densearray::{ReshapeMut, ArrayIndex};
use operator::prelude::*;
use operator::data::{SampleExtractInput};

#[derive(Clone, Copy)]
pub struct SimpleInputOperatorConfig {
  pub batch_sz: usize,
  pub frame_sz: usize,
}

pub struct SimpleInputOperator {
  cfg:  SimpleInputOperatorConfig,
  out:  CommonOperatorOutput<f32>,
}

impl SimpleInputOperator {
  pub fn new(cfg: SimpleInputOperatorConfig, _cap: OpCapability) -> SimpleInputOperator {
    /*let mut in_buf = Vec::with_capacity(cfg.batch_sz * cfg.frame_sz);
    unsafe { in_buf.set_len(cfg.batch_sz * cfg.frame_sz) };*/
    SimpleInputOperator{
      cfg:  cfg,
      out:  CommonOperatorOutput::new(cfg.batch_sz, cfg.frame_sz, OpCapability::Forward),
    }
  }
}

impl Operator<f32, SharedClassSample2d<u8>> for SimpleInputOperator {
  fn load_data(&mut self, samples: &[SharedClassSample2d<u8>]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    let mut output = self.out.out_buf.borrow_mut();
    for (idx, sample) in samples.iter().enumerate() {
      // FIXME(20160920): check input shape.
      /*assert_eq!(self.cfg.frame_sz, sample.input.dim().flat_len());
      assert_eq!(sample.input.stride(), sample.input.dim().least_stride());*/
      sample.extract_input(&mut (&mut *output)[idx * self.cfg.frame_sz .. (idx+1) * self.cfg.frame_sz]);
    }
    output.reshape_mut(batch_size * self.cfg.frame_sz)
      .vector_scale(1.0 / 255.0);
    self.out.batch_size = batch_size;
  }
}

impl DiffOperator<f32> for SimpleInputOperator {
  type Output = CommonOperatorOutput<f32>;

  fn output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn forward(&mut self, _phase: OpPhase) {
    /*let frame_sz = self.cfg.frame_sz;
    let batch_sz = self.out.batch_size;
    self.out.out_buf.borrow_mut()[ .. frame_sz * batch_sz]
      .copy_from_slice(&self.in_buf[ .. frame_sz * batch_sz]);*/
  }

  fn backward(&mut self) {
  }
}
