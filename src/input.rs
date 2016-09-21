use common::{CommonOperatorOutput};

use densearray::{ReshapeMut, ArrayIndex};
use operator::prelude::*;
use operator::data::{SampleExtractInput};

#[derive(Clone, Copy)]
pub struct SimpleInputOperatorConfig {
  pub batch_sz: usize,
  pub stride:   usize,
}

pub struct SimpleInputOperator {
  cfg:  SimpleInputOperatorConfig,
  out:  CommonOperatorOutput<f32>,
}

impl SimpleInputOperator {
  pub fn new(cfg: SimpleInputOperatorConfig, _cap: OpCapability) -> SimpleInputOperator {
    SimpleInputOperator{
      cfg:  cfg,
      out:  CommonOperatorOutput::new(cfg.batch_sz, cfg.stride, OpCapability::Forward),
    }
  }
}

impl<S> Operator<f32, S> for SimpleInputOperator where S: SampleExtractInput<f32> {
  fn load_data(&mut self, samples: &[S]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.cfg.batch_sz);
    let mut output = self.out.out_buf.borrow_mut();
    for (idx, sample) in samples.iter().enumerate() {
      // FIXME(20160920): check input shape.
      /*assert_eq!(self.cfg.stride, sample.input.dim().flat_len());
      assert_eq!(sample.input.stride(), sample.input.dim().least_stride());*/
      sample.extract_input(&mut (&mut *output)[idx * self.cfg.stride .. (idx+1) * self.cfg.stride]);
    }
    output.reshape_mut(batch_size * self.cfg.stride)
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
  }

  fn backward(&mut self) {
  }
}
