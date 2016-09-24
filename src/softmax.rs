use common::{CommonOperatorOutput};

use float::ord::{F32InfNan};
use iter_utils::{argmax}; //, KahanSum};
use operator::prelude::*;
//use operator::data::{SampleClass, SampleWeight};
use rng::xorshift::{Xorshiftplus128Rng};

use std::iter::{Sum};

#[derive(Clone, Copy)]
pub struct SoftmaxOperatorConfig {
  pub batch_sz: usize,
  pub dim:      usize,
}

pub struct SoftmaxOperator {
  cfg:      SoftmaxOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  max_log:  Vec<f32>,
  facts:    Vec<f32>,
  sum_fact: Vec<f32>,
  out:      CommonOperatorOutput<f32>,
}

impl SoftmaxOperator {
  pub fn new(cfg: SoftmaxOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize) -> SoftmaxOperator {
    let mut max_log = Vec::with_capacity(cfg.batch_sz);
    unsafe { max_log.set_len(cfg.batch_sz) };
    let mut facts = Vec::with_capacity(cfg.batch_sz * cfg.dim);
    unsafe { facts.set_len(cfg.batch_sz * cfg.dim) };
    let mut sum_fact = Vec::with_capacity(cfg.batch_sz);
    unsafe { sum_fact.set_len(cfg.batch_sz) };
    SoftmaxOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      max_log:  max_log,
      facts:    facts,
      sum_fact: sum_fact,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.dim, cap),
    }
  }
}

impl DiffOperator<f32> for SoftmaxOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn forward(&mut self, _phase: OpPhase) {
    assert!(self.in_.batch_size <= self.cfg.batch_sz);
    self.out.batch_size = self.in_.batch_size;
    //println!("DEBUG: softmax loss: batch size: {}", self.in_.batch_size);
    for idx in 0 .. self.in_.batch_size {
      let range = idx * self.cfg.dim .. (idx+1) * self.cfg.dim;
      let max_logit_k = argmax(self.in_.out_buf.borrow()[range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      let max_logit = self.in_.out_buf.borrow()[idx * self.cfg.dim + max_logit_k];
      self.max_log[idx] = max_logit;
      for k in 0 .. self.cfg.dim {
        self.facts[idx * self.cfg.dim + k] = (self.in_.out_buf.borrow()[idx * self.cfg.dim + k] - max_logit).exp();
      }
      let sum_fact: f32 = Sum::sum(self.facts[range].iter().map(|&x| x));
      for k in 0 .. self.cfg.dim {
        self.out.out_buf.borrow_mut()[idx * self.cfg.dim + k] = self.facts[idx * self.cfg.dim + k] / sum_fact;
      }
    }
  }

  fn backward(&mut self) {
    assert_eq!(self.out.batch_size, self.in_.batch_size);
    let out_buf = self.out.out_buf.borrow();
    if let Some(ref mut in_grad) = self.in_.out_grad.as_mut() {
      let mut in_grad = in_grad.borrow_mut();
      for idx in 0 .. self.in_.batch_size {
        for k in 0 .. self.cfg.dim {
          // FIXME(20160923)
          unimplemented!();
          //in_grad[idx * self.cfg.dim + k] = 0.0;
        }
      }
    }
  }
}
