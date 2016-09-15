use super::{OpCapability};
use common::{CommonOperatorOutput};
use data::{ClassSample2d};

use float::ord::{F32InfNan};
use iter_utils::{argmax}; //, KahanSum};
use operator::{Operator, InternalOperator, OpPhase};

use std::iter::{Sum};

#[derive(Clone, Copy)]
pub struct ClassLossOperatorConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub num_classes:  usize,
}

pub struct SoftmaxNLLClassLossOperator {
  cfg:      ClassLossOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  max_log:  Vec<f32>,
  facts:    Vec<f32>,
  sum_fact: Vec<f32>,
  hats:     Vec<i32>,
  losses:   Vec<f32>,
  loss1:    f32,
  labels:   Vec<i32>,
  weights:  Vec<f32>,
  out:      CommonOperatorOutput<f32>,
}

impl SoftmaxNLLClassLossOperator {
  pub fn new(cfg: ClassLossOperatorConfig, cap: OpCapability, prev_op: &InternalOperator<f32, Output=CommonOperatorOutput<f32>>, prev_arm: usize) -> SoftmaxNLLClassLossOperator {
    let mut max_log = Vec::with_capacity(cfg.batch_sz);
    unsafe { max_log.set_len(cfg.batch_sz) };
    let mut facts = Vec::with_capacity(cfg.batch_sz * cfg.num_classes);
    unsafe { facts.set_len(cfg.batch_sz * cfg.num_classes) };
    let mut sum_fact = Vec::with_capacity(cfg.batch_sz);
    unsafe { sum_fact.set_len(cfg.batch_sz) };
    let mut hats = Vec::with_capacity(cfg.batch_sz);
    unsafe { hats.set_len(cfg.batch_sz) };
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    unsafe { losses.set_len(cfg.batch_sz) };
    let mut labels = Vec::with_capacity(cfg.batch_sz);
    unsafe { labels.set_len(cfg.batch_sz) };
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    unsafe { weights.set_len(cfg.batch_sz) };
    SoftmaxNLLClassLossOperator{
      cfg:      cfg,
      in_:      prev_op.output(prev_arm),
      max_log:  max_log,
      facts:    facts,
      sum_fact: sum_fact,
      hats:     hats,
      losses:   losses,
      loss1:    0.0,
      labels:   labels,
      weights:  weights,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.num_classes, cap),
    }
  }
}

impl<T> Operator<f32, ClassSample2d<T>> for SoftmaxNLLClassLossOperator where T: Copy {
  fn load_data(&mut self, samples: &[ClassSample2d<T>]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if let Some(cat) = sample.label {
        assert!(0 <= cat && cat < self.cfg.num_classes as i32);
        self.labels[idx] = cat;
      }
      //self.weights[idx] = sample.weight.unwrap_or(1.0) / self.cfg.minibatch_sz as f32;
      self.weights[idx] = sample.weight.unwrap_or(1.0);
    }
    self.out.batch_size = actual_batch_size;
  }

  fn store_loss(&mut self) -> f32 {
    self.loss1
  }
}

impl InternalOperator<f32> for SoftmaxNLLClassLossOperator {
  type Output = CommonOperatorOutput<f32>;

  fn output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn reset_loss(&mut self) {
    self.loss1 = 0.0;
  }

  fn forward(&mut self, _phase: OpPhase) {
    self.out.batch_size = self.in_.batch_size;
    for idx in 0 .. self.in_.batch_size {
      let range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(self.in_.out_buf.borrow()[range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      let max_logit = self.in_.out_buf.borrow()[idx * self.cfg.num_classes + max_logit_k];
      self.max_log[idx] = max_logit;
      self.hats[idx] = max_logit_k as i32;
      for k in 0 .. self.cfg.num_classes {
        self.facts[idx * self.cfg.num_classes + k] = (self.in_.out_buf.borrow()[idx * self.cfg.num_classes + k] - max_logit).exp();
      }
      let sum_fact: f32 = Sum::sum(self.facts[range].iter().map(|&x| x));
      for k in 0 .. self.cfg.num_classes {
        self.out.out_buf.borrow_mut()[idx * self.cfg.num_classes + k] = self.facts[idx * self.cfg.num_classes + k] / sum_fact;
      }
      self.losses[idx] = -self.weights[idx] * self.out.out_buf.borrow()[idx * self.cfg.num_classes + self.labels[idx] as usize].ln();
    }
    self.loss1 += Sum::sum(self.losses.iter().map(|&x| x));
    println!("DEBUG: softmax nll loss: loss1: {:?} loss[0]: {:?} w[0]: {:?} y[0]: {:?} p[0]: {:e}", self.loss1, self.losses[0], self.weights[0], self.labels[0], self.out.out_buf.borrow()[self.labels[0] as usize]);
  }

  fn backward(&mut self) {
    assert_eq!(self.out.batch_size, self.in_.batch_size);
    let out_buf = self.out.out_buf.borrow();
    let mut in_grad = self.in_.out_grad.as_mut().unwrap().borrow_mut();
    for idx in 0 .. self.in_.batch_size {
      for k in 0 .. self.cfg.num_classes {
        in_grad[idx * self.cfg.num_classes + k] =
            out_buf[idx * self.cfg.num_classes + k] - if k == self.labels[idx] as usize { 1.0 } else { 0.0 };
      }
    }
  }
}
