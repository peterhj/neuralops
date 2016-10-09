use common::{CommonResources, CommonOperatorOutput};
use kernels::softmax::{SoftmaxKernel};

use float::ord::{F32InfNan};
use iter_utils::{argmax}; //, KahanSum};
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use std::u32;

#[derive(Clone, Copy)]
pub struct ClassLossConfig {
  pub batch_sz:     usize,
  pub num_classes:  usize,
}

pub struct SoftmaxNLLClassLossOperator {
  cfg:      ClassLossConfig,
  in_:      CommonOperatorOutput<f32>,
  /*max_log:  Vec<f32>,
  facts:    Vec<f32>,
  sum_fact: Vec<f32>,*/
  hats:     Vec<u32>,
  losses:   Vec<f32>,
  loss1:    f32,
  accuracy: usize,
  labels:   Vec<u32>,
  weights:  Vec<f32>,
  sm_kern:  SoftmaxKernel,
  out:      CommonOperatorOutput<f32>,
}

impl SoftmaxNLLClassLossOperator {
  pub fn new(cfg: ClassLossConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> SoftmaxNLLClassLossOperator {
    /*let mut max_log = Vec::with_capacity(cfg.batch_sz);
    unsafe { max_log.set_len(cfg.batch_sz) };
    let mut facts = Vec::with_capacity(cfg.batch_sz * cfg.num_classes);
    unsafe { facts.set_len(cfg.batch_sz * cfg.num_classes) };
    let mut sum_fact = Vec::with_capacity(cfg.batch_sz);
    unsafe { sum_fact.set_len(cfg.batch_sz) };*/
    let mut hats = Vec::with_capacity(cfg.batch_sz);
    hats.resize(cfg.batch_sz, 0);
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut labels = Vec::with_capacity(cfg.batch_sz);
    labels.resize(cfg.batch_sz, 0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 0.0);
    SoftmaxNLLClassLossOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      /*max_log:  max_log,
      facts:    facts,
      sum_fact: sum_fact,*/
      hats:     hats,
      losses:   losses,
      loss1:    0.0,
      accuracy: 0,
      labels:   labels,
      weights:  weights,
      sm_kern:  SoftmaxKernel::new(cfg.batch_sz, cfg.num_classes, res.nnp_pool),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.num_classes, cap),
    }
  }
}

impl<S> DiffOperatorInput<f32, S> for SoftmaxNLLClassLossOperator where S: SampleLabel + SampleLossWeight<ClassLoss> {
  fn load_data(&mut self, samples: &[S]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if let Some(cat) = sample.class() {
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        self.labels[idx] = cat;
      } else {
        self.labels[idx] = u32::MAX;
      }
      self.weights[idx] = sample.weight().unwrap_or(1.0);
    }
    *self.out.batch_size.borrow_mut() = actual_batch_size;
  }
}

impl DiffOperator<f32> for SoftmaxNLLClassLossOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn reset_loss(&mut self) {
    self.loss1 = 0.0;
    self.accuracy = 0;
  }

  fn store_loss(&mut self) -> f32 {
    self.loss1
  }

  fn forward(&mut self, _phase: OpPhase) {
    //self.out.batch_size = self.in_.batch_size;
    let batch_size = *self.in_.batch_size.borrow();
    assert_eq!(batch_size, *self.out.batch_size.borrow());

    /*for idx in 0 .. batch_size {
      let range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(self.in_.out_buf.borrow()[range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      let max_logit = self.in_.out_buf.borrow()[idx * self.cfg.num_classes + max_logit_k];
      self.max_log[idx] = max_logit;
      self.hats[idx] = max_logit_k as u32;
      for k in 0 .. self.cfg.num_classes {
        self.facts[idx * self.cfg.num_classes + k] = (self.in_.out_buf.borrow()[idx * self.cfg.num_classes + k] - max_logit).exp();
      }
      let sum_fact: f32 = Sum::sum(self.facts[range].iter().map(|&x| x));
      for k in 0 .. self.cfg.num_classes {
        self.out.out_buf.borrow_mut()[idx * self.cfg.num_classes + k] = self.facts[idx * self.cfg.num_classes + k] / sum_fact;
      }
      self.losses[idx] = -self.weights[idx] * self.out.out_buf.borrow()[idx * self.cfg.num_classes + self.labels[idx] as usize].ln();
    }*/
    let in_buf = self.in_.out_buf.borrow();
    //println!("DEBUG: softmax: in buf: {:?}", &in_buf[ .. 10]);
    let mut out_buf = self.out.out_buf.borrow_mut();
    self.sm_kern.forward(batch_size, &*in_buf, &mut *out_buf);
    let mut batch_loss = 0.0;
    let mut batch_accuracy = 0;
    for idx in 0 .. batch_size {
      let idx_range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(in_buf[idx_range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      self.hats[idx] = max_logit_k as u32;
      if self.hats[idx] == self.labels[idx] {
        batch_accuracy += 1;
      }
      let loss = if self.labels[idx] == u32::MAX {
        0.0
      } else {
        -self.weights[idx] * out_buf[idx * self.cfg.num_classes + self.labels[idx] as usize].ln()
      };
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    //self.loss1 += Sum::sum(self.losses[ .. batch_size].iter().map(|&x| x));
    //println!("DEBUG: softmax: out buf: {:?}", &out_buf[ .. 10]);
    //println!("DEBUG: softmax: accuracy: {}/{}", batch_accuracy, batch_size);

    let in_loss = *self.in_.out_loss.borrow();
    //self.loss1 += batch_loss + in_loss;
    self.loss1 += batch_loss;
    self.accuracy += batch_accuracy;
    *self.out.out_loss.borrow_mut() = batch_loss + in_loss;
  }

  fn backward(&mut self) {
    //assert_eq!(self.out.batch_size, self.in_.batch_size);
    let batch_size = *self.out.batch_size.borrow();
    if let Some(ref mut in_grad) = self.in_.out_grad.as_mut() {
      let out_buf = self.out.out_buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      let mut p = 0;
      for idx in 0 .. batch_size {
        for k in 0 .. self.cfg.num_classes {
          in_grad[p] =
              self.weights[idx] *
              (out_buf[p] - if k == self.labels[idx] as usize { 1.0 } else { 0.0 });
          p += 1;
        }
      }
    }
  }
}

#[derive(Clone, Copy)]
pub struct EntRegSoftmaxNLLClassLossConfig {
  pub batch_sz:     usize,
  pub num_classes:  usize,
  pub entropy_coef: f32,
}

pub struct EntRegSoftmaxNLLClassLossOperator {
  cfg:      EntRegSoftmaxNLLClassLossConfig,
  in_:      CommonOperatorOutput<f32>,
  /*max_log:  Vec<f32>,
  facts:    Vec<f32>,
  sum_fact: Vec<f32>,*/
  hats:     Vec<u32>,
  losses:   Vec<f32>,
  ents:     Vec<f32>,
  loss1:    f32,
  accuracy: usize,
  labels:   Vec<u32>,
  weights:  Vec<f32>,
  sm_kern:  SoftmaxKernel,
  out:      CommonOperatorOutput<f32>,
}

impl EntRegSoftmaxNLLClassLossOperator {
  pub fn new(cfg: EntRegSoftmaxNLLClassLossConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> EntRegSoftmaxNLLClassLossOperator {
    /*let mut max_log = Vec::with_capacity(cfg.batch_sz);
    unsafe { max_log.set_len(cfg.batch_sz) };
    let mut facts = Vec::with_capacity(cfg.batch_sz * cfg.num_classes);
    unsafe { facts.set_len(cfg.batch_sz * cfg.num_classes) };
    let mut sum_fact = Vec::with_capacity(cfg.batch_sz);
    unsafe { sum_fact.set_len(cfg.batch_sz) };*/
    let mut hats = Vec::with_capacity(cfg.batch_sz);
    hats.resize(cfg.batch_sz, 0);
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut ents = Vec::with_capacity(cfg.batch_sz);
    ents.resize(cfg.batch_sz, 0.0);
    let mut labels = Vec::with_capacity(cfg.batch_sz);
    labels.resize(cfg.batch_sz, 0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 0.0);
    EntRegSoftmaxNLLClassLossOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      /*max_log:  max_log,
      facts:    facts,
      sum_fact: sum_fact,*/
      hats:     hats,
      losses:   losses,
      ents:     ents,
      loss1:    0.0,
      accuracy: 0,
      labels:   labels,
      weights:  weights,
      sm_kern:  SoftmaxKernel::new(cfg.batch_sz, cfg.num_classes, res.nnp_pool),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.num_classes, cap),
    }
  }
}

impl<S> DiffOperatorInput<f32, S> for EntRegSoftmaxNLLClassLossOperator where S: SampleLabel + SampleLossWeight<ClassLoss> {
  fn load_data(&mut self, samples: &[S]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if let Some(cat) = sample.class() {
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        self.labels[idx] = cat;
      } else {
        self.labels[idx] = u32::MAX;
      }
      self.weights[idx] = sample.weight().unwrap_or(1.0);
    }
    *self.out.batch_size.borrow_mut() = actual_batch_size;
  }
}

impl DiffOperator<f32> for EntRegSoftmaxNLLClassLossOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn reset_loss(&mut self) {
    self.loss1 = 0.0;
    self.accuracy = 0;
  }

  fn store_loss(&mut self) -> f32 {
    self.loss1
  }

  fn forward(&mut self, _phase: OpPhase) {
    //self.out.batch_size = self.in_.batch_size;
    let batch_size = *self.in_.batch_size.borrow();
    assert_eq!(batch_size, *self.out.batch_size.borrow());

    /*for idx in 0 .. batch_size {
      let range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(self.in_.out_buf.borrow()[range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      let max_logit = self.in_.out_buf.borrow()[idx * self.cfg.num_classes + max_logit_k];
      self.max_log[idx] = max_logit;
      self.hats[idx] = max_logit_k as u32;
      for k in 0 .. self.cfg.num_classes {
        self.facts[idx * self.cfg.num_classes + k] = (self.in_.out_buf.borrow()[idx * self.cfg.num_classes + k] - max_logit).exp();
      }
      let sum_fact: f32 = Sum::sum(self.facts[range].iter().map(|&x| x));
      for k in 0 .. self.cfg.num_classes {
        self.out.out_buf.borrow_mut()[idx * self.cfg.num_classes + k] = self.facts[idx * self.cfg.num_classes + k] / sum_fact;
      }
      self.losses[idx] = -self.weights[idx] * self.out.out_buf.borrow()[idx * self.cfg.num_classes + self.labels[idx] as usize].ln();
    }*/
    let in_buf = self.in_.out_buf.borrow();
    //println!("DEBUG: softmax: in buf: {:?}", &in_buf[ .. 10]);
    let mut out_buf = self.out.out_buf.borrow_mut();
    self.sm_kern.forward(batch_size, &*in_buf, &mut *out_buf);
    let mut batch_loss = 0.0;
    let mut batch_ent = 0.0;
    let mut batch_accuracy = 0;
    for idx in 0 .. batch_size {
      let idx_range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(in_buf[idx_range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      self.hats[idx] = max_logit_k as u32;
      if self.hats[idx] == self.labels[idx] {
        batch_accuracy += 1;
      }
      let (loss, ent) = if self.labels[idx] == u32::MAX {
        (0.0, 0.0)
      } else {
        let p = out_buf[idx * self.cfg.num_classes + self.labels[idx] as usize];
        let loss = -self.weights[idx] * p.ln();
        let mut ent = 0.0;
        for k in 0 .. self.cfg.num_classes {
          let p_k = out_buf[idx * self.cfg.num_classes + k];
          ent -= p_k * p_k.ln();
        }
        (loss, ent)
      };
      self.losses[idx] = loss;
      self.ents[idx] = ent;
      batch_loss += loss;
      batch_ent += ent;
    }
    //self.loss1 += Sum::sum(self.losses[ .. batch_size].iter().map(|&x| x));
    //println!("DEBUG: softmax: out buf: {:?}", &out_buf[ .. 10]);
    //println!("DEBUG: softmax: accuracy: {}/{}", batch_accuracy, batch_size);

    let in_loss = *self.in_.out_loss.borrow();
    //self.loss1 += batch_loss + in_loss;
    self.loss1 += batch_loss - self.cfg.entropy_coef * batch_ent;
    self.accuracy += batch_accuracy;
    *self.out.out_loss.borrow_mut() = batch_loss + in_loss;
  }

  fn backward(&mut self) {
    //assert_eq!(self.out.batch_size, self.in_.batch_size);
    let batch_size = *self.out.batch_size.borrow();
    if let Some(ref mut in_grad) = self.in_.out_grad.as_mut() {
      let out_buf = self.out.out_buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      let mut p = 0;
      for idx in 0 .. batch_size {
        for k in 0 .. self.cfg.num_classes {
          let y = out_buf[p];
          let w = self.weights[idx];
          let k_truth = self.labels[idx];
          let delta_k = if k == k_truth as usize { 1.0 } else { 0.0 };
          let ent = self.ents[idx];
          if y.is_nan() {
            // FIXME(20161009): try to gracefully handle NaNs?
            unimplemented!();
          } else if y == 0.0 {
            in_grad[p] = -w * delta_k;
          } else {
            in_grad[p] =
                w * (y - delta_k) + self.cfg.entropy_coef * y * (y.ln() + ent);
          }
          p += 1;
        }
      }
    }
  }
}
