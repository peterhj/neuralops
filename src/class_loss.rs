use prelude::*;

use float::ord::{F32InfNan};
use iter_utils::{argmax};
use operator::prelude::*;

use std::u32;
use std::cell::{RefCell, Ref};
use std::iter::{Sum};
use std::rc::{Rc};

#[derive(Clone, Copy)]
pub struct BinaryClassLossConfig {
  pub batch_sz:     usize,
}

#[derive(Clone, Copy)]
pub struct ClassLossConfig {
  pub batch_sz:     usize,
  pub num_classes:  usize,
}

pub struct SoftmaxNLLClassLoss<S, IoBuf: ?Sized> {
  cfg:      ClassLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  batch_nr: Option<usize>,
  //max_log:  Vec<f32>,
  facts:    Vec<f32>,
  //sum_fact: Vec<f32>,
  hats:     Vec<u32>,
  losses:   Vec<f32>,
  r_losses: Vec<f32>,
  labels:   Vec<u32>,
  weights:  Vec<f32>,
  jac_mix:  Vec<f32>,
  nsamples: usize,
  acc_loss: f32,
  reg_loss: f32,
  accuracy: usize,
}

impl<S, IoBuf: ?Sized> SoftmaxNLLClassLoss<S, IoBuf> {
  pub fn new<InOp>(cfg: ClassLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<SoftmaxNLLClassLoss<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    /*let mut max_log = Vec::with_capacity(cfg.batch_sz);
    max_log.resize(cfg.batch_sz, 0.0);*/
    let mut facts = Vec::with_capacity(cfg.batch_sz * cfg.num_classes);
    facts.resize(cfg.batch_sz * cfg.num_classes, 0.0);
    /*let mut sum_fact = Vec::with_capacity(cfg.batch_sz);
    sum_fact.resize(cfg.batch_sz, 0.0);*/
    let mut hats = Vec::with_capacity(cfg.batch_sz);
    hats.resize(cfg.batch_sz, 0);
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut r_losses = Vec::with_capacity(cfg.batch_sz);
    r_losses.resize(cfg.batch_sz, 0.0);
    let mut labels = Vec::with_capacity(cfg.batch_sz);
    labels.resize(cfg.batch_sz, 0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 1.0);
    let mut jac_mix = Vec::with_capacity(cfg.batch_sz);
    jac_mix.resize(cfg.batch_sz, 1.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(SoftmaxNLLClassLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.num_classes, cap),
      batch_nr: None,
      //max_log:  max_log,
      facts:    facts,
      //sum_fact: sum_fact,
      hats:     hats,
      losses:   losses,
      r_losses: r_losses,
      labels:   labels,
      weights:  weights,
      jac_mix:  jac_mix,
      nsamples: 0,
      acc_loss: 0.0,
      reg_loss: 0.0,
      accuracy: 0,
    }))
  }

  /*pub fn batch_probs(&self) -> Ref<[f32]> {
    self.out.buf.borrow()
  }*/

  pub fn reset_jacobian_mixing(&mut self) {
    for idx in 0 .. self.cfg.batch_sz {
      self.jac_mix[idx] = 1.0;
    }
  }

  pub fn set_jacobian_mixing_with_r_loss(&mut self) {
    // FIXME(20161108): useful for natural gradient.
    unimplemented!();
  }
}

impl<S, IoBuf: ?Sized> Operator for SoftmaxNLLClassLoss<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for SoftmaxNLLClassLoss<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<IoBuf: ?Sized> DiffLoss<SampleItem, IoBuf> for SoftmaxNLLClassLoss<SampleItem, IoBuf> {
  fn reset_loss(&mut self) {
    self.nsamples = 0;
    self.acc_loss = 0.0;
    self.reg_loss = 0.0;
    self.accuracy = 0;
  }

  fn store_loss(&mut self) -> f32 {
    self.acc_loss + self.reg_loss
  }

  fn _store_accuracy(&mut self) -> usize {
    self.accuracy
  }

  fn _get_pred(&mut self) -> &[f32] {
    &self.facts
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for SoftmaxNLLClassLoss<S, IoBuf> {
  default fn _load_batch(&mut self, samples: &[S]) {
    unimplemented!();
  }
}

impl<IoBuf: ?Sized> DiffOperatorData<SampleItem> for SoftmaxNLLClassLoss<SampleItem, IoBuf> {
  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleClassLabelKey>() {
        let cat = *sample.kvs.get::<SampleClassLabelKey>().unwrap();
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        self.labels[idx] = cat;
        //println!("DEBUG: load_batch: got label:  {:?}", cat);
      } else {
        self.labels[idx] = u32::MAX;
      }
      if sample.kvs.contains::<SampleWeightKey>() {
        let weight = *sample.kvs.get::<SampleWeightKey>().unwrap();
        //println!("DEBUG: load_batch: got weight: {:?}", weight);
        self.weights[idx] = weight;
      } else {
        self.weights[idx] = 1.0;
      }
      /*if let Some(cat) = sample.class() {
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        self.labels[idx] = cat;
      } else {
        self.labels[idx] = u32::MAX;
      }
      // FIXME(20161013): sample trait bounds.
      self.weights[idx] = 1.0;
      //self.weights[idx] = sample.weight().unwrap_or(1.0);*/
    }
    self.out.batch_sz.set(actual_batch_size);
    self.batch_nr = Some(self.batch_nr.map_or(0, |batch| batch + 1));
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for SoftmaxNLLClassLoss<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for SoftmaxNLLClassLoss<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    self.node.pop(epoch);
  }

  fn _next_iteration(&mut self) {
    self.batch_nr = None;
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert_eq!(batch_size, self.out.batch_sz.get());

    let in_buf = self.in_.buf.borrow();
    //let mut out_buf = self.out.buf.borrow_mut();
    //println!("DEBUG: softmax: input: {:?}", &in_buf[ .. self.cfg.num_classes]);
    for idx in 0 .. batch_size {
      let range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(in_buf[range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      let max_logit = in_buf[idx * self.cfg.num_classes + max_logit_k];
      //self.max_log[idx] = max_logit;
      self.hats[idx] = max_logit_k as u32;
      for k in 0 .. self.cfg.num_classes {
        self.facts[idx * self.cfg.num_classes + k] = (in_buf[idx * self.cfg.num_classes + k] - max_logit).exp();
      }
      let sum_fact: f32 = Sum::sum(self.facts[range].iter().map(|&x| x));
      for k in 0 .. self.cfg.num_classes {
        self.facts[idx * self.cfg.num_classes + k] /= sum_fact;
      }
    }

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
        -self.weights[idx] * self.jac_mix[idx] * self.facts[idx * self.cfg.num_classes + self.labels[idx] as usize].ln()
      };
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    self.nsamples += batch_size;
    self.acc_loss += batch_loss;
    self.accuracy += batch_accuracy;

    if let Some(0) = self.batch_nr {
      let mut reg_loss = 0.0;
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        let block_out = block.borrow()._output(0);
        reg_loss += block_out.buf.borrow()[0];
      }*/
      self.reg_loss += reg_loss;
    }

    // FIXME(20161018): what to put in the output buffer? one or all losses?
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref mut in_grad) = self.in_.grad.as_mut() {
      //let out_buf = self.out.buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      let mut p = 0;
      for idx in 0 .. batch_size {
        for k in 0 .. self.cfg.num_classes {
          in_grad[p] =
              self.weights[idx] * self.jac_mix[idx]
              * (self.facts[p] - if k == self.labels[idx] as usize { 1.0 } else { 0.0 });
          p += 1;
        }
      }
    }
  }

  fn _r_forward(&mut self) {
    // FIXME(20161108)
    unimplemented!();
  }
}

impl<S, IoBuf: ?Sized> LossReport<ClassLossStats> for SoftmaxNLLClassLoss<S, IoBuf> {
  fn update_stats(&mut self, iter_nr: usize, stats: &mut ClassLossStats) {
    //let batch_size = self.out.batch_sz.get();
    stats.iter_nr = iter_nr;
    stats.sample_count += self.nsamples;
    stats.correct_count += self.accuracy;
    stats.accum_loss += self.acc_loss + self.reg_loss;
  }
}

#[derive(Clone, Copy)]
pub struct EntRegSoftmaxNLLClassLossConfig {
  pub batch_sz:     usize,
  pub num_classes:  usize,
  pub entropy_coef: f32,
}

pub struct EntRegSoftmaxNLLClassLoss<S, IoBuf: ?Sized> {
  cfg:      EntRegSoftmaxNLLClassLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  batch_nr: Option<usize>,
  //max_log:  Vec<f32>,
  facts:    Vec<f32>,
  //sum_fact: Vec<f32>,
  hats:     Vec<u32>,
  ents:     Vec<f32>,
  losses:   Vec<f32>,
  nsamples: usize,
  acc_loss: f32,
  reg_loss: f32,
  accuracy: usize,
  labels:   Vec<u32>,
  weights:  Vec<f32>,
}

/*impl<S> EntRegSoftmaxNLLClassLoss<S> {
  pub fn new<InOp>(cfg: EntRegSoftmaxNLLClassLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<EntRegSoftmaxNLLClassLoss<S>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf=[f32]> {
    /*let mut max_log = Vec::with_capacity(cfg.batch_sz);
    max_log.resize(cfg.batch_sz, 0.0);*/
    let mut facts = Vec::with_capacity(cfg.batch_sz * cfg.num_classes);
    facts.resize(cfg.batch_sz * cfg.num_classes, 0.0);
    /*let mut sum_fact = Vec::with_capacity(cfg.batch_sz);
    sum_fact.resize(cfg.batch_sz, 0.0);*/
    let mut hats = Vec::with_capacity(cfg.batch_sz);
    hats.resize(cfg.batch_sz, 0);
    let mut ents = Vec::with_capacity(cfg.batch_sz);
    ents.resize(cfg.batch_sz, 0.0);
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut labels = Vec::with_capacity(cfg.batch_sz);
    labels.resize(cfg.batch_sz, 0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 1.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(EntRegSoftmaxNLLClassLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.num_classes, cap),
      batch_nr: None,
      //max_log:  max_log,
      facts:    facts,
      //sum_fact: sum_fact,
      hats:     hats,
      ents:     ents,
      losses:   losses,
      nsamples: 0,
      acc_loss: 0.0,
      reg_loss: 0.0,
      accuracy: 0,
      labels:   labels,
      weights:  weights,
    }))
  }

  pub fn batch_probs(&self) -> Ref<[f32]> {
    self.out.buf.borrow()
  }
}

impl<S> Operator for EntRegSoftmaxNLLClassLoss<S> /*where S: SampleLabel*/ {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for EntRegSoftmaxNLLClassLoss<S> /*where S: SampleLabel*/ {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl DiffLoss<SampleItem> for EntRegSoftmaxNLLClassLoss<SampleItem> {
  fn reset_loss(&mut self) {
    self.nsamples = 0;
    self.acc_loss = 0.0;
    self.reg_loss = 0.0;
    self.accuracy = 0;
  }

  fn store_loss(&mut self) -> f32 {
    self.acc_loss + self.reg_loss
  }

  fn _store_accuracy(&mut self) -> usize {
    self.accuracy
  }

  fn _get_pred(&mut self) -> &[f32] {
    &self.facts
  }
}

impl DiffOperator<SampleItem> for EntRegSoftmaxNLLClassLoss<SampleItem> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<SampleItem, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<SampleItem, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    self.node.pop(epoch);
  }

  fn _next_iteration(&mut self) {
    self.batch_nr = None;
  }

  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleClassLabelKey>() {
        let cat = *sample.kvs.get::<SampleClassLabelKey>().unwrap();
        assert!(cat < self.cfg.num_classes as u32);
        assert!(cat != u32::MAX);
        self.labels[idx] = cat;
      } else {
        self.labels[idx] = u32::MAX;
      }
      if sample.kvs.contains::<SampleWeightKey>() {
        let weight = *sample.kvs.get::<SampleWeightKey>().unwrap();
        self.weights[idx] = weight;
      } else {
        self.weights[idx] = 1.0;
      }
    }
    self.out.batch_sz.set(actual_batch_size);
    self.batch_nr = Some(self.batch_nr.map_or(0, |batch| batch + 1));
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert_eq!(batch_size, self.out.batch_sz.get());

    let in_buf = self.in_.buf.borrow();
    //let mut out_buf = self.out.buf.borrow_mut();
    //println!("DEBUG: softmax: input: {:?}", &in_buf[ .. self.cfg.num_classes]);
    for idx in 0 .. batch_size {
      let range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(in_buf[range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      let max_logit = in_buf[idx * self.cfg.num_classes + max_logit_k];
      //self.max_log[idx] = max_logit;
      self.hats[idx] = max_logit_k as u32;
      for k in 0 .. self.cfg.num_classes {
        self.facts[idx * self.cfg.num_classes + k] = (in_buf[idx * self.cfg.num_classes + k] - max_logit).exp();
      }
      let sum_fact: f32 = Sum::sum(self.facts[range].iter().map(|&x| x));
      for k in 0 .. self.cfg.num_classes {
        //out_buf[idx * self.cfg.num_classes + k] = self.facts[idx * self.cfg.num_classes + k] / sum_fact;
        self.facts[idx * self.cfg.num_classes + k] /= sum_fact;
      }
    }

    //let mut batch_entropy = 0.0;
    let mut batch_loss = 0.0;
    let mut batch_accuracy = 0;
    for idx in 0 .. batch_size {
      let mut entropy = 0.0;
      for k in 0 .. self.cfg.num_classes {
        let j = idx * self.cfg.num_classes + k;
        let p = self.facts[j];
        let entropy_k = if p > 0.0 {
          -p * p.ln()
        } else if p == 0.0 {
          0.0
        } else {
          unreachable!();
        };
        entropy += entropy_k;
      }
      self.ents[idx] = entropy;
      let entropy_loss = self.cfg.entropy_coef * entropy;
      let idx_range = idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes;
      let max_logit_k = argmax(in_buf[idx_range.clone()].iter().map(|&x| F32InfNan(x))).unwrap();
      self.hats[idx] = max_logit_k as u32;
      if self.hats[idx] == self.labels[idx] {
        batch_accuracy += 1;
      }
      let loss = if self.labels[idx] == u32::MAX {
        entropy_loss
      } else {
        entropy_loss - self.weights[idx] * self.facts[idx * self.cfg.num_classes + self.labels[idx] as usize].ln()
      };
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    self.nsamples += batch_size;
    self.acc_loss += batch_loss;
    self.accuracy += batch_accuracy;

    if let Some(0) = self.batch_nr {
      let mut reg_loss = 0.0;
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        let block_out = block.borrow()._output(0);
        reg_loss += block_out.buf.borrow()[0];
      }*/
      self.reg_loss += reg_loss;
    }

    // FIXME(20161018): what to put in the output buffer? one or all losses?
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref mut in_grad) = self.in_.grad.as_mut() {
      //let out_buf = self.out.buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      let mut p = 0;
      for idx in 0 .. batch_size {
        for k in 0 .. self.cfg.num_classes {
          let entropy_self_grad = if self.facts[p] > 0.0 {
            -self.facts[p] * self.facts[p].ln()
          } else if self.facts[p] == 0.0 {
            0.0
          } else {
            unreachable!();
          };
          let entropy_grad = entropy_self_grad - self.facts[p] * self.ents[idx];
          let nll_grad = self.facts[p] - if k == self.labels[idx] as usize { 1.0 } else { 0.0 };
          in_grad[p] = self.cfg.entropy_coef * entropy_grad + self.weights[idx] * nll_grad;
          p += 1;
        }
      }
    }
  }
}*/

pub struct LogisticNLLClassLoss<S, IoBuf: ?Sized> {
  cfg:      BinaryClassLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  batch_nr: Option<usize>,
  facts:    Vec<f32>,
  hats:     Vec<u32>,
  losses:   Vec<f32>,
  r_losses: Vec<f32>,
  labels:   Vec<u32>,
  weights:  Vec<f32>,
  jac_mix:  Vec<f32>,
  nsamples: usize,
  acc_loss: f32,
  reg_loss: f32,
  accuracy: usize,
}

impl<S, IoBuf: ?Sized> LogisticNLLClassLoss<S, IoBuf> {
  pub fn new<InOp>(cfg: BinaryClassLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<LogisticNLLClassLoss<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    unimplemented!();
  }
}
