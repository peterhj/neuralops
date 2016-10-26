use prelude::*;
use common::{CommonResources, CommonOperatorOutput};

use float::ord::{F32InfNan};
use iter_utils::{argmax};
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use std::f32::consts::{PI};
use std::u32;
use std::cell::{RefCell, Ref};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct RegressLossConfig {
  pub batch_sz:     usize,
}

pub struct LstSqRegressLossOperator {
  cfg:      RegressLossConfig,
  in_:      CommonOperatorOutput<f32>,
  losses:   Vec<f32>,
  loss1:    f32,
  targets:  Vec<f32>,
  weights:  Vec<f32>,
  out:      CommonOperatorOutput<f32>,
}

impl LstSqRegressLossOperator {
  pub fn new(cfg: RegressLossConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, _res: CommonResources) -> LstSqRegressLossOperator {
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    unsafe { losses.set_len(cfg.batch_sz) };
    let mut targets = Vec::with_capacity(cfg.batch_sz);
    unsafe { targets.set_len(cfg.batch_sz) };
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    unsafe { weights.set_len(cfg.batch_sz) };
    LstSqRegressLossOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      losses:   losses,
      loss1:    0.0,
      targets:  targets,
      weights:  weights,
      out:      CommonOperatorOutput::new(cfg.batch_sz, 1, cap),
    }
  }
}

impl<S> DiffOperatorInput<f32, S> for LstSqRegressLossOperator where S: SampleLabel + SampleLossWeight<RegressLoss> {
  fn load_data(&mut self, samples: &[S]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      //if let Some(target) = sample.scalar_target() {
      if let Some(target) = sample.target() {
        self.targets[idx] = target;
      } else {
        self.targets[idx] = 0.0;
      }
      //self.weights[idx] = sample.scalar_target_weight().unwrap_or(1.0);
      self.weights[idx] = sample.weight().unwrap_or(1.0);
    }
    *self.out.batch_size.borrow_mut() = actual_batch_size;
  }
}

impl DiffOperator<f32> for LstSqRegressLossOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn reset_loss(&mut self) {
    self.loss1 = 0.0;
  }

  fn forward(&mut self, _phase: OpPhase) {
    //self.out.batch_size = self.in_.batch_size;
    let batch_size = *self.in_.batch_size.borrow();
    assert_eq!(batch_size, *self.out.batch_size.borrow());
    let in_buf = self.in_.out_buf.borrow();
    let mut out_buf = self.out.out_buf.borrow_mut();
    out_buf[ .. batch_size].copy_from_slice(&in_buf[ .. batch_size]);
    let mut batch_loss = 0.0;
    for idx in 0 .. batch_size {
      let dx = in_buf[idx] - self.targets[idx];
      let loss = 0.5 * self.weights[idx] * dx * dx;
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    //let in_loss = *self.in_.out_loss.borrow();
    self.loss1 += batch_loss;
    *self.out.out_loss.borrow_mut() = self.loss1;
  }

  fn backward(&mut self) {
    //assert_eq!(self.out.batch_size, self.in_.batch_size);
    let batch_size = *self.in_.batch_size.borrow();
    if let Some(ref mut in_grad) = self.in_.out_grad.as_mut() {
      let in_buf = self.in_.out_buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      for idx in 0 .. batch_size {
        in_grad[idx] = self.weights[idx] * (in_buf[idx] - self.targets[idx]);
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct NormLstSqRegressLossConfig {
  pub batch_sz: usize,
  pub avg_rate: f32,
  pub epsilon:  f32,
  pub init_var: f32,
}

pub struct NormLstSqRegressLossOperator {
  cfg:      NormLstSqRegressLossConfig,
  in_:      CommonOperatorOutput<f32>,
  nsamples: usize,
  var:      f32,
  run_var:  f32,
  losses:   Vec<f32>,
  loss1:    f32,
  targets:  Vec<f32>,
  weights:  Vec<f32>,
  out:      CommonOperatorOutput<f32>,
}

impl NormLstSqRegressLossOperator {
  pub fn new(cfg: NormLstSqRegressLossConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, _res: CommonResources) -> NormLstSqRegressLossOperator {
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    unsafe { losses.set_len(cfg.batch_sz) };
    let mut targets = Vec::with_capacity(cfg.batch_sz);
    unsafe { targets.set_len(cfg.batch_sz) };
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    unsafe { weights.set_len(cfg.batch_sz) };
    NormLstSqRegressLossOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      nsamples: 0,
      var:      0.0,
      run_var:  1.0,
      losses:   losses,
      loss1:    0.0,
      targets:  targets,
      weights:  weights,
      out:      CommonOperatorOutput::new(cfg.batch_sz, 1, cap),
    }
  }
}

impl<S> DiffOperatorInput<f32, S> for NormLstSqRegressLossOperator where S: SampleLabel + SampleLossWeight<RegressLoss> {
  fn load_data(&mut self, samples: &[S]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if let Some(target) = sample.target() {
        self.targets[idx] = target;
      } else {
        self.targets[idx] = 0.0;
      }
      self.weights[idx] = sample.weight().unwrap_or(1.0);
    }
    *self.out.batch_size.borrow_mut() = actual_batch_size;
  }
}

impl DiffOperator<f32> for NormLstSqRegressLossOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn nondiff_param_sz(&self) -> usize {
    1
  }

  fn init_param(&mut self, _rng: &mut Xorshiftplus128Rng) {
    self.nsamples = 0;
    self.var = 0.0;
    self.run_var = self.cfg.init_var;
  }

  fn update_nondiff_param(&mut self, _iter: usize) {
    self.run_var += self.cfg.avg_rate * (self.var / self.nsamples as f32 - self.run_var);
    self.nsamples = 0;
    self.var = 0.0;
  }

  fn reset_loss(&mut self) {
    self.loss1 = 0.0;
  }

  fn forward(&mut self, _phase: OpPhase) {
    //self.out.batch_size = self.in_.batch_size;
    let batch_size = *self.in_.batch_size.borrow();
    assert_eq!(batch_size, *self.out.batch_size.borrow());
    let in_buf = self.in_.out_buf.borrow();
    let mut out_buf = self.out.out_buf.borrow_mut();
    out_buf[ .. batch_size].copy_from_slice(&in_buf[ .. batch_size]);
    let v = self.run_var + self.cfg.epsilon;
    let loss_norm_term = 0.5 * (2.0 * PI * v).ln();
    let mut batch_loss = 0.0;
    for idx in 0 .. batch_size {
      let dx = in_buf[idx] - self.targets[idx];
      self.nsamples += 1;
      self.var += dx * dx;
      let loss = self.weights[idx] * (0.5 * dx * dx / v + loss_norm_term);
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    //let in_loss = *self.in_.out_loss.borrow();
    self.loss1 += batch_loss;
    *self.out.out_loss.borrow_mut() = self.loss1;
  }

  fn backward(&mut self) {
    //assert_eq!(self.out.batch_size, batch_size);
    let batch_size = *self.in_.batch_size.borrow();
    let v = self.run_var + self.cfg.epsilon;
    if let Some(ref mut in_grad) = self.in_.out_grad.as_mut() {
      let in_buf = self.in_.out_buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      for idx in 0 .. batch_size {
        in_grad[idx] = self.weights[idx] * (in_buf[idx] - self.targets[idx]) / v;
      }
    }
  }
}

pub struct LstSqRegressLoss<S> {
  cfg:      RegressLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  batch_nr: Option<usize>,
  losses:   Vec<f32>,
  acc_loss: f32,
  reg_loss: f32,
  targets:  Vec<f32>,
  weights:  Vec<f32>,
  preds:    Vec<f32>,
}

impl<S> LstSqRegressLoss<S> {
  pub fn new<InOp>(cfg: RegressLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<LstSqRegressLoss<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut targets = Vec::with_capacity(cfg.batch_sz);
    targets.resize(cfg.batch_sz, 0.0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 1.0);
    let mut preds = Vec::with_capacity(cfg.batch_sz);
    preds.resize(cfg.batch_sz, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(LstSqRegressLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(1, 1, cap),
      batch_nr: None,
      losses:   losses,
      acc_loss: 0.0,
      reg_loss: 0.0,
      targets:  targets,
      weights:  weights,
      preds:    preds,
    }))
  }

  pub fn batch_probs(&self) -> Ref<[f32]> {
    self.out.buf.borrow()
  }
}

impl<S> Operator for LstSqRegressLoss<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for LstSqRegressLoss<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl DiffLoss<SampleItem> for LstSqRegressLoss<SampleItem> {
  fn reset_loss(&mut self) {
    self.acc_loss = 0.0;
    self.reg_loss = 0.0;
  }

  fn store_loss(&mut self) -> f32 {
    self.acc_loss + self.reg_loss
  }

  fn _get_pred(&mut self) -> &[f32] {
    &self.preds
  }

  /*fn _extract_pred(&mut self, output: &mut [f32]) {
    let batch_size = self.in_.batch_sz.get();
    output[ .. batch_size].copy_from_slice(&*self.in_.buf[ .. batch_size]);
  }*/
}

impl NewDiffOperator<SampleItem> for LstSqRegressLoss<SampleItem> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
  }

  fn _next_iteration(&mut self) {
    self.batch_nr = None;
  }

  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleRegressTargetKey>() {
        let target = *sample.kvs.get::<SampleRegressTargetKey>().unwrap();
        self.targets[idx] = target;
      } else {
        self.targets[idx] = 0.0;
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
    self.preds[ .. batch_size].copy_from_slice(&in_buf[ .. batch_size]);
    let mut batch_loss = 0.0;
    for idx in 0 .. batch_size {
      let dx = in_buf[idx] - self.targets[idx];
      let loss = 0.5 * self.weights[idx] * dx * dx;
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    self.acc_loss += batch_loss;

    if let Some(0) = self.batch_nr {
      let mut reg_loss = 0.0;
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        let block_out = block.borrow()._output(0);
        reg_loss += block_out.buf.borrow()[0];
      }*/
      self.reg_loss = reg_loss;
    }

    // FIXME(20161018): what to put in the output buffer? one or all losses?
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref mut in_grad) = self.in_.grad.as_mut() {
      let in_buf = self.in_.buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      for idx in 0 .. batch_size {
        in_grad[idx] = self.weights[idx] * (in_buf[idx] - self.targets[idx]);
      }
    }
  }
}

pub struct NormLstSqRegressLoss<S> {
  cfg:      NormLstSqRegressLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  batch_nr: Option<usize>,
  losses:   Vec<f32>,
  acc_loss: f32,
  reg_loss: f32,
  targets:  Vec<f32>,
  weights:  Vec<f32>,
  preds:    Vec<f32>,
  nsamples: usize,
  var:      f32,
  run_var:  f32,
  run_norm: f32,
}

impl<S> NormLstSqRegressLoss<S> {
  pub fn new<InOp>(cfg: NormLstSqRegressLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NormLstSqRegressLoss<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut targets = Vec::with_capacity(cfg.batch_sz);
    targets.resize(cfg.batch_sz, 0.0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 1.0);
    let mut preds = Vec::with_capacity(cfg.batch_sz);
    preds.resize(cfg.batch_sz, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(NormLstSqRegressLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(1, 1, cap),
      batch_nr: None,
      losses:   losses,
      acc_loss: 0.0,
      reg_loss: 0.0,
      targets:  targets,
      weights:  weights,
      preds:    preds,
      nsamples: 0,
      var:      0.0,
      run_var:  0.0, //cfg.init_var,
      run_norm: 0.0,
    }))
  }

  pub fn batch_probs(&self) -> Ref<[f32]> {
    self.out.buf.borrow()
  }
}

impl<S> Operator for NormLstSqRegressLoss<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for NormLstSqRegressLoss<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl DiffLoss<SampleItem> for NormLstSqRegressLoss<SampleItem> {
  fn reset_loss(&mut self) {
    self.acc_loss = 0.0;
    self.reg_loss = 0.0;
  }

  fn store_loss(&mut self) -> f32 {
    self.acc_loss + self.reg_loss
  }

  fn _get_pred(&mut self) -> &[f32] {
    &self.preds
  }

  /*fn _extract_pred(&mut self, output: &mut [f32]) {
    let batch_size = self.in_.batch_sz.get();
    output[ .. batch_size].copy_from_slice(&*self.in_.buf[ .. batch_size]);
  }*/
}

impl NewDiffOperator<SampleItem> for NormLstSqRegressLoss<SampleItem> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
  }

  fn _next_iteration(&mut self) {
    self.batch_nr = None;
  }

  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleRegressTargetKey>() {
        let target = *sample.kvs.get::<SampleRegressTargetKey>().unwrap();
        self.targets[idx] = target;
      } else {
        self.targets[idx] = 0.0;
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

  fn _update_nondiff_param(&mut self, iter: usize) {
    /*if iter == 0 {
      self.run_var = self.var / self.nsamples as f32;
    } else {
      self.run_var = self.run_var + self.cfg.avg_rate * (self.var - self.run_var / self.nsamples as f32);
    }*/
    self.run_var = self.run_var + self.cfg.avg_rate * (self.var - self.run_var / self.nsamples as f32);
    self.run_norm = 1.0 / (1.0 - (1.0 - self.cfg.avg_rate).powi((iter + 1) as i32));
    self.nsamples = 0;
    self.var = 0.0;
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert_eq!(batch_size, self.out.batch_sz.get());

    let lambda = 1.0 / (self.run_norm * self.run_var + self.cfg.epsilon * self.cfg.epsilon);
    let in_buf = self.in_.buf.borrow();
    self.preds[ .. batch_size].copy_from_slice(&in_buf[ .. batch_size]);
    let mut batch_loss = 0.0;
    for idx in 0 .. batch_size {
      let dx = in_buf[idx] - self.targets[idx];
      let dx2 = dx * dx;
      self.var += dx2;
      let loss = 0.5 * lambda * self.weights[idx] * dx2;
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    self.nsamples += batch_size;
    self.acc_loss += batch_loss;

    if let Some(0) = self.batch_nr {
      let mut reg_loss = 0.0;
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        let block_out = block.borrow()._output(0);
        reg_loss += block_out.buf.borrow()[0];
      }*/
      self.reg_loss = reg_loss;
    }

    // FIXME(20161018): what to put in the output buffer? one or all losses?
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref mut in_grad) = self.in_.grad.as_mut() {
      let in_buf = self.in_.buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      let lambda = 1.0 / (self.run_norm * self.run_var + self.cfg.epsilon * self.cfg.epsilon);
      for idx in 0 .. batch_size {
        in_grad[idx] = lambda * self.weights[idx] * (in_buf[idx] - self.targets[idx]);
      }
    }
  }
}

pub struct IndLstSqRegressLoss<S> {
  cfg:      ClassLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  batch_nr: Option<usize>,
  losses:   Vec<f32>,
  acc_loss: f32,
  reg_loss: f32,
  targets:  Vec<f32>,
  labels:   Vec<u32>,
  weights:  Vec<f32>,
  preds:    Vec<f32>,
}

impl<S> IndLstSqRegressLoss<S> {
  pub fn new<InOp>(cfg: ClassLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<IndLstSqRegressLoss<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut targets = Vec::with_capacity(cfg.batch_sz);
    targets.resize(cfg.batch_sz, 0.0);
    let mut labels = Vec::with_capacity(cfg.batch_sz);
    labels.resize(cfg.batch_sz, 0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 1.0);
    let mut preds = Vec::with_capacity(cfg.batch_sz * cfg.num_classes);
    preds.resize(cfg.batch_sz * cfg.num_classes, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(IndLstSqRegressLoss{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(1, 1, cap),
      batch_nr: None,
      losses:   losses,
      acc_loss: 0.0,
      reg_loss: 0.0,
      targets:  targets,
      labels:   labels,
      weights:  weights,
      preds:    preds,
    }))
  }

  pub fn batch_probs(&self) -> Ref<[f32]> {
    self.out.buf.borrow()
  }
}

impl<S> Operator for IndLstSqRegressLoss<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for IndLstSqRegressLoss<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl DiffLoss<SampleItem> for IndLstSqRegressLoss<SampleItem> {
  fn reset_loss(&mut self) {
    self.acc_loss = 0.0;
    self.reg_loss = 0.0;
  }

  fn store_loss(&mut self) -> f32 {
    self.acc_loss + self.reg_loss
  }

  fn _get_pred(&mut self) -> &[f32] {
    &self.preds
  }

  /*fn _extract_pred(&mut self, output: &mut [f32]) {
    let batch_size = self.in_.batch_sz.get();
    output[ .. batch_size].copy_from_slice(&*self.in_.buf[ .. batch_size]);
  }*/
}

impl NewDiffOperator<SampleItem> for IndLstSqRegressLoss<SampleItem> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<SampleItem, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    if let Some(0) = self.batch_nr {
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        apply(&mut *block.borrow_mut());
      }*/
    }
  }

  fn _next_iteration(&mut self) {
    self.batch_nr = None;
  }

  fn _load_batch(&mut self, samples: &[SampleItem]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if sample.kvs.contains::<SampleRegressTargetKey>() {
        let target = *sample.kvs.get::<SampleRegressTargetKey>().unwrap();
        self.targets[idx] = target;
      } else {
        self.targets[idx] = 0.0;
      }
      if sample.kvs.contains::<SampleClassLabelKey>() {
        let label = *sample.kvs.get::<SampleClassLabelKey>().unwrap();
        self.labels[idx] = label;
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
    self.preds[ .. batch_size * self.cfg.num_classes].copy_from_slice(&in_buf[ .. batch_size * self.cfg.num_classes]);
    let mut batch_loss = 0.0;
    for idx in 0 .. batch_size {
      let label_k = if self.labels[idx] != u32::MAX {
        self.labels[idx] as usize
      } else {
        argmax(in_buf[idx * self.cfg.num_classes .. (idx+1) * self.cfg.num_classes].iter().map(|&v| F32InfNan(v))).unwrap()
      };
      assert!(label_k < self.cfg.num_classes);
      let x = in_buf[idx * self.cfg.num_classes + label_k];
      let dx = x - self.targets[idx];
      let loss = 0.5 * self.weights[idx] * dx * dx;
      self.losses[idx] = loss;
      batch_loss += loss;
    }
    self.acc_loss += batch_loss;

    if let Some(0) = self.batch_nr {
      let mut reg_loss = 0.0;
      // FIXME(20161013): L2 reg.
      /*for block in self.blocks.iter() {
        let block_out = block.borrow()._output(0);
        reg_loss += block_out.buf.borrow()[0];
      }*/
      self.reg_loss = reg_loss;
    }

    // FIXME(20161018): what to put in the output buffer? one or all losses?
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(ref mut in_grad) = self.in_.grad.as_mut() {
      let in_buf = self.in_.buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      for idx in 0 .. batch_size {
        let label_k = if self.labels[idx] != u32::MAX {
          self.labels[idx] as usize
        } else {
          unreachable!();
        };
        assert!(label_k < self.cfg.num_classes);
        let x = in_buf[idx * self.cfg.num_classes + label_k];
        for k in 0 .. self.cfg.num_classes {
          if k == label_k {
            in_grad[idx * self.cfg.num_classes + k] = self.weights[idx] * (x - self.targets[idx]);
          } else {
            in_grad[idx * self.cfg.num_classes + k] = 0.0;
          }
        }
      }
    }
  }
}
