use prelude::*;

//use float::ord::{F32InfNan};
//use iter_utils::{argmax};
use operator::prelude::*;

//use std::f32::consts::{PI};
use std::u32;
use std::cell::{RefCell, Ref};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct RegressLossConfig {
  pub batch_sz:     usize,
}

#[derive(Clone, Copy, Debug)]
pub struct LstSqRegressLossConfig {
  pub batch_sz:     usize,
  pub grad_clip:    Option<f32>,
}

pub struct LstSqRegressLoss<S, IoBuf: ?Sized> {
  cfg:      LstSqRegressLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
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

impl<S, IoBuf: ?Sized> LstSqRegressLoss<S, IoBuf> {
  pub fn new<InOp>(cfg: LstSqRegressLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<LstSqRegressLoss<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
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

  /*pub fn batch_probs(&self) -> Ref<[f32]> {
    self.out.buf.borrow()
  }*/
}

impl<S, IoBuf: ?Sized> Operator for LstSqRegressLoss<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for LstSqRegressLoss<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<IoBuf: ?Sized> DiffLoss<SampleItem, IoBuf> for LstSqRegressLoss<SampleItem, IoBuf> {
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

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for LstSqRegressLoss<S, IoBuf> {
  default fn _load_batch(&mut self, samples: &[S]) {
    unimplemented!();
  }
}

impl<IoBuf: ?Sized> DiffOperatorData<SampleItem> for LstSqRegressLoss<SampleItem, IoBuf> {
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
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for LstSqRegressLoss<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for LstSqRegressLoss<S, IoBuf> {
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
        let mut g = in_buf[idx] - self.targets[idx];
        if let Some(grad_clip) = self.cfg.grad_clip {
          if g > grad_clip {
            g = grad_clip;
          } else if g < -grad_clip {
            g = -grad_clip
          }
        }
        in_grad[idx] = self.weights[idx] * g;
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

/*pub struct NormLstSqRegressLoss<S> {
  cfg:      NormLstSqRegressLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf=[f32]>>>,
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
  pub fn new<InOp>(cfg: NormLstSqRegressLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NormLstSqRegressLoss<S>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf=[f32]> {
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

impl DiffOperator<SampleItem> for NormLstSqRegressLoss<SampleItem> {
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
}*/

#[derive(Clone, Copy, Debug)]
pub struct IndLstSqRegressLossConfig {
  pub batch_sz:     usize,
  pub index_sz:     usize,
  pub grad_clip:    Option<f32>,
}

pub struct IndLstSqRegressLoss<S, IoBuf: ?Sized> {
  cfg:      IndLstSqRegressLossConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
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

impl<S, IoBuf: ?Sized> IndLstSqRegressLoss<S, IoBuf> {
  pub fn new<InOp>(cfg: IndLstSqRegressLossConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<IndLstSqRegressLoss<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    losses.resize(cfg.batch_sz, 0.0);
    let mut targets = Vec::with_capacity(cfg.batch_sz);
    targets.resize(cfg.batch_sz, 0.0);
    let mut labels = Vec::with_capacity(cfg.batch_sz);
    labels.resize(cfg.batch_sz, 0);
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    weights.resize(cfg.batch_sz, 1.0);
    let mut preds = Vec::with_capacity(cfg.batch_sz * cfg.index_sz);
    preds.resize(cfg.batch_sz * cfg.index_sz, 0.0);
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

  /*pub fn batch_probs(&self) -> Ref<[f32]> {
    self.out.buf.borrow()
  }*/
}

impl<S, IoBuf: ?Sized> Operator for IndLstSqRegressLoss<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for IndLstSqRegressLoss<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<IoBuf: ?Sized> DiffLoss<SampleItem, IoBuf> for IndLstSqRegressLoss<SampleItem, IoBuf> {
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

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for IndLstSqRegressLoss<S, IoBuf> {
  default fn _load_batch(&mut self, samples: &[S]) {
    unimplemented!();
  }
}

impl<IoBuf: ?Sized> DiffOperatorData<SampleItem> for IndLstSqRegressLoss<SampleItem, IoBuf> {
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
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for IndLstSqRegressLoss<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for IndLstSqRegressLoss<S, IoBuf> {
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
    self.preds[ .. batch_size * self.cfg.index_sz].copy_from_slice(&in_buf[ .. batch_size * self.cfg.index_sz]);
    let mut batch_loss = 0.0;
    for idx in 0 .. batch_size {
      let loss = if self.labels[idx] != u32::MAX {
        let label_k = self.labels[idx] as usize;
        assert!(label_k < self.cfg.index_sz);
        let x = in_buf[idx * self.cfg.index_sz + label_k];
        let dx = x - self.targets[idx];
        let loss = 0.5 * self.weights[idx] * dx * dx;
        loss
      } else {
        0.0
      };
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
        assert!(label_k < self.cfg.index_sz);
        let x = in_buf[idx * self.cfg.index_sz + label_k];
        for k in 0 .. self.cfg.index_sz {
          if k == label_k {
            in_grad[idx * self.cfg.index_sz + k] = self.weights[idx] * (x - self.targets[idx]);
          } else {
            in_grad[idx * self.cfg.index_sz + k] = 0.0;
          }
        }
      }
    }
  }
}
