use common::{CommonResources, CommonOperatorOutput};

use float::ord::{F32InfNan};
use operator::prelude::*;
use operator::data::{SampleScalarTarget}; //, SampleWeight};
use rng::xorshift::{Xorshiftplus128Rng};

use std::f32::consts::{PI};

#[derive(Clone, Copy, Debug)]
pub struct RegressLossConfig {
  pub batch_sz:     usize,
}

pub struct LeastSquaresRegressLossOperator {
  cfg:      RegressLossConfig,
  in_:      CommonOperatorOutput<f32>,
  losses:   Vec<f32>,
  loss1:    f32,
  targets:  Vec<f32>,
  weights:  Vec<f32>,
  out:      CommonOperatorOutput<f32>,
}

impl LeastSquaresRegressLossOperator {
  pub fn new(cfg: RegressLossConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> LeastSquaresRegressLossOperator {
    let mut losses = Vec::with_capacity(cfg.batch_sz);
    unsafe { losses.set_len(cfg.batch_sz) };
    let mut targets = Vec::with_capacity(cfg.batch_sz);
    unsafe { targets.set_len(cfg.batch_sz) };
    let mut weights = Vec::with_capacity(cfg.batch_sz);
    unsafe { weights.set_len(cfg.batch_sz) };
    LeastSquaresRegressLossOperator{
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

impl<S> DiffOperatorInput<f32, S> for LeastSquaresRegressLossOperator where S: SampleScalarTarget<f32> {
  fn load_data(&mut self, samples: &[S]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if let Some(target) = sample.scalar_target() {
        self.targets[idx] = target;
      } else {
        self.targets[idx] = 0.0;
      }
      self.weights[idx] = sample.scalar_target_weight().unwrap_or(1.0);
    }
    self.out.batch_size = actual_batch_size;
  }
}

impl DiffOperator<f32> for LeastSquaresRegressLossOperator {
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
    self.out.batch_size = self.in_.batch_size;
    let in_buf = self.in_.out_buf.borrow();
    let mut out_buf = self.out.out_buf.borrow_mut();
    out_buf[ .. self.out.batch_size].copy_from_slice(&in_buf[ .. self.out.batch_size]);
    for idx in 0 .. self.out.batch_size {
      let dx = in_buf[idx] - self.targets[idx];
      let loss = 0.5 * self.weights[idx] * dx * dx;
      self.losses[idx] = loss;
      self.loss1 += loss;
    }
  }

  fn backward(&mut self) {
    assert_eq!(self.out.batch_size, self.in_.batch_size);
    if let Some(ref mut in_grad) = self.in_.out_grad.as_mut() {
      let out_buf = self.out.out_buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      for idx in 0 .. self.out.batch_size {
        in_grad[idx] = self.weights[idx] * (out_buf[idx] - self.targets[idx]);
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct NormLstSqRegressLossConfig {
  pub batch_sz: usize,
  pub avg_rate: f32,
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
  pub fn new(cfg: NormLstSqRegressLossConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> NormLstSqRegressLossOperator {
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

impl<S> DiffOperatorInput<f32, S> for NormLstSqRegressLossOperator where S: SampleScalarTarget<f32> {
  fn load_data(&mut self, samples: &[S]) {
    let actual_batch_size = samples.len();
    assert!(actual_batch_size <= self.cfg.batch_sz);
    for (idx, sample) in samples.iter().enumerate() {
      if let Some(target) = sample.scalar_target() {
        self.targets[idx] = target;
      } else {
        self.targets[idx] = 0.0;
      }
      self.weights[idx] = sample.scalar_target_weight().unwrap_or(1.0);
    }
    self.out.batch_size = actual_batch_size;
  }
}

impl DiffOperator<f32> for NormLstSqRegressLossOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn init_param(&mut self, _rng: &mut Xorshiftplus128Rng) {
    self.nsamples = 0;
    self.var = 0.0;
    self.run_var = 1.0;
  }

  fn update_nondiff_param(&mut self) {
    self.run_var += self.cfg.avg_rate * (self.var / self.nsamples as f32 - self.run_var);
    self.nsamples = 0;
    self.var = 0.0;
  }

  fn reset_loss(&mut self) {
    self.loss1 = 0.0;
  }

  fn forward(&mut self, _phase: OpPhase) {
    self.out.batch_size = self.in_.batch_size;
    let in_buf = self.in_.out_buf.borrow();
    let mut out_buf = self.out.out_buf.borrow_mut();
    out_buf[ .. self.out.batch_size].copy_from_slice(&in_buf[ .. self.out.batch_size]);
    let loss_norm_term = (2.0 * PI * self.run_var).ln();
    for idx in 0 .. self.out.batch_size {
      let dx = in_buf[idx] - self.targets[idx];
      self.nsamples += 1;
      self.var += dx * dx;
      let loss = self.weights[idx] * (0.5 * dx * dx / self.run_var + loss_norm_term);
      self.losses[idx] = loss;
      self.loss1 += loss;
    }
  }

  fn backward(&mut self) {
    assert_eq!(self.out.batch_size, self.in_.batch_size);
    if let Some(ref mut in_grad) = self.in_.out_grad.as_mut() {
      let out_buf = self.out.out_buf.borrow();
      let mut in_grad = in_grad.borrow_mut();
      for idx in 0 .. self.out.batch_size {
        in_grad[idx] = self.weights[idx] * (out_buf[idx] - self.targets[idx]) / self.run_var;
      }
    }
  }
}
