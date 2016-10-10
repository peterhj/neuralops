use prelude::*;
use common::{CommonResources};
//use join::{AddJoinOperator};
use kernels::activate::{ActivateKernel};
use kernels::batchnorm::{BatchNorm2dKernel};
use kernels::conv::*;
use ops::*;
//use split::{CopySplitOperator};

use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array4d};
use densearray::linalg::{Transpose};
use nnpack::{NnpackHandle, NnpackPthreadPool};
use nnpack::ffi::*;
use operator::prelude::*;
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cmp::{max};
use std::ptr::{null_mut};
use std::rc::{Rc};

/*#[derive(Clone, Copy, Debug)]
pub enum Conv2dBackend {
  Auto,
  Nnpack,
  Mkl,
}

impl Default for Conv2dBackend {
  fn default() -> Conv2dBackend {
    Conv2dBackend::Nnpack
  }
}*/

#[derive(Clone, Copy, Debug)]
pub struct Conv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub kernel_w: usize,
  pub kernel_h: usize,
  pub stride_w: usize,
  pub stride_h: usize,
  pub pad_w:    usize,
  pub pad_h:    usize,
  pub out_chan: usize,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl Conv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, in_chan) = self.in_dim;
    /*let out_w = (self.pad_left  + self.pad_right  + in_w - self.kernel_w + 1) / self.stride_w;
    let out_h = (self.pad_bot   + self.pad_top    + in_h - self.kernel_h + 1) / self.stride_h;*/
    /*let out_w = (2 * self.pad_w + in_w - self.kernel_w + 1) / self.stride_w;
    let out_h = (2 * self.pad_h + in_h - self.kernel_h + 1) / self.stride_h;*/
    let out_w = max(0, (in_w + 2 * self.pad_w - self.kernel_w + self.stride_w) as isize) as usize / self.stride_w;
    let out_h = max(0, (in_h + 2 * self.pad_h - self.kernel_h + self.stride_h) as isize) as usize / self.stride_h;
    (out_w, out_h, self.out_chan)
  }
}

#[derive(Clone, Copy)]
pub struct BatchNormConv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub kernel_w: usize,
  pub kernel_h: usize,
  pub stride_w: usize,
  pub stride_h: usize,
  pub pad_w:    usize,
  pub pad_h:    usize,
  pub out_chan: usize,
  pub avg_rate: f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl BatchNormConv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, in_chan) = self.in_dim;
    /*let out_w = (2 * self.pad_w + in_w - self.kernel_w + 1) / self.stride_w;
    let out_h = (2 * self.pad_h + in_h - self.kernel_h + 1) / self.stride_h;*/
    let out_w = max(0, (in_w + 2 * self.pad_w - self.kernel_w + self.stride_w) as isize) as usize / self.stride_w;
    let out_h = max(0, (in_h + 2 * self.pad_h - self.kernel_h + self.stride_h) as isize) as usize / self.stride_h;
    (out_w, out_h, self.out_chan)
  }

  pub fn prefer_gemm_conv(&self) -> bool {
    //self.cfg.stride_w != 1 || self.cfg.stride_h != 1
    true
  }
}

#[derive(Clone, Copy, Debug)]
pub struct ResidualConv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub avg_rate: f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

pub struct ResidualConv2dOperator {
  cfg:      ResidualConv2dOperatorConfig,
  split:    CopySplitOperator,
  conv1:    BatchNormConv2dOperator,
  conv2:    BatchNormConv2dOperator,
  join:     AddJoinOperator,
  act_k:    ActivateKernel,
  out:      CommonOperatorOutput<f32>,
}

impl ResidualConv2dOperator {
  pub fn new(cfg: ResidualConv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> ResidualConv2dOperator {
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      cfg.in_dim.flat_len(),
    };
    let conv1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,
      kernel_h: 3,
      stride_w: 1,
      stride_h: 1,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.in_dim.2,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv2_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,
      kernel_h: 3,
      stride_w: 1,
      stride_h: 1,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.in_dim.2,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.in_dim.flat_len(),
    };
    let split = CopySplitOperator::new(split_cfg, cap, prev_op, prev_arm, res.clone());
    let conv1 = BatchNormConv2dOperator::new(conv1_cfg, cap, &split, 0, res.clone());
    let conv2 = BatchNormConv2dOperator::new(conv2_cfg, cap, &conv1, 0, res.clone());
    let join = AddJoinOperator::new(join_cfg, cap, &[(&split, 1), (&conv2, 0)], res.clone());
    let act_k = ActivateKernel::new(cfg.batch_sz, cfg.in_dim.flat_len(), cfg.act_kind, res.nnp_pool.clone());
    ResidualConv2dOperator{
      cfg:      cfg,
      split:    split,
      conv1:    conv1,
      conv2:    conv2,
      join:     join,
      act_k:    act_k,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.in_dim.flat_len(), cap),
    }
  }
}

impl DiffOperator<f32> for ResidualConv2dOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn diff_param_sz(&self) -> usize {
    self.conv1.diff_param_sz()
        + self.conv2.diff_param_sz()
  }

  fn nondiff_param_sz(&self) -> usize {
    self.conv1.nondiff_param_sz()
        + self.conv2.nondiff_param_sz()
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    self.conv1.init_param(rng);
    self.conv2.init_param(rng);
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.load_param(param_reader, offset);
    offset += self.conv2.load_param(param_reader, offset);
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.store_param(param_writer, offset);
    offset += self.conv2.store_param(param_writer, offset);
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.update_param(alpha, beta, grad_reader, offset);
    offset += self.conv2.update_param(alpha, beta, grad_reader, offset);
    offset - init_offset
  }

  fn update_nondiff_param(&mut self, iter: usize) {
    self.conv1.update_nondiff_param(iter);
    self.conv2.update_nondiff_param(iter);
  }

  fn reset_grad(&mut self) {
    self.conv1.reset_grad();
    self.conv2.reset_grad();
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    self.conv1.apply_grad_reg(reg);
    self.conv2.apply_grad_reg(reg);
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.store_grad(grad_writer, offset);
    offset += self.conv2.store_grad(grad_writer, offset);
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.accumulate_grad(alpha, beta, grad_accum, offset);
    offset += self.conv2.accumulate_grad(alpha, beta, grad_accum, offset);
    offset - init_offset
  }

  fn forward(&mut self, phase: OpPhase) {
    self.split.forward(phase);
    self.conv1.forward(phase);
    self.conv2.forward(phase);
    self.join.forward(phase);
    let join_out = self.join._output(0);
    let batch_size = *join_out.batch_size.borrow();
    self.act_k.forward(batch_size, &*join_out.out_buf.borrow(), &mut *self.out.out_buf.borrow_mut());
    *self.out.batch_size.borrow_mut() = batch_size;
  }

  fn backward(&mut self) {
    let batch_size = *self.out.batch_size.borrow();
    let join_out = self.join._output(0);
    self.act_k.backward(batch_size, &*join_out.out_buf.borrow(), &*self.out.out_grad.as_ref().unwrap().borrow(), &mut *join_out.out_grad.as_ref().unwrap().borrow_mut());
    self.join.backward();
    self.conv2.backward();
    self.conv1.backward();
    self.split.backward();
  }
}

#[derive(Clone, Copy, Debug)]
pub struct ProjResidualConv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub stride_w: usize,
  pub stride_h: usize,
  pub out_chan: usize,
  pub avg_rate: f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl ProjResidualConv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, _) = self.in_dim;
    let kernel_w = 3;
    let kernel_h = 3;
    let pad_w = 1;
    let pad_h = 1;
    let out_w = max(0, (in_w + 2 * pad_w - kernel_w + self.stride_w) as isize) as usize / self.stride_w;
    let out_h = max(0, (in_h + 2 * pad_h - kernel_h + self.stride_h) as isize) as usize / self.stride_h;
    (out_w, out_h, self.out_chan)
  }
}

pub struct ProjResidualConv2dOperator {
  cfg:      ProjResidualConv2dOperatorConfig,
  split:    CopySplitOperator,
  conv1:    BatchNormConv2dOperator,
  conv2:    BatchNormConv2dOperator,
  conv1x1:  BatchNormConv2dOperator,
  join:     AddJoinOperator,
  act_k:    ActivateKernel,
  out:      CommonOperatorOutput<f32>,
}

impl ProjResidualConv2dOperator {
  pub fn new(cfg: ProjResidualConv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> ProjResidualConv2dOperator {
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      cfg.in_dim.flat_len(),
    };
    let conv1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,
      kernel_h: 3,
      stride_w: cfg.stride_w,
      stride_h: cfg.stride_h,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv2_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.out_dim(),
      kernel_w: 3,
      kernel_h: 3,
      stride_w: 1,
      stride_h: 1,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let conv1x1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 1,
      kernel_h: 1,
      stride_w: cfg.stride_w,
      stride_h: cfg.stride_h,
      pad_w:    0,
      pad_h:    0,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.out_dim().flat_len(),
    };
    let split = CopySplitOperator::new(split_cfg, cap, prev_op, prev_arm, res.clone());
    let conv1 = BatchNormConv2dOperator::new(conv1_cfg, cap, &split, 0, res.clone());
    let conv2 = BatchNormConv2dOperator::new(conv2_cfg, cap, &conv1, 0, res.clone());
    let conv1x1 = BatchNormConv2dOperator::new(conv1x1_cfg, cap, &split, 1, res.clone());
    let join = AddJoinOperator::new(join_cfg, cap, &[(&conv2, 0), (&conv1x1, 0)], res.clone());
    let act_k = ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind, res.nnp_pool.clone());
    ProjResidualConv2dOperator{
      cfg:      cfg,
      split:    split,
      conv1:    conv1,
      conv2:    conv2,
      conv1x1:  conv1x1,
      join:     join,
      act_k:    act_k,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
    }
  }
}

impl DiffOperator<f32> for ProjResidualConv2dOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn diff_param_sz(&self) -> usize {
    self.conv1.diff_param_sz()
        + self.conv2.diff_param_sz()
        + self.conv1x1.diff_param_sz()
  }

  fn nondiff_param_sz(&self) -> usize {
    self.conv1.nondiff_param_sz()
        + self.conv2.nondiff_param_sz()
        + self.conv1x1.nondiff_param_sz()
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    self.conv1.init_param(rng);
    self.conv2.init_param(rng);
    self.conv1x1.init_param(rng);
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.load_param(param_reader, offset);
    offset += self.conv2.load_param(param_reader, offset);
    offset += self.conv1x1.load_param(param_reader, offset);
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.store_param(param_writer, offset);
    offset += self.conv2.store_param(param_writer, offset);
    offset += self.conv1x1.store_param(param_writer, offset);
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.update_param(alpha, beta, grad_reader, offset);
    offset += self.conv2.update_param(alpha, beta, grad_reader, offset);
    offset += self.conv1x1.update_param(alpha, beta, grad_reader, offset);
    offset - init_offset
  }

  fn update_nondiff_param(&mut self, iter: usize) {
    self.conv1.update_nondiff_param(iter);
    self.conv2.update_nondiff_param(iter);
    self.conv1x1.update_nondiff_param(iter);
  }

  fn reset_grad(&mut self) {
    self.conv1.reset_grad();
    self.conv2.reset_grad();
    self.conv1x1.reset_grad();
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    self.conv1.apply_grad_reg(reg);
    self.conv2.apply_grad_reg(reg);
    self.conv1x1.apply_grad_reg(reg);
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.store_grad(grad_writer, offset);
    offset += self.conv2.store_grad(grad_writer, offset);
    offset += self.conv1x1.store_grad(grad_writer, offset);
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.accumulate_grad(alpha, beta, grad_accum, offset);
    offset += self.conv2.accumulate_grad(alpha, beta, grad_accum, offset);
    offset += self.conv1x1.accumulate_grad(alpha, beta, grad_accum, offset);
    offset - init_offset
  }

  fn forward(&mut self, phase: OpPhase) {
    self.split.forward(phase);
    self.conv1.forward(phase);
    self.conv2.forward(phase);
    self.conv1x1.forward(phase);
    self.join.forward(phase);
    let join_out = self.join._output(0);
    let batch_size = *join_out.batch_size.borrow();
    self.act_k.forward(batch_size, &*join_out.out_buf.borrow(), &mut *self.out.out_buf.borrow_mut());
    *self.out.batch_size.borrow_mut() = batch_size;
  }

  fn backward(&mut self) {
    let batch_size = *self.out.batch_size.borrow();
    let join_out = self.join._output(0);
    self.act_k.backward(batch_size, &*join_out.out_buf.borrow(), &*self.out.out_grad.as_ref().unwrap().borrow(), &mut *join_out.out_grad.as_ref().unwrap().borrow_mut());
    self.join.backward();
    self.conv1x1.backward();
    self.conv2.backward();
    self.conv1.backward();
    self.split.backward();
  }
}
