use prelude::*;
use kernels::activate::{ActivateKernel};
//use kernels::batchnorm::{BatchNorm2dKernel};
//use kernels::conv::*;

use densearray::prelude::*;
/*use nnpack::{NnpackHandle, NnpackPthreadPool};
use nnpack::ffi::*;*/
use operator::prelude::*;
//use rng::xorshift::{Xorshiftplus128Rng};

//use rand::distributions::{IndependentSample};
//use rand::distributions::normal::{Normal};
//use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{max};
//use std::ptr::{null_mut};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct Conv1dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize),
  pub kernel:   usize,
  pub stride:   usize,
  pub dilation: usize,
  pub pad:      usize,
  pub out_chan: usize,
  pub bias:     bool,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl Conv1dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize) {
    // FIXME(20161106): dilation.
    let (in_u, _) = self.in_dim;
    let out_u = max(0, (in_u + 2 * self.pad - self.kernel + self.stride) as isize) as usize / self.stride;
    (out_u, self.out_chan)
  }
}

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
  pub bias:     bool,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl Conv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, _) = self.in_dim;
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
pub struct Conv3dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize, usize),
  pub kernel:   (usize, usize, usize),
  pub stride:   (usize, usize, usize),
  pub dilation: (usize, usize, usize),
  pub pad:      (usize, usize, usize),
  pub out_chan: usize,
  pub bias:     bool,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl Conv3dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize, usize) {
    // FIXME(20161106): dilation.
    let (in_u, in_v, in_w, _) = self.in_dim;
    let (kernel_u, kernel_v, kernel_w) = self.kernel;
    let (stride_u, stride_v, stride_w) = self.stride;
    let (pad_u, pad_v, pad_w) = self.pad;
    let out_u = max(0, (in_u + 2 * pad_u - kernel_u + stride_u) as isize) as usize / stride_u;
    let out_v = max(0, (in_v + 2 * pad_v - kernel_v + stride_v) as isize) as usize / stride_v;
    let out_w = max(0, (in_w + 2 * pad_w - kernel_w + stride_w) as isize) as usize / stride_w;
    (out_u, out_v, out_w, self.out_chan)
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
  pub epsilon:  f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl BatchNormConv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, _) = self.in_dim;
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
  pub epsilon:  f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjResidualConv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub stride_w: usize,
  pub stride_h: usize,
  pub out_chan: usize,
  pub avg_rate: f32,
  pub epsilon:  f32,
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

pub struct NewResidualConv2dOperator<S> {
  cfg:      ResidualConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<NewAddJoinOperator<S>>>,
  out:      CommonOutput,
  act_k:    ActivateKernel,
}

impl<S> NewResidualConv2dOperator<S> where S: 'static {
  pub fn new<InOp>(cfg: ResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewResidualConv2dOperator<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
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
      epsilon:  cfg.epsilon,
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
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.in_dim.flat_len(),
    };
    let split_op = NewCopySplitOperator::new(split_cfg, cap, prev_op, prev_arm);
    let conv1_op = NewBatchNormConv2dOperator::new(conv1_cfg, cap, split_op.clone(), 0);
    let conv2_op = NewBatchNormConv2dOperator::new(conv2_cfg, cap, conv1_op, 0);
    let join_op = NewAddJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv2_op, 0);
    join_op.borrow_mut().append_input(split_op, 1);
    Rc::new(RefCell::new(NewResidualConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
      out:      CommonOutput::new(cfg.batch_sz, cfg.in_dim.flat_len(), cap),
      act_k:    ActivateKernel::new(cfg.batch_sz, cfg.in_dim.flat_len(), cfg.act_kind),
    }))
  }
}

impl<S> Operator for NewResidualConv2dOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for NewResidualConv2dOperator<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for NewResidualConv2dOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
    let join_out = self.join_op.borrow()._output(0);
    let batch_size = join_out.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    self.act_k.forward(batch_size, &*join_out.buf.borrow(), &mut *self.out.buf.borrow_mut());
  }

  fn _backward(&mut self) {
    let join_out = self.join_op.borrow()._output(0);
    if let Some(ref join_grad) = join_out.grad.as_ref() {
      let batch_size = self.out.batch_sz.get();
      self.act_k.backward(batch_size, &*join_out.buf.borrow(), &*self.out.grad.as_ref().unwrap().borrow(), &mut *join_grad.borrow_mut());
    }
  }
}

pub struct NewProjResidualConv2dOperator<S> {
  cfg:      ProjResidualConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<NewAddJoinOperator<S>>>,
  out:      CommonOutput,
  act_k:    ActivateKernel,
}

impl<S> NewProjResidualConv2dOperator<S> where S: 'static {
  pub fn new<InOp>(cfg: ProjResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewProjResidualConv2dOperator<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
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
      epsilon:  cfg.epsilon,
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
      epsilon:  cfg.epsilon,
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
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.out_dim().flat_len(),
    };
    let split_op = NewCopySplitOperator::new(split_cfg, cap, prev_op, prev_arm);
    let conv1_op = NewBatchNormConv2dOperator::new(conv1_cfg, cap, split_op.clone(), 0);
    let conv2_op = NewBatchNormConv2dOperator::new(conv2_cfg, cap, conv1_op, 0);
    let conv1x1_op = NewBatchNormConv2dOperator::new(conv1x1_cfg, cap, split_op, 1);
    let join_op = NewAddJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv2_op, 0);
    join_op.borrow_mut().append_input(conv1x1_op, 0);
    Rc::new(RefCell::new(NewProjResidualConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      act_k:    ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S> Operator for NewProjResidualConv2dOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for NewProjResidualConv2dOperator<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for NewProjResidualConv2dOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
    let join_out = self.join_op.borrow()._output(0);
    let batch_size = join_out.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    self.act_k.forward(batch_size, &*join_out.buf.borrow(), &mut *self.out.buf.borrow_mut());
  }

  fn _backward(&mut self) {
    let join_out = self.join_op.borrow()._output(0);
    if let Some(ref join_grad) = join_out.grad.as_ref() {
      let batch_size = self.out.batch_sz.get();
      self.act_k.backward(batch_size, &*join_out.buf.borrow(), &*self.out.grad.as_ref().unwrap().borrow(), &mut *join_grad.borrow_mut());
    }
  }
}
