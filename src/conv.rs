use prelude::*;
use kernels::activate::*;

use densearray::prelude::*;
use operator::prelude::*;

use std::cell::{RefCell};
use std::cmp::{max};
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
pub struct Conv2d1x1OperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub out_chan: usize,
  pub bias:     bool,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl Conv2d1x1OperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, _) = self.in_dim;
    (in_w, in_h, self.out_chan)
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

pub struct NewResidualConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      ResidualConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<NewAddJoinOperator<S, IoBuf>>>,
  out:      CommonOutput,
  act_k:    ActivateKernel,
}

impl<S, IoBuf: ?Sized> NewResidualConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: ResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewResidualConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
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

impl<S, IoBuf: ?Sized> Operator for NewResidualConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for NewResidualConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for NewResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for NewResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for NewResidualConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
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

pub struct ParallelResidualConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      ResidualConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<NewAddJoinOperator<S, IoBuf>>>,
  out:      CommonOutput,
  act_k:    ParallelActivateKernel,
}

impl<S, IoBuf: ?Sized> ParallelResidualConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: ResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<ParallelResidualConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
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
    let conv1_op = ParallelBatchNormConv2dOperator::new(conv1_cfg, cap, split_op.clone(), 0);
    let conv2_op = ParallelBatchNormConv2dOperator::new(conv2_cfg, cap, conv1_op, 0);
    let join_op = NewAddJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv2_op, 0);
    join_op.borrow_mut().append_input(split_op, 1);
    Rc::new(RefCell::new(ParallelResidualConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
      out:      CommonOutput::new(cfg.batch_sz, cfg.in_dim.flat_len(), cap),
      act_k:    ParallelActivateKernel::new(cfg.batch_sz, cfg.in_dim.flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for ParallelResidualConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for ParallelResidualConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for ParallelResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for ParallelResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for ParallelResidualConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
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

pub struct NewProjResidualConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      ProjResidualConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<NewAddJoinOperator<S, IoBuf>>>,
  out:      CommonOutput,
  act_k:    ActivateKernel,
}

impl<S, IoBuf: ?Sized> NewProjResidualConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: ProjResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewProjResidualConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
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

impl<S, IoBuf: ?Sized> Operator for NewProjResidualConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for NewProjResidualConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for NewProjResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for NewProjResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for NewProjResidualConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
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

pub struct ParallelProjResidualConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      ProjResidualConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<NewAddJoinOperator<S, IoBuf>>>,
  out:      CommonOutput,
  act_k:    ParallelActivateKernel,
}

impl<S, IoBuf: ?Sized> ParallelProjResidualConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: ProjResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<ParallelProjResidualConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
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
    let conv1_op = ParallelBatchNormConv2dOperator::new(conv1_cfg, cap, split_op.clone(), 0);
    let conv2_op = ParallelBatchNormConv2dOperator::new(conv2_cfg, cap, conv1_op, 0);
    let conv1x1_op = ParallelBatchNormConv2dOperator::new(conv1x1_cfg, cap, split_op, 1);
    let join_op = NewAddJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv2_op, 0);
    join_op.borrow_mut().append_input(conv1x1_op, 0);
    Rc::new(RefCell::new(ParallelProjResidualConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      act_k:    ParallelActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for ParallelProjResidualConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for ParallelProjResidualConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for ParallelProjResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for ParallelProjResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for ParallelProjResidualConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
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

#[derive(Clone, Copy, Debug)]
pub struct SqueezeConv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub stride_w: usize,
  pub stride_h: usize,
  pub squeeze:  usize,
  pub out_chan: usize,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl SqueezeConv2dOperatorConfig {
  pub fn squeeze_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, _) = self.in_dim;
    let out_w = max(0, (in_w + self.stride_w - 1) as isize) as usize / self.stride_w;
    let out_h = max(0, (in_h + self.stride_h - 1) as isize) as usize / self.stride_h;
    (out_w, out_h, self.squeeze)
  }

  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, _) = self.in_dim;
    let out_w = max(0, (in_w + self.stride_w - 1) as isize) as usize / self.stride_w;
    let out_h = max(0, (in_h + self.stride_h - 1) as isize) as usize / self.stride_h;
    (out_w, out_h, self.out_chan)
  }
}

pub struct SqueezeConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      SqueezeConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<ConcatJoinOperator<S, IoBuf>>>,
}

impl<S, IoBuf: ?Sized> SqueezeConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: SqueezeConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<SqueezeConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let squeeze_dim = cfg.squeeze_dim();
    let expand_chan = cfg.out_chan / 2;
    assert_eq!(0, cfg.out_chan % 2);
    let conv1_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 1,            kernel_h: 1,
      stride_w: cfg.stride_w, stride_h: cfg.stride_h,
      pad_w:    0,            pad_h:    0,
      out_chan: cfg.squeeze,
      bias:     false,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      squeeze_dim.flat_len(),
    };
    let conv1x1_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      kernel_w: 1,  kernel_h: 1,
      stride_w: 1,  stride_h: 1,
      pad_w:    0,  pad_h:    0,
      out_chan: expand_chan,
      bias:     false,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv3x3_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      kernel_w: 3,  kernel_h: 3,
      stride_w: 1,  stride_h: 1,
      pad_w:    1,  pad_h:    1,
      out_chan: expand_chan,
      bias:     false,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let join_cfg = ConcatJoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      in_dims:  vec![conv1x1_cfg.out_dim().flat_len(), conv3x3_cfg.out_dim().flat_len()],
    };
    let conv1_op = NewConv2dOperator::new(conv1_cfg, cap, prev_op, prev_arm);
    let split_op = NewCopySplitOperator::new(split_cfg, cap, conv1_op, 0);
    let conv1x1_op = NewConv2dOperator::new(conv1x1_cfg, cap, split_op.clone(), 0);
    let conv3x3_op = NewConv2dOperator::new(conv3x3_cfg, cap, split_op.clone(), 1);
    let join_op = ConcatJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv1x1_op, 0);
    join_op.borrow_mut().append_input(conv3x3_op, 0);
    Rc::new(RefCell::new(SqueezeConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for SqueezeConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for SqueezeConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    let join_out = self.join_op.borrow()._output(0);
    join_out
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for SqueezeConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for SqueezeConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for SqueezeConv2dOperator<S, IoBuf> {
  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
  }

  fn _backward(&mut self) {
  }
}

pub struct ParallelSqueezeConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      SqueezeConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<ConcatJoinOperator<S, IoBuf>>>,
}

impl<S, IoBuf: ?Sized> ParallelSqueezeConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: SqueezeConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<ParallelSqueezeConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let squeeze_dim = cfg.squeeze_dim();
    let expand_chan = cfg.out_chan / 2;
    assert_eq!(0, cfg.out_chan % 2);
    let conv1_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 1,            kernel_h: 1,
      stride_w: cfg.stride_w, stride_h: cfg.stride_h,
      pad_w:    0,            pad_h:    0,
      out_chan: cfg.squeeze,
      bias:     false,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      squeeze_dim.flat_len(),
    };
    let conv1x1_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      kernel_w: 1,  kernel_h: 1,
      stride_w: 1,  stride_h: 1,
      pad_w:    0,  pad_h:    0,
      out_chan: expand_chan,
      bias:     false,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv3x3_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      kernel_w: 3,  kernel_h: 3,
      stride_w: 1,  stride_h: 1,
      pad_w:    1,  pad_h:    1,
      out_chan: expand_chan,
      bias:     false,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let join_cfg = ConcatJoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      in_dims:  vec![conv1x1_cfg.out_dim().flat_len(), conv3x3_cfg.out_dim().flat_len()],
    };
    let conv1_op = ParallelConv2dOperator::new(conv1_cfg, cap, prev_op, prev_arm);
    let split_op = NewCopySplitOperator::new(split_cfg, cap, conv1_op, 0);
    let conv1x1_op = ParallelConv2dOperator::new(conv1x1_cfg, cap, split_op.clone(), 0);
    let conv3x3_op = ParallelConv2dOperator::new(conv3x3_cfg, cap, split_op.clone(), 1);
    let join_op = ConcatJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv1x1_op, 0);
    join_op.borrow_mut().append_input(conv3x3_op, 0);
    Rc::new(RefCell::new(ParallelSqueezeConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for ParallelSqueezeConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for ParallelSqueezeConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    let join_out = self.join_op.borrow()._output(0);
    join_out
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for ParallelSqueezeConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for ParallelSqueezeConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for ParallelSqueezeConv2dOperator<S, IoBuf> {
  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
  }

  fn _backward(&mut self) {
  }
}
