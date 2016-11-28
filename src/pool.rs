use prelude::*;
use kernels::ffi::*;

use densearray::prelude::*;
//use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array4d};
//use densearray::linalg::{Transpose};
//use nnpack::{NnpackHandle, NnpackPthreadPool};
//use nnpack::ffi::*;
use operator::prelude::*;
//use operator::rw::{ReadAccumulateBuffer, AccumulateBuffer};

use std::cell::{RefCell};
use std::cmp::{max};
//use std::ptr::{null_mut};
use std::rc::{Rc};

/*#[derive(Clone, Copy, Debug)]
pub enum PoolKind {
  Average,
  Max,
}*/

#[derive(Clone, Copy, Debug)]
pub struct Pool2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub pool_w:   usize,
  pub pool_h:   usize,
  pub stride_w: usize,
  pub stride_h: usize,
  pub pad_w:    usize,
  pub pad_h:    usize,
  pub kind:     PoolKind,
}

impl Pool2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, in_chan) = self.in_dim;
    let out_w = max(0, (in_w + 2 * self.pad_w - self.pool_w + self.stride_w) as isize) as usize / self.stride_w;
    let out_h = max(0, (in_h + 2 * self.pad_h - self.pool_h + self.stride_h) as isize) as usize / self.stride_h;
    (out_w, out_h, in_chan)
  }
}

pub struct NewPool2dOperator<S, IoBuf: ?Sized> {
  cfg:      Pool2dOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
}

impl<S, IoBuf: ?Sized> NewPool2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewPool2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(NewPool2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for NewPool2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for NewPool2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for NewPool2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for NewPool2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);
    match self.cfg.kind {
      PoolKind::Average => {
        if self.cfg.pad_w == 0 && self.cfg.pad_h == 0 {
          if self.cfg.pool_w == 2 && self.cfg.pool_h == 2 &&
              self.cfg.stride_w == 2 && self.cfg.stride_h == 2
          {
            unsafe { neuralops_avgpool2d_2x2_fwd(
                batch_size,
                self.cfg.in_dim.0,
                self.cfg.in_dim.1,
                self.cfg.in_dim.2,
                self.in_.buf.borrow().as_ptr(),
                self.out.buf.borrow_mut().as_mut_ptr(),
            ) };
          } else if self.cfg.pool_w == self.cfg.stride_w &&
              self.cfg.pool_h == self.cfg.stride_h
          {
            unsafe { neuralops_avgpool2d_fwd(
                batch_size,
                self.cfg.in_dim.0,
                self.cfg.in_dim.1,
                self.cfg.in_dim.2,
                self.in_.buf.borrow().as_ptr(),
                self.out.buf.borrow_mut().as_mut_ptr(),
                self.cfg.pool_w,
                self.cfg.pool_h,
            ) };
          } else {
            unimplemented!();
          }
        } else {
          unimplemented!();
        }
      }
      _ => unimplemented!(),
    }
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(in_grad) = self.in_.grad.as_ref() {
      match self.cfg.kind {
        PoolKind::Average => {
          if self.cfg.pad_w == 0 && self.cfg.pad_h == 0 {
            if self.cfg.pool_w == 2 && self.cfg.pool_h == 2 &&
                self.cfg.stride_w == 2 && self.cfg.stride_h == 2
            {
              unsafe { neuralops_avgpool2d_2x2_bwd(
                  batch_size,
                  self.cfg.in_dim.0,
                  self.cfg.in_dim.1,
                  self.cfg.in_dim.2,
                  self.in_.buf.borrow().as_ptr(),
                  self.out.grad.as_ref().unwrap().borrow().as_ptr(),
                  in_grad.borrow_mut().as_mut_ptr(),
              ) };
            } else if self.cfg.pool_w == self.cfg.stride_w &&
                self.cfg.pool_h == self.cfg.stride_h
            {
              unsafe { neuralops_avgpool2d_bwd(
                  batch_size,
                  self.cfg.in_dim.0,
                  self.cfg.in_dim.1,
                  self.cfg.in_dim.2,
                  self.in_.buf.borrow().as_ptr(),
                  self.out.grad.as_ref().unwrap().borrow().as_ptr(),
                  in_grad.borrow_mut().as_mut_ptr(),
                  self.cfg.pool_w,
                  self.cfg.pool_h,
              ) };
            } else {
              unimplemented!();
            }
          } else {
            unimplemented!();
          }
        }
        _ => unimplemented!(),
      }
    }
  }
}

pub struct ParallelPool2dOperator<S, IoBuf: ?Sized> {
  cfg:      Pool2dOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
}

impl<S, IoBuf: ?Sized> ParallelPool2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<ParallelPool2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(ParallelPool2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for ParallelPool2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for ParallelPool2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for ParallelPool2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for ParallelPool2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.batch_sz);
    self.out.batch_sz.set(batch_size);
    let (out_w, out_h, _) = self.cfg.out_dim();
    match self.cfg.kind {
      PoolKind::Average => {
        unsafe { neuralops_omp_caffe_avgpool2d_fwd(
            batch_size,
            self.cfg.in_dim.0,
            self.cfg.in_dim.1,
            self.cfg.in_dim.2,
            self.in_.buf.borrow().as_ptr(),
            out_w,
            out_h,
            self.out.buf.borrow_mut().as_mut_ptr(),
            self.cfg.pool_w,
            self.cfg.pool_h,
            self.cfg.stride_w,
            self.cfg.stride_h,
            self.cfg.pad_w,
            self.cfg.pad_h,
        ) };
      }
      _ => unimplemented!(),
    }
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    let (out_w, out_h, _) = self.cfg.out_dim();
    if let Some(in_grad) = self.in_.grad.as_ref() {
      match self.cfg.kind {
        PoolKind::Average => {
        unsafe { neuralops_omp_caffe_avgpool2d_bwd(
            batch_size,
            self.cfg.in_dim.0,
            self.cfg.in_dim.1,
            self.cfg.in_dim.2,
            out_w,
            out_h,
            self.out.grad.as_ref().unwrap().borrow().as_ptr(),
            in_grad.borrow_mut().as_mut_ptr(),
            self.cfg.pool_w,
            self.cfg.pool_h,
            self.cfg.stride_w,
            self.cfg.stride_h,
            self.cfg.pad_w,
            self.cfg.pad_h,
        ) };
        }
        _ => unimplemented!(),
      }
    }
  }
}
