use prelude::*;
use common::{CommonResources, CommonOperatorOutput};
use kernels::ffi::*;

use densearray::{ArrayIndex};
//use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array4d};
//use densearray::linalg::{Transpose};
//use nnpack::{NnpackHandle, NnpackPthreadPool};
//use nnpack::ffi::*;
use operator::prelude::*;
//use operator::rw::{ReadAccumulateBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use std::cell::{RefCell};
//use std::cmp::{max, min};
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
    // FIXME(20161002)
    //unimplemented!();
    (self.in_dim.0 / self.stride_w, self.in_dim.1 / self.stride_h, self.in_dim.2)
  }
}

pub struct Pool2dOperator {
  cfg:      Pool2dOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  out:      CommonOperatorOutput<f32>,
  //nnp_h:    NnpackHandle,
  //nnp_pool: Rc<NnpackPthreadPool>,
}

impl Pool2dOperator {
  pub fn new(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> Pool2dOperator {
    Pool2dOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
    }
  }
}

impl DiffOperator<f32> for Pool2dOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn forward(&mut self, _phase: OpPhase) {
    let batch_size = *self.in_.batch_size.borrow();
    assert!(batch_size <= self.cfg.batch_sz);
    *self.out.batch_size.borrow_mut() = batch_size;
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
                self.in_.out_buf.borrow().as_ptr(),
                self.out.out_buf.borrow_mut().as_mut_ptr(),
            ) };
          } else if self.cfg.pool_w == self.cfg.stride_w &&
              self.cfg.pool_h == self.cfg.stride_h
          {
            unsafe { neuralops_avgpool2d_fwd(
                batch_size,
                self.cfg.in_dim.0,
                self.cfg.in_dim.1,
                self.cfg.in_dim.2,
                self.in_.out_buf.borrow().as_ptr(),
                self.out.out_buf.borrow_mut().as_mut_ptr(),
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

  fn backward(&mut self) {
    let batch_size = *self.out.batch_size.borrow();
    if let Some(in_grad) = self.in_.out_grad.as_ref() {
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
                  self.in_.out_buf.borrow().as_ptr(),
                  self.out.out_grad.as_ref().unwrap().borrow().as_ptr(),
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
                  self.in_.out_buf.borrow().as_ptr(),
                  self.out.out_grad.as_ref().unwrap().borrow().as_ptr(),
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

pub struct NewPool2dOperator<S> {
  cfg:      Pool2dOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<NewDiffOperator<S, IoBuf=[f32]>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
}

impl<S> NewPool2dOperator<S> {
  pub fn new<InOp>(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewPool2dOperator<S>>> where InOp: 'static + CommonOperator + NewDiffOperator<S, IoBuf=[f32]> {
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

impl<S> Operator for NewPool2dOperator<S> {
  fn _next(&self) -> u64 {
    self.node._next()
  }

  fn _epoch(&self) -> u64 {
    self.node._epoch()
  }
}

impl<S> CommonOperator for NewPool2dOperator<S> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S> NewDiffOperator<S> for NewPool2dOperator<S> {
  type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>)) {
    self.node.step(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
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
