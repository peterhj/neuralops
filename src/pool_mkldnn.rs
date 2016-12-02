use prelude::*;
use kernels::ffi::*;

use densearray::prelude::*;
use mkl_dnn::prelude::*;
use operator::prelude::*;

use std::cell::{RefCell};
use std::cmp::{max};
//use std::ptr::{null_mut};
use std::rc::{Rc};

pub struct MklPool2dOperator<S, IoBuf: ?Sized> {
  cfg:      Pool2dOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  fwd:      MklDnnPool2dFwd<f32>,
  bwd:      MklDnnPool2dBwd<f32>,
}

impl<S, IoBuf: ?Sized> MklPool2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: Pool2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<MklPool2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let out_dim = cfg.out_dim();
    let pool_cfg = MklDnnPool2dConfig{
      in_dim:   vec![cfg.in_dim.0, cfg.in_dim.1, cfg.in_dim.2, cfg.batch_sz],
      out_dim:  vec![out_dim.0, out_dim.1, out_dim.2, cfg.batch_sz],
      pool_dim: vec![cfg.pool_w, cfg.pool_h],
      stride:   vec![cfg.stride_w, cfg.stride_h],
      pad:      vec![cfg.pad_w, cfg.pad_h],
      algo:     match cfg.kind {
        PoolKind::Average => MklDnnPoolAlgo::Average,
        PoolKind::Max     => MklDnnPoolAlgo::Max,
      },
    };
    Rc::new(RefCell::new(MklPool2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      fwd:      MklDnnPool2dFwd::create(pool_cfg.clone()).unwrap(),
      bwd:      MklDnnPool2dBwd::create(pool_cfg.clone()).unwrap(),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for MklPool2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for MklPool2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for MklPool2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for MklPool2dOperator<S, IoBuf> {
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
    unsafe { self.fwd.execute(
        self.in_.buf.borrow().as_ptr(),
        self.out.buf.borrow_mut().as_mut_ptr(),
    ).unwrap() };
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    //let (out_w, out_h, _) = self.cfg.out_dim();
    if let Some(in_grad) = self.in_.grad.as_ref() {
      let in_len = self.cfg.in_dim.flat_len();
      in_grad.borrow_mut().reshape_mut(batch_size * in_len).set_constant(0.0);
      let workspace = self.fwd._workspace();
      unsafe { self.bwd.execute(
          self.out.grad.as_ref().unwrap().borrow().as_ptr(),
          in_grad.borrow_mut().as_mut_ptr(),
          workspace,
      ).unwrap() };
    }
  }
}
