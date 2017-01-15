use prelude::*;
use kernels::ffi::*;

use densearray::prelude::*;
use operator::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
//use rand::distributions::{IndependentSample};
//use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{max};
use std::rc::{Rc};

#[derive(Clone, Debug)]
pub struct DropoutOperatorConfig {
  pub batch_sz:     usize,
  pub dim:          usize,
  pub drop_frac:    f32,
}

pub struct DropoutOperator<S, IoBuf: ?Sized> {
  cfg:      DropoutOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  mask:     Vec<f32>,
  rng:      Xorshiftplus128Rng,
  //dist:     Range<f32>,
}

impl<S, IoBuf: ?Sized> DropoutOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: DropoutOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<DropoutOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let in_ = prev_op.borrow()._output(prev_arm);
    let out = CommonOutput::new(cfg.batch_sz, cfg.dim, cap);
    let mut mask = Vec::with_capacity(cfg.batch_sz * cfg.dim);
    mask.resize(cfg.batch_sz * cfg.dim, 0.0);
    Rc::new(RefCell::new(DropoutOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      out,
      mask:     mask,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      //dist:     Range::new(0.0, 1.0),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for DropoutOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for DropoutOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorData<S> for DropoutOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for DropoutOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for DropoutOperator<S, IoBuf> {
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

  fn _forward(&mut self, phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    assert!(batch_size <= self.cfg.batch_sz);
    self.out.batch_sz.set(batch_size);
    match phase {
      OpPhase::Inference => {
        self.out.buf.borrow_mut()[ .. batch_size * self.cfg.dim]
          .copy_from_slice(&self.in_.buf.borrow()[ .. batch_size * self.cfg.dim]);
      }
      OpPhase::Learning => {
        let in_buf = &self.in_.buf.borrow()[ .. batch_size * self.cfg.dim];
        let out_buf = &mut self.out.buf.borrow_mut()[ .. batch_size * self.cfg.dim];
        for p in 0 .. batch_size * self.cfg.dim {
          let u: f32 = self.rng.gen();
          if u < self.cfg.drop_frac {
            self.mask[p] = 0.0;
            out_buf[p] = 0.0;
          } else {
            self.mask[p] = 1.0;
            out_buf[p] = in_buf[p];
          }
        }
      }
    }
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();
    if let Some(in_grad) = self.in_.grad.as_ref() {
      let in_grad = &mut in_grad.borrow_mut()[ .. batch_size * self.cfg.dim];
      let out_grad = &self.out.grad.as_ref().unwrap().borrow()[ .. batch_size * self.cfg.dim];
      for p in 0 .. batch_size * self.cfg.dim {
        in_grad[p] = self.mask[p] * out_grad[p];
      }
    }
  }
}
