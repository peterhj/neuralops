use super::{OpCapability, OperatorConfig};
use common::{CommonOperatorOutput};
use data::{ClassSample2d};
use affine::{AffineOperator};
use conv::{Conv2dOperator};
use input::{SimpleInputOperator};
use loss::{SoftmaxNLLClassLossOperator};
//use prelude::*;

use operator::{Operator, InternalOperator, OpPhase, Regularization};
use operator::rw::{ReadAccumulateBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

//use rand::{Rng};

pub struct SeqOperator<T, S, Out> {
  input_op:     Box<Operator<T, S, Output=Out>>,
  loss_op:      Box<Operator<T, S, Output=Out>>,
  inner_ops:    Vec<Box<InternalOperator<T, Output=Out>>>,
}

impl SeqOperator<f32, ClassSample2d<u8>, CommonOperatorOutput<f32>> {
  pub fn new(cfgs: Vec<OperatorConfig>, cap: OpCapability) -> SeqOperator<f32, ClassSample2d<u8>, CommonOperatorOutput<f32>> {
    let num_ops = cfgs.len();
    let input_op = match cfgs[0] {
      OperatorConfig::SimpleInput(cfg) => {
        Box::new(SimpleInputOperator::new(cfg, cap))
      }
      _ => unreachable!(),
    };
    let mut inner_ops: Vec<Box<InternalOperator<f32, Output=CommonOperatorOutput<f32>>>> = vec![];
    for (idx, cfg) in cfgs[1 .. num_ops-1].iter().enumerate() {
      let op: Box<InternalOperator<f32, Output=CommonOperatorOutput<f32>>> = {
        let prev_op = match idx {
          0 => &*input_op as &InternalOperator<f32, Output=CommonOperatorOutput<f32>>,
          _ => &*inner_ops[idx-1] as &InternalOperator<f32, Output=CommonOperatorOutput<f32>>,
        };
        match cfg {
          &OperatorConfig::Affine(cfg) => {
            Box::new(AffineOperator::new(cfg, cap, prev_op, 0))
          }
          &OperatorConfig::Conv2d(cfg) => {
            Box::new(Conv2dOperator::new(cfg, cap, prev_op, 0))
          }
          _ => unreachable!(),
        }
      };
      inner_ops.push(op);
    }
    let loss_op = match cfgs[num_ops-1] {
      OperatorConfig::SoftmaxNLLClassLoss(cfg) => {
        let prev_op = match inner_ops.len() {
          0 => &*input_op as &InternalOperator<f32, Output=CommonOperatorOutput<f32>>,
          _ => &*inner_ops[inner_ops.len()-1] as &InternalOperator<f32, Output=CommonOperatorOutput<f32>>,
        };
        Box::new(SoftmaxNLLClassLossOperator::new(cfg, cap, prev_op, 0))
      }
      _ => unreachable!(),
    };
    SeqOperator{
      input_op:     input_op,
      loss_op:      loss_op,
      inner_ops:    inner_ops,
    }
  }
}

impl<T, S, Out> Operator<T, S> for SeqOperator<T, S, Out> where Out: Clone {
  fn load_data(&mut self, samples: &[S]) {
    self.input_op.load_data(samples);
    self.loss_op.load_data(samples);
  }
}

impl<T, S, Out> InternalOperator<T> for SeqOperator<T, S, Out> where Out: Clone {
  type Output = Out;

  fn output(&self, _arm: usize) -> Out {
    assert_eq!(0, _arm);
    self.loss_op.output(0)
  }

  fn param_len(&self) -> usize {
    let mut p = 0;
    for op in self.inner_ops.iter() {
      p += op.param_len();
    }
    p
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    for op in self.inner_ops.iter_mut() {
      op.init_param(rng);
    }
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<T>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    for op in self.inner_ops.iter_mut() {
      offset += op.update_param(alpha, beta, grad_reader, offset);
    }
    offset - init_offset
  }

  fn reset_grad(&mut self) {
    for op in self.inner_ops.iter_mut() {
      op.reset_grad();
    }
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    for op in self.inner_ops.iter_mut() {
      op.apply_grad_reg(reg);
    }
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<T>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    for op in self.inner_ops.iter_mut() {
      offset += op.accumulate_grad(alpha, beta, grad_accum, offset);
    }
    offset - init_offset
  }

  fn forward(&mut self, phase: OpPhase) {
    self.input_op.forward(phase);
    for op in self.inner_ops.iter_mut() {
      op.forward(phase);
    }
    self.loss_op.forward(phase);
  }

  fn backward(&mut self) {
    self.loss_op.backward();
    for op in self.inner_ops.iter_mut().rev() {
      op.backward();
    }
  }
}
