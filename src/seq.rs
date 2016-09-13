use super::{OpConfig};
use common::{CommonOperatorOutput};
use data::{ClassSample2d};

use operator::{Operator, InternalOperator, OpPhase};

pub struct SeqOperator<T, S, Out> {
  input_op:     Box<Operator<T, S, Output=Out>>,
  loss_op:      Box<Operator<T, S, Output=Out>>,
  inner_ops:    Vec<Box<InternalOperator<T, Output=Out>>>,
}

impl SeqOperator<f32, ClassSample2d<u8>, CommonOperatorOutput<f32>> {
  pub fn new(cfgs: Vec<OpConfig>) -> SeqOperator<f32, ClassSample2d<u8>, CommonOperatorOutput<f32>> {
    unimplemented!();
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
