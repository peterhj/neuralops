use prelude::*;
use common::{CommonResources};
use ops::*;

use operator::prelude::*;
use operator::data::{SampleExtractInput, SampleClass, SampleWeight};
use operator::rw::{ReadBuffer, WriteBuffer, ReadAccumulateBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{RwSlice};

//use rand::{Rng};
//use std::cell::{Ref};
//use std::ops::{Deref};

#[derive(Clone)]
pub enum SeqOperatorConfig {
  SimpleInput(SimpleInputOperatorConfig),
  Affine(AffineOperatorConfig),
  Conv2d(Conv2dOperatorConfig),
  BatchNormConv2d(BatchNormConv2dOperatorConfig),
  ResidualConv2d(ResidualConv2dOperatorConfig),
  ProjResidualConv2d(ProjResidualConv2dOperatorConfig),
  Pool2d(Pool2dOperatorConfig),
  SoftmaxNLLClassLoss(ClassLossConfig),
}

pub struct SeqOperator<T, S> {
  input_op:     Box<DiffOperatorInput<T, S, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>>,
  loss_op:      Box<DiffOperatorInput<T, S, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>>,
  inner_ops:    Vec<Box<DiffOperator<T, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>>>,
}

impl<S> SeqOperator<f32, S> where S: SampleExtractInput<f32> + SampleClass + SampleWeight {
  pub fn new(cfgs: Vec<SeqOperatorConfig>, cap: OpCapability) -> SeqOperator<f32, S> {
    let res = CommonResources::new();
    let num_ops = cfgs.len();
    let input_op = match &cfgs[0] {
      &SeqOperatorConfig::SimpleInput(ref cfg) => {
        let op = SimpleInputOperator::new(cfg.clone(), cap, res.clone());
        Box::new(op)
      }
      _ => unreachable!(),
    };
    let mut inner_ops: Vec<Box<DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>>> = vec![];
    for (idx, cfg) in cfgs[1 .. num_ops-1].iter().enumerate() {
      let op: Box<DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>> = {
        let prev_op = match idx {
          0 => &*input_op as &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>,
          _ => &*inner_ops[idx-1] as &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>,
        };
        match cfg {
          &SeqOperatorConfig::Affine(cfg) => {
            let op = AffineOperator::new(cfg, cap, prev_op, 0, res.clone());
            Box::new(op)
          }
          &SeqOperatorConfig::Conv2d(cfg) => {
            let op = Conv2dOperator::new(cfg, cap, prev_op, 0, res.clone());
            Box::new(op)
          }
          &SeqOperatorConfig::BatchNormConv2d(cfg) => {
            let op = BatchNormConv2dOperator::new(cfg, cap, prev_op, 0, res.clone());
            Box::new(op)
          }
          &SeqOperatorConfig::ResidualConv2d(cfg) => {
            let op = ResidualConv2dOperator::new(cfg, cap, prev_op, 0, res.clone());
            Box::new(op)
          }
          &SeqOperatorConfig::ProjResidualConv2d(cfg) => {
            let op = ProjResidualConv2dOperator::new(cfg, cap, prev_op, 0, res.clone());
            Box::new(op)
          }
          &SeqOperatorConfig::Pool2d(cfg) => {
            let op = Pool2dOperator::new(cfg, cap, prev_op, 0, res.clone());
            Box::new(op)
          }
          _ => unreachable!(),
        }
      };
      inner_ops.push(op);
      //inner_outs.push(out);
    }
    let loss_op = match cfgs[num_ops-1] {
      SeqOperatorConfig::SoftmaxNLLClassLoss(cfg) => {
        let prev_op = match inner_ops.len() {
          0 => &*input_op as &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>,
          _ => &*inner_ops[inner_ops.len()-1] as &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>,
        };
        let op = SoftmaxNLLClassLossOperator::new(cfg, cap, prev_op, 0, res.clone());
        Box::new(op)
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

impl<T, S> DiffOperatorInput<T, S> for SeqOperator<T, S> {
  fn load_data(&mut self, samples: &[S]) {
    self.input_op.load_data(samples);
    self.loss_op.load_data(samples);
  }
}

impl<T, S> DiffOperator<T> for SeqOperator<T, S> {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.loss_op._output(0)
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
    self.loss_op.init_param(rng);
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<T>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    for op in self.inner_ops.iter_mut() {
      offset += op.load_param(param_reader, offset);
    }
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<T>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    for op in self.inner_ops.iter_mut() {
      offset += op.store_param(param_writer, offset);
    }
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<T>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    for op in self.inner_ops.iter_mut() {
      offset += op.update_param(alpha, beta, grad_reader, offset);
    }
    offset - init_offset
  }

  fn update_nondiff_param(&mut self, iter: usize) {
    for op in self.inner_ops.iter_mut() {
      op.update_nondiff_param(iter);
    }
    self.loss_op.update_nondiff_param(iter);
  }

  fn reset_grad(&mut self) {
    for op in self.inner_ops.iter_mut() {
      op.reset_grad();
    }
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<T>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    for op in self.inner_ops.iter_mut() {
      offset += op.store_grad(grad_writer, offset);
    }
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<T>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    for op in self.inner_ops.iter_mut() {
      offset += op.accumulate_grad(alpha, beta, grad_accum, offset);
    }
    offset - init_offset
  }

  fn reset_loss(&mut self) {
    self.loss_op.reset_loss();
  }

  fn store_loss(&mut self) -> f32 {
    self.loss_op.store_loss()
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    // FIXME(20160921): regularization contributes extra loss from each of the
    // sub-operators; add those to the designated loss operator.
    for op in self.inner_ops.iter_mut() {
      op.apply_grad_reg(reg);
    }
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
    self.input_op.backward();
  }
}

impl<S> DiffOperatorOutput<f32, RwSlice<f32>> for SeqOperator<f32, S> {
  fn get_output(&mut self) -> RwSlice<f32> {
    //self.loss_out.out_buf.as_slice()
    self.loss_op._output(0).out_buf.as_slice()
  }
}

impl<S> DiffOperatorIo<f32, S, RwSlice<f32>> for SeqOperator<f32, S> {
}
