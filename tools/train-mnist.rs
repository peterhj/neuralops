extern crate operator;
extern crate neuralops;

use operator::opt::{OptWorker};
use operator::opt::sgd::{SgdOptConfig, SgdOptWorker};
use neuralops::data::{RandomSamplingDataIter};
use neuralops::data::mnist::{MnistDataShard};
use neuralops::prelude::*;

use std::path::{PathBuf};

fn main() {
  let batch_sz = 50;
  let mut op_cfg = vec![];
  op_cfg.push(OperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    frame_sz:   784,
  }));
  op_cfg.push(OperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     784,
    out_dim:    10,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Normal{mean: 0.0, std: 0.05},
  }));
  op_cfg.push(OperatorConfig::SoftmaxNLLClassLoss(ClassLossOperatorConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    num_classes:    10,
  }));
  let op = SeqOperator::new(op_cfg);

  let sgd_cfg = SgdOptConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      0.01,
    momentum:       None,
  };
  let mut sgd = SgdOptWorker::new(sgd_cfg, op);

  let mut data =
      RandomSamplingDataIter::new(
      MnistDataShard::new(
          PathBuf::from("mnist/train-images-idx3-ubyte"),
          PathBuf::from("mnist/train-labels-idx1-ubyte"),
      ));

  for _ in 0 .. 10000 {
    sgd.step(&mut data);
  }
}
