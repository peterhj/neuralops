extern crate neuralops;
extern crate operator;
extern crate rand;
extern crate rng;

use neuralops::data::{RandomSamplingDataIter};
use neuralops::data::mnist::{MnistDataShard};
use neuralops::prelude::*;
use operator::opt::{OptWorker};
use operator::opt::sgd::{SgdOptConfig, SgdOptWorker};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
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
    w_init:     ParamInitKind::Normal{mean: 0.0, std: 0.01},
  }));
  op_cfg.push(OperatorConfig::SoftmaxNLLClassLoss(ClassLossOperatorConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    num_classes:    10,
  }));
  let op = SeqOperator::new(op_cfg, OpCapability::Backward);

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

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  sgd.init_param(&mut rng);
  for iter_nr in 0 .. 100 {
    sgd.step(&mut data);
    if iter_nr % 1000 == 0 {
      println!("DEBUG: iter {}", iter_nr);
    }
  }
}
