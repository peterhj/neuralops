extern crate neuralops;
extern crate operator;
extern crate rand;
extern crate rng;

use neuralops::data::{CyclicSamplingDataIter, RandomSamplingDataIter};
use neuralops::data::mnist::{MnistDataShard};
use neuralops::prelude::*;
use operator::opt::{OptWorker};
use operator::opt::sgd::{SgdOptConfig, SgdOptWorker};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};

fn main() {
  let batch_sz = 32;
  let mut op_cfg = vec![];

  /*op_cfg.push(OperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    frame_sz:   784,
  }));
  /*op_cfg.push(OperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     784,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    //act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Xavier,
  }));*/
  op_cfg.push(OperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     784,
    out_dim:    50,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(OperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     50,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(OperatorConfig::SoftmaxNLLClassLoss(ClassLossOperatorConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    num_classes:    10,
  }));*/

  op_cfg.push(OperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    frame_sz:   784,
  }));
  op_cfg.push(OperatorConfig::Conv2d(Conv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (28, 28, 1),
    kernel_w:   5,
    kernel_h:   5,
    stride_w:   1,
    stride_h:   1,
    pad_left:   2,
    pad_right:  2,
    pad_bot:    2,
    pad_top:    2,
    out_chan:   10,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(OperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     7840,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
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
    l2_reg:         Some(1.0e-4),
  };
  let mut sgd = SgdOptWorker::new(sgd_cfg, op);

  let mut train_data =
      RandomSamplingDataIter::new(
      MnistDataShard::new(
          PathBuf::from("mnist/train-images-idx3-ubyte"),
          PathBuf::from("mnist/train-labels-idx1-ubyte"),
      ));
  let mut valid_data =
      CyclicSamplingDataIter::new(
      MnistDataShard::new(
          PathBuf::from("mnist/t10k-images-idx3-ubyte"),
          PathBuf::from("mnist/t10k-labels-idx1-ubyte"),
      ));

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  println!("DEBUG: training...");
  sgd.init_param(&mut rng);
  for iter_nr in 0 .. 200 {
    sgd.step(&mut train_data);
    if iter_nr % 1000 == 0 {
      println!("DEBUG: iter {}", iter_nr);
    }
  }
  println!("DEBUG: validation...");
  sgd.eval(10000, &mut valid_data);
}
