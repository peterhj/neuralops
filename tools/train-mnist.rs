extern crate neuralops;
extern crate operator;
extern crate rand;
extern crate rng;

use neuralops::prelude::*;
use neuralops::data::{CyclicDataIter, SubsampleDataIter};
use neuralops::data::mnist::{MnistDataShard};
use neuralops::input::{InputPreproc};
use operator::prelude::*;
use operator::opt::{StepSize};
use operator::opt::adam::{AdamConfig, AdamWorker};
use operator::opt::sgd::{SgdConfig, SgdWorker};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};

fn main() {
  let batch_sz = 32;

  let mut op_cfg = vec![];
  op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     784,
    preprocs:   vec![
      InputPreproc::ShiftScale{shift: None, scale: Some(1.0 / 255.0)},
    ],
  }));
  /*op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     784,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    //act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Xavier,
  }));*/
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     784,
    out_dim:    50,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     50,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));

  /*op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    frame_sz:   784,
  }));
  op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
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
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     7840,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossOperatorConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    num_classes:    10,
  }));*/

  let op = SeqOperator::new(op_cfg, OpCapability::Backward);

  let mut train_data =
      SubsampleDataIter::new(
      batch_sz,
      MnistDataShard::new(
          PathBuf::from("datasets/mnist/train-images-idx3-ubyte"),
          PathBuf::from("datasets/mnist/train-labels-idx1-ubyte"),
      ));
  let mut valid_data =
      CyclicDataIter::new(
      MnistDataShard::new(
          PathBuf::from("datasets/mnist/t10k-images-idx3-ubyte"),
          PathBuf::from("datasets/mnist/t10k-labels-idx1-ubyte"),
      ));

  let sgd_cfg = SgdConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      StepSize::Constant(0.1),
    momentum:       Some(0.9),
    l2_reg:         None,
    //l2_reg:         Some(1.0e-4),
  };
  let mut sgd = SgdWorker::new(sgd_cfg, op);
  /*let sgd_cfg = AdamConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      StepSize::Constant(0.01),
    gamma1:         0.1,
    gamma2:         0.05,
    epsilon:        1.0e-12,
    l2_reg:         None,
    //l2_reg:         Some(1.0e-4),
  };
  let mut sgd = AdamWorker::new(sgd_cfg, op);*/

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  println!("DEBUG: training...");
  sgd.reset_opt_stats();
  sgd.init_param(&mut rng);
  for iter_nr in 0 .. 1000 {
    sgd.step(&mut train_data);
    if (iter_nr + 1) % 20 == 0 {
      println!("DEBUG: iter: {} stats: {:?}", iter_nr + 1, sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
  }
  println!("DEBUG: validation...");
  sgd.reset_opt_stats();
  sgd.eval(valid_data.len(), &mut valid_data);
  println!("DEBUG: valid stats: {:?}", sgd.get_opt_stats());
}
