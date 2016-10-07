extern crate neuralops;
extern crate operator;
extern crate rand;
extern crate rng;

use neuralops::prelude::*;
use neuralops::archs::*;
use neuralops::data::{CyclicDataIter, SubsampleDataIter};
use neuralops::data::cifar::{CifarFlavor, CifarDataShard};
//use neuralops::input::{InputPreproc};
use operator::prelude::*;
use operator::opt::adam::{AdamConfig, AdamWorker};
use operator::opt::sgd::{SgdConfig, SgdWorker};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};

fn main() {
  let batch_sz = 128;

  //let op_cfg = build_cifar10_simple_seq(batch_sz);
  let op_cfg = build_cifar10_simple2_seq(batch_sz);
  let op_cfg = build_cifar10_simple2b_seq(batch_sz);
  let op_cfg = build_cifar10_resnet_seq(batch_sz, 20);
  let op = SeqOperator::new(op_cfg, OpCapability::Backward);

  let mut train_data =
      SubsampleDataIter::new(
      batch_sz,
      CifarDataShard::new(
          CifarFlavor::Cifar10,
          PathBuf::from("datasets/cifar10/train.bin"),
      ));
  let mut valid_data =
      CyclicDataIter::new(
      CifarDataShard::new(
          CifarFlavor::Cifar10,
          PathBuf::from("datasets/cifar10/test.bin"),
      ));

  let sgd_cfg = SgdConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      StepSize::Constant(0.01),
    //step_size:      StepSize::Adaptive{init_step: 1.0, test_iters: 100, epoch_iters: 1600, sched: AdaptiveStepSizeSchedule::Pow10},
    //momentum:       None,
    momentum:       Some(0.9),
    l2_reg:         None,
    //l2_reg:         Some(1.0e-4),
  };
  //let mut sgd = SgdWorker::new(sgd_cfg, op);
  let sgd_cfg = AdamConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      StepSize::Constant(0.001),
    gamma1:         0.1,
    gamma2:         0.05,
    epsilon:        1.0e-12,
    l2_reg:         None,
    //l2_reg:         Some(1.0e-4),
  };
  let mut sgd = AdamWorker::new(sgd_cfg, op);

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  println!("DEBUG: training...");
  sgd.reset_opt_stats();
  sgd.init_param(&mut rng);
  for iter_nr in 0 .. 100000 {
    sgd.step(&mut train_data);
    if (iter_nr + 1) % 10 == 0 {
      println!("DEBUG: iter: {} stats: {:?}", iter_nr + 1, sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
    if (iter_nr + 1) % 100 == 0 {
      sgd.eval(valid_data.len(), &mut valid_data);
      println!("DEBUG: valid stats: {:?}", sgd.get_opt_stats());
      sgd.reset_opt_stats();
    }
  }
  /*println!("DEBUG: validation...");
  sgd.reset_opt_stats();
  sgd.eval(valid_data.len(), &mut valid_data);
  println!("DEBUG: valid stats: {:?}", sgd.get_opt_stats());*/
}
