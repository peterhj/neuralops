extern crate neuralops;
extern crate operator;
extern crate rand;
extern crate rng;

use neuralops::prelude::*;
use neuralops::data::{CyclicDataIter, RandomSampleDataIter};
use neuralops::data::cifar::{CifarFlavor, CifarDataShard};
use neuralops::archs::*;
use operator::prelude::*;
use operator::opt::sgd::{SgdConfig, SgdUpdate};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};

fn main() {
  let batch_sz = 128;

  let mut train_data =
      RandomSampleDataIter::new(
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

  let loss = build_cifar10_resnet20_loss(batch_sz);

  let sgd_cfg = SgdConfig{
    step_size:  StepSize::Decay{init_step: 0.1, step_decay: 0.1, decay_iters: 50000},
    momentum:   Some(GradientMomentum::Nesterov(0.9)),
  };
  let mut checkpoint = CheckpointState::new(CheckpointConfig{
    prefix: PathBuf::from("logs/cifar10_resnet20_sgd"),
    trace:  true,
  });
  checkpoint.append_config_info(&sgd_cfg);
  let mut sgd: StochasticGradWorker<f32, SgdUpdate<_>, _, _, _> = StochasticGradWorker::new(batch_sz, batch_sz, sgd_cfg, loss);
  let mut stats = ClassLossStats::default();
  let mut display_stats = ClassLossStats::default();
  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());

  println!("DEBUG: training...");
  sgd.init(&mut rng);
  for iter_nr in 0 .. 150000 {
    checkpoint.start_timing();
    sgd.step(&mut train_data);
    checkpoint.stop_timing();
    sgd.update_stats(&mut stats);
    sgd.update_stats(&mut display_stats);
    checkpoint.append_class_stats_train(&stats);
    stats.reset();
    if (iter_nr + 1) % 1 == 0 {
      println!("DEBUG: iter: {} accuracy: {:.4} stats: {:?}", iter_nr + 1, display_stats.accuracy(), display_stats);
      display_stats.reset();
    }
    if (iter_nr + 1) % 500 == 0 {
      println!("DEBUG: validating...");
      checkpoint.start_timing();
      sgd.eval(valid_data.len(), &mut valid_data);
      checkpoint.stop_timing();
      sgd.update_stats(&mut stats);
      checkpoint.append_class_stats_valid(&stats);
      println!("DEBUG: valid: accuracy: {:.4} stats: {:?}", stats.accuracy(), stats);
      stats.reset();
    }
  }
}
