extern crate neuralops;
extern crate operator;
extern crate rand;
extern crate rng;

use neuralops::prelude::*;
use neuralops::archs::*;
use neuralops::data::{CyclicDataIter, RandomSampleDataIter, PartitionDataShard};
use neuralops::data::cifar::{CifarFlavor, CifarDataShard};
//use neuralops::input::{InputPreproc};
use operator::prelude::*;
use operator::opt::adam::{AdamConfig, AdamWorker};
use operator::opt::sgd::{SgdConfig, SgdWorker};
use operator::opt::shared_sgd::{SharedSyncSgdBuilder};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{thread_rng};
use std::path::{PathBuf};
use std::thread::{spawn};

fn main() {
  let batch_sz = 32;
  let num_workers = 4;

  let sgd_cfg = SgdConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      StepSize::Constant(0.1),
    momentum:       Some(0.9),
    l2_reg:         Some(1.0e-4),
  };
  //let mut sgd = SgdWorker::new(sgd_cfg, op);
  /*let sgd_cfg = AdamConfig{
    batch_sz:       batch_sz,
    minibatch_sz:   batch_sz,
    step_size:      StepSize::Constant(0.001),
    gamma1:         0.1,
    gamma2:         0.05,
    epsilon:        1.0e-12,
    l2_reg:         None,
    //l2_reg:         Some(1.0e-4),
  };
  let mut sgd = AdamWorker::new(sgd_cfg, op);*/

  let builder = SharedSyncSgdBuilder::new(sgd_cfg, num_workers);

  println!("DEBUG: training...");
  let mut handles = vec![];
  for rank in 0 .. num_workers {
    let builder = builder.clone();
    let handle = spawn(move || {
      let mut train_data =
          RandomSampleDataIter::new(
          CifarDataShard::new(
              CifarFlavor::Cifar10,
              PathBuf::from("datasets/cifar10/train.bin"),
          ));
      let mut valid_data =
          CyclicDataIter::new(
          PartitionDataShard::new(
              rank, num_workers,
          CifarDataShard::new(
              CifarFlavor::Cifar10,
              PathBuf::from("datasets/cifar10/test.bin"),
          )));

      //let op_cfg = build_cifar10_simple2b_seq(batch_sz);
      //let op_cfg = build_cifar10_simple2res_seq(batch_sz);
      let op_cfg = build_cifar10_resnet_seq(batch_sz, 20);
      let operator = SeqOperator::new(op_cfg, OpCapability::Backward);
      let mut sgd = builder.into_worker(rank, operator);
      let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());

      sgd.init_param(&mut rng);
      /*println!("DEBUG: validating...");
      sgd.reset_opt_stats();
      sgd.eval(valid_data.len(), &mut valid_data);
      println!("DEBUG: valid stats: {:?}", sgd.get_opt_stats());
      sgd.reset_opt_stats();*/
      for iter_nr in 0 .. 100000 {
        sgd.step(&mut train_data);
        if (iter_nr + 1) % 5 == 0 && rank == 0 {
          println!("DEBUG: iter: {} stats: {:?}", iter_nr + 1, sgd.get_opt_stats());
          sgd.reset_opt_stats();
        }
        if (iter_nr + 1) % 100 == 0 {
          println!("DEBUG: validating...");
          sgd.reset_opt_stats();
          sgd.eval(valid_data.len(), &mut valid_data);
          println!("DEBUG: valid stats: {:?}", sgd.get_opt_stats());
          sgd.reset_opt_stats();
        }
      }
    });
    handles.push(handle);
  }
  for handle in handles.drain(..) {
    handle.join().unwrap();
  }
}
