use prelude::*;

use operator::prelude::*;

use std::cell::{RefCell};
use std::rc::{Rc};

const RESNET_AVG_RATE:  f32 = 0.05;
const RESNET_EPSILON:   f32 = 1.0e-6;

/*pub fn build_cifar10_simple_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
  let mut op_cfg = vec![];
  op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     32 * 32 * 3,
    preprocs:   vec![
      InputPreproc::ShiftScale{shift: None, scale: Some(1.0 / 255.0)},
    ],
  }));
  op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  //op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   16,
    bias:       true,
    //avg_rate:   0.01,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   4,
    bias:       true,
    //avg_rate:   0.05,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4096,
    out_dim:    64,
    bias:       true,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}*/

/*pub fn build_cifar10_simple2_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
  let mut op_cfg = vec![];
  op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     32 * 32 * 3,
    preprocs:   vec![
      InputPreproc::ShiftScale{shift: None, scale: Some(1.0 / 255.0)},
    ],
  }));
  op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  //op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   16,
    bias:       true,
    //avg_rate:   0.05,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    pool_w:     2,
    pool_h:     2,
    stride_w:   2,
    stride_h:   2,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  //op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 16),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   32,
    bias:       true,
    //avg_rate:   0.05,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    pool_w:     2,
    pool_h:     2,
    stride_w:   2,
    stride_h:   2,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  //op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 32),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   64,
    bias:       true,
    //avg_rate:   0.05,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    pool_w:     8,
    pool_h:     8,
    stride_w:   8,
    stride_h:   8,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}*/

/*pub fn build_cifar10_simple2b_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
  let mut op_cfg = vec![];
  /*op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     32 * 32 * 3,
    preprocs:   vec![
      InputPreproc::ShiftScale{shift: None, scale: Some(1.0 / 255.0)},
    ],
  }));*/
  op_cfg.push(SeqOperatorConfig::VarInput(VarInputOperatorConfig{
    batch_sz:   batch_sz,
    //stride:     32 * 32 * 3,
    max_stride: 32 * 32 * 3,
    out_dim:    (32, 32, 3),
    in_dtype:   Dtype::F32,
    preprocs:   vec![
      // XXX: the pixel mean is:
      // (1.25306915e2 1.2295039e2 1.1386535e2).
      VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 256.0},
      //VarInputPreproc::RandomResize2d{lo: 256, hi: 480, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomCrop2d{crop_w: 32, crop_h: 32, pad_w: 4, pad_h: 4, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomFlipX{phases: vec![OpPhase::Learning]},
    ],
  }));
  //op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   16,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    pool_w:     2,
    pool_h:     2,
    stride_w:   2,
    stride_h:   2,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  //op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 16),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   32,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    pool_w:     2,
    pool_h:     2,
    stride_w:   2,
    stride_h:   2,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  //op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 32),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   64,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    pool_w:     8,
    pool_h:     8,
    stride_w:   8,
    stride_h:   8,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}*/

/*pub fn build_cifar10_simple2res_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
  let mut op_cfg = vec![];
  /*op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     32 * 32 * 3,
    preprocs:   vec![
      InputPreproc::ShiftScale{shift: None, scale: Some(1.0 / 255.0)},
    ],
  }));*/
  op_cfg.push(SeqOperatorConfig::VarInput(VarInputOperatorConfig{
    batch_sz:   batch_sz,
    //stride:     32 * 32 * 3,
    max_stride: 32 * 32 * 3,
    out_dim:    (32, 32, 3),
    in_dtype:   Dtype::F32,
    preprocs:   vec![
      // XXX: the pixel mean is:
      // (1.25306915e2 1.2295039e2 1.1386535e2).
      VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 256.0},
      //VarInputPreproc::RandomResize2d{lo: 256, hi: 480, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomCrop2d{crop_w: 32, crop_h: 32, pad_w: 4, pad_h: 4, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomFlipX{phases: vec![OpPhase::Learning]},
    ],
  }));
  //op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
  //op_cfg.push(SeqOperatorConfig::ResidualConv2d(ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   16,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  /*op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    pool_w:     2,
    pool_h:     2,
    stride_w:   2,
    stride_h:   2,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));*/
  //op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  //op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
  op_cfg.push(SeqOperatorConfig::ProjResidualConv2d(ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    //in_dim:     (16, 16, 16),
    in_dim:     (32, 32, 16),
    stride_w:   2,
    stride_h:   2,
    out_chan:   32,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  /*op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    pool_w:     2,
    pool_h:     2,
    stride_w:   2,
    stride_h:   2,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));*/
  //op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  //op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
  op_cfg.push(SeqOperatorConfig::ProjResidualConv2d(ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    //in_dim:     (8, 8, 32),
    in_dim:     (16, 16, 32),
    stride_w:   2,
    stride_h:   2,
    out_chan:   64,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    pool_w:     8,
    pool_h:     8,
    stride_w:   8,
    stride_h:   8,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}*/

/*pub fn build_cifar10_krizh_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
  let mut op_cfg = vec![];
  op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     32 * 32 * 3,
    preprocs:   vec![
      InputPreproc::ShiftScale{shift: None, scale: Some(1.0 / 255.0)},
    ],
  }));
  op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   16,
    avg_rate:   0.05,
    epsilon:    1.0e-6,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  // FIXME(20161002): unimplemented.
  op_cfg
}*/

//pub fn build_cifar10_resnet20_loss<S>(batch_sz: usize) -> Rc<RefCell<SoftmaxNLLClassLoss<S>>> where S: 'static + SampleDatum<[f32]> + SampleLabel {
pub fn build_cifar10_resnet20_loss(batch_sz: usize) -> Rc<RefCell<SoftmaxNLLClassLoss<SampleItem, [f32]>>> {
  let input_cfg = VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 32 * 32 * 3,
    out_dim:    (32, 32, 3),
    in_dtype:   Dtype::F32,
    preprocs:   vec![
      // XXX: the pixel mean is:
      // (1.25306915e2 1.2295039e2 1.1386535e2).
      VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 256.0},
      VarInputPreproc::RandomCrop2d{crop_w: 32, crop_h: 32, pad_w: 4, pad_h: 4, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomFlipX{phases: vec![OpPhase::Learning]},
    ],
  };
  let conv1_cfg = BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   1,  stride_h:   1,
    pad_w:      1,  pad_h:      1,
    out_chan:   16,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res1_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res2_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    stride_w:   2,  stride_h:   2,
    out_chan:   32,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res2_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let proj_res3_cfg = ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    stride_w:   2,  stride_h:   2,
    out_chan:   64,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let res3_cfg = ResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let pool_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    pool_w:     8,  pool_h:     8,
    stride_w:   8,  stride_h:   8,
    pad_w:      0,  pad_h:      0,
    kind:       PoolKind::Average,
  };
  let affine_cfg = AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  };
  let input = NewVarInputOperator::new(input_cfg, OpCapability::Backward);
  let conv1 = NewBatchNormConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0);
  let res1_1 = NewResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, conv1, 0);
  let res1_2 = NewResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_1, 0);
  let res1_3 = NewResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_2, 0);
  let res2_1 = NewProjResidualConv2dOperator::new(proj_res2_cfg, OpCapability::Backward, res1_3, 0);
  let res2_2 = NewResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_1, 0);
  let res2_3 = NewResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_2, 0);
  let res3_1 = NewProjResidualConv2dOperator::new(proj_res3_cfg, OpCapability::Backward, res2_3, 0);
  let res3_2 = NewResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_1, 0);
  let res3_3 = NewResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_2, 0);
  let pool = NewPool2dOperator::new(pool_cfg, OpCapability::Backward, res3_3, 0);
  let affine = NewAffineOperator::new(affine_cfg, OpCapability::Backward, pool, 0);
  let loss = SoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, affine, 0);
  loss
}

/*pub fn build_cifar10_resnet_seq(batch_sz: usize, num_layers: usize) -> Vec<SeqOperatorConfig> {
  let num_res = (num_layers - 2) / 6;
  assert_eq!(0, (num_layers - 2) % 6);
  let mut op_cfg = vec![];
  op_cfg.push(SeqOperatorConfig::VarInput(VarInputOperatorConfig{
    batch_sz:   batch_sz,
    //stride:     32 * 32 * 3,
    max_stride: 32 * 32 * 3,
    out_dim:    (32, 32, 3),
    in_dtype:   Dtype::F32,
    preprocs:   vec![
      // XXX: the pixel mean is:
      // (1.25306915e2 1.2295039e2 1.1386535e2).
      VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 256.0},
      //VarInputPreproc::RandomResize2d{lo: 256, hi: 480, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomCrop2d{crop_w: 32, crop_h: 32, pad_w: 4, pad_h: 4, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomFlipX{phases: vec![OpPhase::Learning]},
    ],
  }));
  //op_cfg.push(SeqOperatorConfig::Conv2d(Conv2dOperatorConfig{
  op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   16,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  for _ in 0 .. num_res {
    op_cfg.push(SeqOperatorConfig::ResidualConv2d(ResidualConv2dOperatorConfig{
      batch_sz:   batch_sz,
      in_dim:     (32, 32, 16),
      avg_rate:   RESNET_AVG_RATE,
      epsilon:    RESNET_EPSILON,
      act_kind:   ActivationKind::Rect,
      w_init:     ParamInitKind::Kaiming,
    }));
  }
  op_cfg.push(SeqOperatorConfig::ProjResidualConv2d(ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    stride_w:   2,
    stride_h:   2,
    out_chan:   32,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  for _ in 1 .. num_res {
    op_cfg.push(SeqOperatorConfig::ResidualConv2d(ResidualConv2dOperatorConfig{
      batch_sz:   batch_sz,
      in_dim:     (16, 16, 32),
      avg_rate:   RESNET_AVG_RATE,
      epsilon:    RESNET_EPSILON,
      act_kind:   ActivationKind::Rect,
      w_init:     ParamInitKind::Kaiming,
    }));
  }
  op_cfg.push(SeqOperatorConfig::ProjResidualConv2d(ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    stride_w:   2,
    stride_h:   2,
    out_chan:   64,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  for _ in 1 .. num_res {
    op_cfg.push(SeqOperatorConfig::ResidualConv2d(ResidualConv2dOperatorConfig{
      batch_sz:   batch_sz,
      in_dim:     (8, 8, 64),
      avg_rate:   RESNET_AVG_RATE,
      epsilon:    RESNET_EPSILON,
      act_kind:   ActivationKind::Rect,
      w_init:     ParamInitKind::Kaiming,
    }));
  }
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    pool_w:     8,
    pool_h:     8,
    stride_w:   8,
    stride_h:   8,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    //in_dim:     8 * 8 * 64,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}*/

/*pub fn build_cifar10_resnet_avgpool_seq(batch_sz: usize, num_layers: usize) -> Vec<SeqOperatorConfig> {
  let num_res = (num_layers - 2) / 3;
  assert_eq!(0, (num_layers - 2) % 3);
  let mut op_cfg = vec![];
  op_cfg.push(SeqOperatorConfig::SimpleInput(SimpleInputOperatorConfig{
    batch_sz:   batch_sz,
    stride:     32 * 32 * 3,
    preprocs:   vec![
      // TODO(20161002): subtract the pixel mean (using `shift`).
      InputPreproc::ShiftScale{shift: None, scale: Some(1.0 / 255.0)},
    ],
  }));
  op_cfg.push(SeqOperatorConfig::BatchNormConv2d(BatchNormConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    kernel_w:   3,
    kernel_h:   3,
    stride_w:   1,
    stride_h:   1,
    pad_w:      1,
    pad_h:      1,
    out_chan:   16,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  for _ in 0 .. num_res {
    op_cfg.push(SeqOperatorConfig::ResidualConv2d(ResidualConv2dOperatorConfig{
      batch_sz:   batch_sz,
      in_dim:     (32, 32, 16),
      avg_rate:   RESNET_AVG_RATE,
      epsilon:    RESNET_EPSILON,
      act_kind:   ActivationKind::Rect,
      w_init:     ParamInitKind::Kaiming,
    }));
  }
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 16),
    pool_w:     2,
    pool_h:     2,
    stride_w:   2,
    stride_h:   2,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::ProjResidualConv2d(ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 16),
    stride_w:   1,
    stride_h:   1,
    out_chan:   32,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  for _ in 1 .. num_res {
    op_cfg.push(SeqOperatorConfig::ResidualConv2d(ResidualConv2dOperatorConfig{
      batch_sz:   batch_sz,
      in_dim:     (16, 16, 32),
      avg_rate:   RESNET_AVG_RATE,
      epsilon:    RESNET_EPSILON,
      act_kind:   ActivationKind::Rect,
      w_init:     ParamInitKind::Kaiming,
    }));
  }
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (16, 16, 32),
    pool_w:     2,
    pool_h:     2,
    stride_w:   2,
    stride_h:   2,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::ProjResidualConv2d(ProjResidualConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 32),
    stride_w:   1,
    stride_h:   1,
    out_chan:   64,
    avg_rate:   RESNET_AVG_RATE,
    epsilon:    RESNET_EPSILON,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  for _ in 1 .. num_res {
    op_cfg.push(SeqOperatorConfig::ResidualConv2d(ResidualConv2dOperatorConfig{
      batch_sz:   batch_sz,
      in_dim:     (8, 8, 64),
      avg_rate:   RESNET_AVG_RATE,
      epsilon:    RESNET_EPSILON,
      act_kind:   ActivationKind::Rect,
      w_init:     ParamInitKind::Kaiming,
    }));
  }
  op_cfg.push(SeqOperatorConfig::Pool2d(Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (8, 8, 64),
    pool_w:     8,
    pool_h:     8,
    stride_w:   8,
    stride_h:   8,
    pad_w:      0,
    pad_h:      0,
    kind:       PoolKind::Average,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    bias:       true,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}*/

pub fn build_squeezenet_v1_1_loss(batch_sz: usize) -> Rc<RefCell<SoftmaxNLLClassLoss<SampleItem, [f32]>>> {
  let input_cfg = VarInputOperatorConfig{
    batch_sz:   batch_sz,
    max_stride: 16 * 480 * 480 * 3,
    out_dim:    (224, 224, 3),
    in_dtype:   Dtype::F32,
    preprocs:   vec![
      VarInputPreproc::RandomResize2d{hi: 480, lo: 256, phases: vec![OpPhase::Learning]},
      VarInputPreproc::RandomResize2d{hi: 256, lo: 256, phases: vec![OpPhase::Inference]},
      VarInputPreproc::RandomCrop2d{crop_w: 224, crop_h: 224, pad_w: 0, pad_h: 0, phases: vec![OpPhase::Learning]},
      VarInputPreproc::CenterCrop2d{crop_w: 224, crop_h: 224, phases: vec![OpPhase::Inference]},
      VarInputPreproc::RandomFlipX{phases: vec![OpPhase::Learning]},
      //VarInputPreproc::ChannelShift{shift: vec![125.0, 123.0, 114.0]},
      VarInputPreproc::Scale{scale: 1.0 / 255.0},
      //VarInputPreproc::AddPixelwisePCALigtingNoise{},
    ],
  };
  let conv1_cfg = Conv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (224, 224, 3),
    kernel_w:   3,  kernel_h:   3,
    stride_w:   2,  stride_h:   2,
    pad_w:      1,  pad_h:      1,
    out_chan:   64,
    bias:       true,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let pool1_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (112, 112, 64),
    pool_w:     3,  pool_h:     3,
    stride_w:   2,  stride_h:   2,
    pad_w:      1,  pad_h:      1,
    kind:       PoolKind::Max,
  };
  let squeeze2_cfg = SqueezeConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (56, 56, 64),
    squeeze:    16,
    out_chan:   128,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let squeeze3_cfg = SqueezeConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (56, 56, 128),
    squeeze:    16,
    out_chan:   128,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let pool3_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (56, 56, 128),
    pool_w:     3,  pool_h:     3,
    stride_w:   2,  stride_h:   2,
    pad_w:      1,  pad_h:      1,
    kind:       PoolKind::Max,
  };
  let squeeze4_cfg = SqueezeConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (28, 28, 128),
    squeeze:    32,
    out_chan:   256,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let squeeze5_cfg = SqueezeConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (28, 28, 256),
    squeeze:    32,
    out_chan:   256,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let pool5_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (28, 28, 256),
    pool_w:     3,  pool_h:     3,
    stride_w:   2,  stride_h:   2,
    pad_w:      1,  pad_h:      1,
    kind:       PoolKind::Max,
  };
  let squeeze6_cfg = SqueezeConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 256),
    squeeze:    48,
    out_chan:   384,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let squeeze7_cfg = SqueezeConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 384),
    squeeze:    48,
    out_chan:   384,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let squeeze8_cfg = SqueezeConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 384),
    squeeze:    64,
    out_chan:   512,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let squeeze9_cfg = SqueezeConv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 512),
    squeeze:    64,
    out_chan:   512,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let conv10_cfg = Conv2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 512),
    kernel_w:   1,  kernel_h:   1,
    stride_w:   1,  stride_h:   1,
    pad_w:      0,  pad_h:      0,
    out_chan:   1000,
    bias:       true,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  };
  let pool10_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 1000),
    pool_w:     14, pool_h:     14,
    stride_w:   14, stride_h:   14,
    pad_w:      0,  pad_h:      0,
    kind:       PoolKind::Average,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    1000,
  };
  let input = NewVarInputOperator::new(input_cfg, OpCapability::Backward);
  let conv1 = ParallelConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0);
  let pool1 = ParallelPool2dOperator::new(pool1_cfg, OpCapability::Backward, conv1, 0);
  let squeeze2 = ParallelSqueezeConv2dOperator::new(squeeze2_cfg, OpCapability::Backward, pool1, 0);
  let squeeze3 = ParallelSqueezeConv2dOperator::new(squeeze3_cfg, OpCapability::Backward, squeeze2, 0);
  let pool3 = ParallelPool2dOperator::new(pool3_cfg, OpCapability::Backward, squeeze3, 0);
  let squeeze4 = ParallelSqueezeConv2dOperator::new(squeeze4_cfg, OpCapability::Backward, pool3, 0);
  let squeeze5 = ParallelSqueezeConv2dOperator::new(squeeze5_cfg, OpCapability::Backward, squeeze4, 0);
  let pool5 = ParallelPool2dOperator::new(pool5_cfg, OpCapability::Backward, squeeze5, 0);
  let squeeze6 = ParallelSqueezeConv2dOperator::new(squeeze6_cfg, OpCapability::Backward, pool5, 0);
  let squeeze7 = ParallelSqueezeConv2dOperator::new(squeeze7_cfg, OpCapability::Backward, squeeze6, 0);
  let squeeze8 = ParallelSqueezeConv2dOperator::new(squeeze8_cfg, OpCapability::Backward, squeeze7, 0);
  let squeeze9 = ParallelSqueezeConv2dOperator::new(squeeze9_cfg, OpCapability::Backward, squeeze8, 0);
  let conv10 = ParallelConv2dOperator::new(conv10_cfg, OpCapability::Backward, squeeze9, 0);
  let pool10 = ParallelPool2dOperator::new(pool10_cfg, OpCapability::Backward, conv10, 0);
  let loss = SoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, pool10, 0);
  loss
}
