use prelude::*;
use input::{InputPreproc, VarInputPreproc};

use operator::prelude::*;

const RESNET_AVG_RATE:  f32 = 0.05;
const RESNET_EPSILON:   f32 = 1.0e-6;

pub fn build_cifar10_simple_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
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
    //avg_rate:   0.05,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     4096,
    out_dim:    64,
    act_kind:   ActivationKind::Rect,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::Affine(AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     64,
    out_dim:    10,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}

pub fn build_cifar10_simple2_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
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
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}

pub fn build_cifar10_simple2b_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
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
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}

pub fn build_cifar10_simple2res_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
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
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}

pub fn build_cifar10_krizh_seq(batch_sz: usize) -> Vec<SeqOperatorConfig> {
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
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  // FIXME(20161002): unimplemented.
  op_cfg
}

pub fn build_cifar10_resnet_seq(batch_sz: usize, num_layers: usize) -> Vec<SeqOperatorConfig> {
  let num_res = (num_layers - 2) / 6;
  assert_eq!(0, (num_layers - 2) % 6);
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
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}

pub fn build_cifar10_resnet_avgpool_seq(batch_sz: usize, num_layers: usize) -> Vec<SeqOperatorConfig> {
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
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Xavier,
  }));
  op_cfg.push(SeqOperatorConfig::SoftmaxNLLClassLoss(ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  }));
  op_cfg
}
