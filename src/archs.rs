use prelude::*;

use operator::prelude::*;

use std::cell::{RefCell};
use std::rc::{Rc};

const RESNET_AVG_RATE:  f32 = 0.05;
const RESNET_EPSILON:   f32 = 1.0e-6;

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
  let input = NewVarInputOperator::new(input_cfg, OpCapability::Forward);
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

pub fn build_cifar10_resnet20_parallel_loss(batch_sz: usize) -> Rc<RefCell<SoftmaxNLLClassLoss<SampleItem, [f32]>>> {
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
    bias:       false,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  };
  let input = NewVarInputOperator::new(input_cfg, OpCapability::Forward);
  let conv1 = ParallelBatchNormConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0);
  let res1_1 = ParallelResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, conv1, 0);
  let res1_2 = ParallelResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_1, 0);
  let res1_3 = ParallelResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_2, 0);
  let res2_1 = ParallelProjResidualConv2dOperator::new(proj_res2_cfg, OpCapability::Backward, res1_3, 0);
  let res2_2 = ParallelResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_1, 0);
  let res2_3 = ParallelResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_2, 0);
  let res3_1 = ParallelProjResidualConv2dOperator::new(proj_res3_cfg, OpCapability::Backward, res2_3, 0);
  let res3_2 = ParallelResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_1, 0);
  let res3_3 = ParallelResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_2, 0);
  let pool = ParallelPool2dOperator::new(pool_cfg, OpCapability::Backward, res3_3, 0);
  let affine = ParallelAffineOperator::new(affine_cfg, OpCapability::Backward, pool, 0);
  let loss = SoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, affine, 0);
  loss
}

pub fn build_cifar10_resnet20_mkl_loss(batch_sz: usize) -> Rc<RefCell<SoftmaxNLLClassLoss<SampleItem, [f32]>>> {
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
    bias:       false,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    10,
  };
  let input = NewVarInputOperator::new(input_cfg, OpCapability::Forward);
  let conv1 = MklBatchNormConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0);
  let res1_1 = MklResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, conv1, 0);
  let res1_2 = MklResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_1, 0);
  let res1_3 = MklResidualConv2dOperator::new(res1_cfg, OpCapability::Backward, res1_2, 0);
  let res2_1 = MklProjResidualConv2dOperator::new(proj_res2_cfg, OpCapability::Backward, res1_3, 0);
  let res2_2 = MklResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_1, 0);
  let res2_3 = MklResidualConv2dOperator::new(res2_cfg, OpCapability::Backward, res2_2, 0);
  let res3_1 = MklProjResidualConv2dOperator::new(proj_res3_cfg, OpCapability::Backward, res2_3, 0);
  let res3_2 = MklResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_1, 0);
  let res3_3 = MklResidualConv2dOperator::new(res3_cfg, OpCapability::Backward, res3_2, 0);
  let pool = ParallelPool2dOperator::new(pool_cfg, OpCapability::Backward, res3_3, 0);
  let affine = ParallelAffineOperator::new(affine_cfg, OpCapability::Backward, pool, 0);
  let loss = SoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, affine, 0);
  loss
}

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
  let input = NewVarInputOperator::new(input_cfg, OpCapability::Forward);
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

pub fn build_squeezenet_v1_1_mkl_loss(batch_sz: usize) -> Rc<RefCell<SoftmaxNLLClassLoss<SampleItem, [f32]>>> {
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
  unimplemented!();
  /*let input = NewVarInputOperator::new(input_cfg, OpCapability::Backward);
  let conv1 = MklConv2dOperator::new(conv1_cfg, OpCapability::Backward, input, 0);
  let pool1 = ParallelPool2dOperator::new(pool1_cfg, OpCapability::Backward, conv1, 0);
  let squeeze2 = MklSqueezeConv2dOperator::new(squeeze2_cfg, OpCapability::Backward, pool1, 0);
  let squeeze3 = MklSqueezeConv2dOperator::new(squeeze3_cfg, OpCapability::Backward, squeeze2, 0);
  let pool3 = ParallelPool2dOperator::new(pool3_cfg, OpCapability::Backward, squeeze3, 0);
  let squeeze4 = MklSqueezeConv2dOperator::new(squeeze4_cfg, OpCapability::Backward, pool3, 0);
  let squeeze5 = MklSqueezeConv2dOperator::new(squeeze5_cfg, OpCapability::Backward, squeeze4, 0);
  let pool5 = ParallelPool2dOperator::new(pool5_cfg, OpCapability::Backward, squeeze5, 0);
  let squeeze6 = MklSqueezeConv2dOperator::new(squeeze6_cfg, OpCapability::Backward, pool5, 0);
  let squeeze7 = MklSqueezeConv2dOperator::new(squeeze7_cfg, OpCapability::Backward, squeeze6, 0);
  let squeeze8 = MklSqueezeConv2dOperator::new(squeeze8_cfg, OpCapability::Backward, squeeze7, 0);
  let squeeze9 = MklSqueezeConv2dOperator::new(squeeze9_cfg, OpCapability::Backward, squeeze8, 0);
  let conv10 = MklConv2dOperator::new(conv10_cfg, OpCapability::Backward, squeeze9, 0);
  let pool10 = ParallelPool2dOperator::new(pool10_cfg, OpCapability::Backward, conv10, 0);
  let loss = SoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, pool10, 0);
  loss*/
}

pub fn build_fake_squeezenet_v1_1_mkl_loss(batch_sz: usize) -> Rc<RefCell<SoftmaxNLLClassLoss<SampleItem, [f32]>>> {
  /*let input_cfg = VarInputOperatorConfig{
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
  };*/
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
  let dummy_cfg = DummyOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (32, 32, 3),
    out_dim:    (224, 224, 3),
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
    pool_w:     2,  pool_h:     2,
    stride_w:   2,  stride_h:   2,
    pad_w:      0,  pad_h:      0,
    kind:       PoolKind::Average,
    //kind:       PoolKind::Max,
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
    pool_w:     2,  pool_h:     2,
    stride_w:   2,  stride_h:   2,
    pad_w:      0,  pad_h:      0,
    kind:       PoolKind::Average,
    //kind:       PoolKind::Max,
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
    pool_w:     2,  pool_h:     2,
    stride_w:   2,  stride_h:   2,
    pad_w:      0,  pad_h:      0,
    kind:       PoolKind::Average,
    //kind:       PoolKind::Max,
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
  let global_pool_cfg = Pool2dOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     (14, 14, 512),
    pool_w:     14, pool_h:     14,
    stride_w:   14, stride_h:   14,
    pad_w:      0,  pad_h:      0,
    kind:       PoolKind::Average,
  };
  let affine_cfg = AffineOperatorConfig{
    batch_sz:   batch_sz,
    in_dim:     512,
    out_dim:    1000,
    bias:       false,
    act_kind:   ActivationKind::Identity,
    w_init:     ParamInitKind::Kaiming,
  };
  let loss_cfg = ClassLossConfig{
    batch_sz:       batch_sz,
    num_classes:    1000,
  };
  let input = NewVarInputOperator::new(input_cfg, OpCapability::Forward);
  let dummy = DummyOperator::new(dummy_cfg, OpCapability::Forward, input, 0);
  let conv1 = MklConv2dOperator::new(conv1_cfg, OpCapability::Backward, dummy, 0);
  let pool1 = NewPool2dOperator::new(pool1_cfg, OpCapability::Backward, conv1, 0);
  let squeeze2 = MklSqueezeConv2dOperator::new(squeeze2_cfg, OpCapability::Backward, pool1, 0);
  let squeeze3 = MklSqueezeConv2dOperator::new(squeeze3_cfg, OpCapability::Backward, squeeze2, 0);
  let pool3 = NewPool2dOperator::new(pool3_cfg, OpCapability::Backward, squeeze3, 0);
  let squeeze4 = MklSqueezeConv2dOperator::new(squeeze4_cfg, OpCapability::Backward, pool3, 0);
  let squeeze5 = MklSqueezeConv2dOperator::new(squeeze5_cfg, OpCapability::Backward, squeeze4, 0);
  let pool5 = NewPool2dOperator::new(pool5_cfg, OpCapability::Backward, squeeze5, 0);
  let squeeze6 = MklSqueezeConv2dOperator::new(squeeze6_cfg, OpCapability::Backward, pool5, 0);
  let squeeze7 = MklSqueezeConv2dOperator::new(squeeze7_cfg, OpCapability::Backward, squeeze6, 0);
  let squeeze8 = MklSqueezeConv2dOperator::new(squeeze8_cfg, OpCapability::Backward, squeeze7, 0);
  let squeeze9 = MklSqueezeConv2dOperator::new(squeeze9_cfg, OpCapability::Backward, squeeze8, 0);
  let global_pool = NewPool2dOperator::new(global_pool_cfg, OpCapability::Backward, squeeze9, 0);
  let affine = ParallelAffineOperator::new(affine_cfg, OpCapability::Backward, global_pool, 0);
  let loss = SoftmaxNLLClassLoss::new(loss_cfg, OpCapability::Backward, affine, 0);
  loss
}
