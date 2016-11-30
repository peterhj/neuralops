//pub use super::{OperatorConfig};
pub use affine::{
  AffineOperatorConfig,
  BatchNormAffineOperatorConfig,
  NewAffineOperator,
  ParallelAffineOperator,
};
pub use class_loss::{
  BinaryClassLossConfig,
  ClassLossConfig,
  SoftmaxNLLClassLoss,
  EntRegSoftmaxNLLClassLossConfig,
  EntRegSoftmaxNLLClassLoss,
  LogisticNLLClassLoss,
};
pub use common::{
  ActivationKind, ParamInitKind, PoolKind,
  CommonOperator, CommonOutput,
};
pub use conv::{
  Conv2dOperatorConfig,
  BatchNormConv2dOperatorConfig,
  ResidualConv2dOperatorConfig,
  ProjResidualConv2dOperatorConfig,
  SqueezeConv2dOperatorConfig,
  NewResidualConv2dOperator,
  ParallelResidualConv2dOperator,
  NewProjResidualConv2dOperator,
  ParallelProjResidualConv2dOperator,
  SqueezeConv2dOperator,
  ParallelSqueezeConv2dOperator,
};
pub use conv_gemm::{
  ParallelConv2dOperator,
  ParallelBatchNormConv2dOperator,
};
#[cfg(feature = "mkldnn")]
pub use conv_mkldnn::{
  MklConv2dOperator,
  MklBatchNormConv2dOperator,
  MklResidualConv2dOperator,
  MklProjResidualConv2dOperator,
  MklSqueezeConv2dOperator,
};
//#[cfg(not(feature = "mkl"))]
pub use conv_nnpack::{
  NewConv2dOperator,
  NewBatchNormConv2dOperator,
};
pub use data::{Dtype};
pub use deconv::{
  ConvTranspose2dOperatorConfig,
  BatchNormConvTranspose2dOperatorConfig,
};
pub use dummy::{
  DummyOperatorConfig,
  DummyOperator,
};
//pub use graph::{GraphOperator, GraphOperatorConfig};
pub use input::{
  //SimpleInputOperatorConfig,
  VarInputOperatorConfig,
  VarInputPreproc,
  NewVarInputOperator,
  ParallelVarInputOperator,
};
pub use join::{
  JoinOperatorConfig,
  ConcatJoinOperatorConfig,
  NewAddJoinOperator,
  ConcatJoinOperator,
};
pub use loss::{
  L2RegOperator,
};
pub use param::{
  ParamBlock,
};
pub use pool::{
  Pool2dOperatorConfig,
  NewPool2dOperator,
  ParallelPool2dOperator,
};
#[cfg(feature = "mkldnn")]
pub use pool_mkldnn::{
  MklPool2dOperator,
};
pub use regress_loss::{
  RegressLossConfig,
  LstSqRegressLossConfig,
  LstSqRegressLoss,
  NormLstSqRegressLossConfig,
  //NormLstSqRegressLoss,
  IndLstSqRegressLossConfig,
  IndLstSqRegressLoss,
};
//pub use seq::{SeqOperator, SeqOperatorConfig};
pub use split::{
  SplitOperatorConfig,
  NewCopySplitOperator,
};
