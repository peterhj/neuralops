//pub use super::{OperatorConfig};
pub use affine::{
  AffineOperatorConfig,
  BatchNormAffineOperatorConfig,
  NewAffineOperator,
};
pub use class_loss::{
  ClassLossConfig,
  SoftmaxNLLClassLoss,
  EntRegSoftmaxNLLClassLossConfig,
  EntRegSoftmaxNLLClassLoss,
};
pub use common::{
  ActivationKind, ParamInitKind, PoolKind,
  CommonOperator, CommonOutput,
};
pub use conv::{
  Conv2dOperatorConfig,
  BatchNormConv2dOperatorConfig,
  ResidualConv2dOperatorConfig,
  NewResidualConv2dOperator,
  ProjResidualConv2dOperatorConfig,
  NewProjResidualConv2dOperator,
};
//#[cfg(not(feature = "mkl"))]
pub use conv_nnpack::{
  NewConv2dOperator,
  NewBatchNormConv2dOperator,
};
pub use deconv::{
  ConvTranspose2dOperatorConfig,
  BatchNormConvTranspose2dOperatorConfig,
};
pub use graph::{GraphOperator, GraphOperatorConfig};
pub use input::{
  SimpleInputOperatorConfig,
  VarInputOperatorConfig,
  VarInputPreproc,
  NewVarInputOperator,
};
pub use join::{
  JoinOperatorConfig,
  NewAddJoinOperator,
};
pub use pool::{
  Pool2dOperatorConfig,
  NewPool2dOperator,
};
pub use regress_loss::{
  RegressLossConfig,
  NormLstSqRegressLossConfig,
  LstSqRegressLoss,
  NormLstSqRegressLoss,
};
pub use seq::{SeqOperator, SeqOperatorConfig};
pub use split::{
  SplitOperatorConfig,
  NewCopySplitOperator,
};
