//pub use super::{OperatorConfig};
pub use affine::{AffineOperatorConfig};
pub use common::{ActivationKind, ParamInitKind, PoolKind};
pub use conv::{
  Conv2dOperatorConfig,
  BatchNormConv2dOperatorConfig,
  ResidualConv2dOperatorConfig,
  ProjResidualConv2dOperatorConfig,
};
pub use graph::{GraphOperator, GraphOperatorConfig};
pub use input::{SimpleInputOperatorConfig};
pub use loss::{ClassLossConfig};
pub use pool::{Pool2dOperatorConfig};
pub use regress_loss::{RegressLossConfig};
pub use seq::{SeqOperator, SeqOperatorConfig};
