use prelude::*;
use common::{CommonResources};
use kernels::activate::{ActivateKernel};
use kernels::batchnorm::{BatchNorm2dKernel};
use kernels::conv::*;
use ops::*;

use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array4d};
use densearray::linalg::{Transpose};
use mkl_dnn::*;
use operator::prelude::*;
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cmp::{max};
use std::ptr::{null_mut};
use std::rc::{Rc};

pub struct Conv2dOperator {
  cfg:      Conv2dOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  weights:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  bias:     Array1d<f32>,
  b_grad:   Array1d<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  act_kern: ActivateKernel,
  out:      CommonOperatorOutput<f32>,
}

pub struct BatchNormConv2dOperator {
  cfg:      BatchNormConv2dOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  weights:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  zerobias: Array1d<f32>,
  tmp_buf:      Vec<f32>,
  tmp_grad:     Vec<f32>,
  tmp2_buf:     Vec<f32>,
  tmp2_grad:    Vec<f32>,
  tmp3_buf:     Vec<f32>,
  tmp3_grad:    Vec<f32>,
  bnorm_k:  BatchNorm2dKernel,
  scale_k:  ConvScale2dKernel,
  act_kern: ActivateKernel,
  out:      CommonOperatorOutput<f32>,
  _nnp_h:   NnpackHandle,
  nnp_pool: Rc<NnpackPthreadPool>,
}
