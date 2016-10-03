use common::{CommonResources, CommonOperatorOutput, ActivationKind, ParamInitKind};
use join::{AddJoinOperator};
use kernels::activate::{ActivateKernel};
use kernels::batchnorm::{BatchNorm2dKernel};
use kernels::conv::*;
use split::{CopySplitOperator};

use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array4d};
use densearray::linalg::{Transpose};
use nnpack::{NnpackHandle, NnpackPthreadPool};
use nnpack::ffi::*;
use operator::prelude::*;
use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cmp::{max};
use std::ptr::{null_mut};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub enum Conv2dBackend {
  Auto,
  Nnpack,
  Mkl,
}

impl Default for Conv2dBackend {
  fn default() -> Conv2dBackend {
    Conv2dBackend::Nnpack
  }
}

#[derive(Clone, Copy, Debug)]
pub struct Conv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub kernel_w: usize,
  pub kernel_h: usize,
  pub stride_w: usize,
  pub stride_h: usize,
  pub pad_w:    usize,
  pub pad_h:    usize,
  pub out_chan: usize,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl Conv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, in_chan) = self.in_dim;
    /*let out_w = (self.pad_left  + self.pad_right  + in_w - self.kernel_w + 1) / self.stride_w;
    let out_h = (self.pad_bot   + self.pad_top    + in_h - self.kernel_h + 1) / self.stride_h;*/
    let out_w = (2 * self.pad_w + in_w - self.kernel_w + 1) / self.stride_w;
    let out_h = (2 * self.pad_h + in_h - self.kernel_h + 1) / self.stride_h;
    (out_w, out_h, self.out_chan)
  }
}

pub struct Conv2dOperator {
  cfg:      Conv2dOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  weights:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  bias:     Vec<f32>,
  b_grad:   Vec<f32>,
  //bias:     Array1d<f32>,
  //b_grad:   Array1d<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  act_kern: ActivateKernel,
  out:      CommonOperatorOutput<f32>,
  nnp_h:    NnpackHandle,
  nnp_pool: Rc<NnpackPthreadPool>,
}

impl Conv2dOperator {
  pub fn new(cfg: Conv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> Conv2dOperator {
    assert_eq!(1, cfg.stride_w);
    assert_eq!(1, cfg.stride_h);
    let mut bias = Vec::with_capacity(cfg.out_chan);
    unsafe { bias.set_len(cfg.out_chan) };
    let mut b_grad = Vec::with_capacity(cfg.out_chan);
    unsafe { b_grad.set_len(cfg.out_chan) };
    let mut tmp_buf = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp_buf.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    let mut tmp_grad = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp_grad.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    Conv2dOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      weights:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      bias:     bias,
      b_grad:   b_grad,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind, res.nnp_pool.clone()),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      nnp_h:    NnpackHandle::new(),
      nnp_pool: res.nnp_pool,
    }
  }
}

impl DiffOperator<f32> for Conv2dOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn param_len(&self) -> usize {
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan + self.cfg.out_chan
  }

  fn diff_param_sz(&self) -> usize {
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan + self.cfg.out_chan
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    match self.cfg.w_init {
      ParamInitKind::Disabled => {
        panic!("parameter initialization explicitly disabled");
      }
      ParamInitKind::Uniform{lo, hi} => {
        let dist = Range::new(lo, hi);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Xavier => {
        let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    for e in self.bias.iter_mut() {
      *e = 0.0;
    }
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.weights.as_mut_slice());
    offset += grad_reader.read_accumulate(alpha, beta, offset, &mut self.bias);
    offset - init_offset
  }

  fn reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0);
    for e in self.b_grad.iter_mut() {
      *e = 0.0;
    }
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        let weights_len = self.weights.dim().flat_len();
        let weights = self.weights.as_mut_slice();
        let w_grad = self.w_grad.as_mut_slice();
        for j in 0 .. weights_len {
          w_grad[j] += lambda * weights[j];
        }
        for j in 0 .. self.bias.len() {
          self.b_grad[j] += lambda * self.bias[j];
        }
      }
    }
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.w_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, &self.b_grad);
    offset - init_offset
  }

  fn forward(&mut self, _phase: OpPhase) {
    //assert!(self.in_.batch_size <= self.cfg.batch_sz);
    //self.out.batch_size = self.in_.batch_size;
    let batch_size = *self.in_.batch_size.borrow();
    *self.out.batch_size.borrow_mut() = batch_size;
    assert!(batch_size <= self.cfg.batch_sz);

    let status = unsafe { nnp_convolution_output(
        nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
        batch_size,
        self.cfg.in_dim.2,
        self.cfg.out_chan,
        nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
        //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
        nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
        nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
        self.in_.out_buf.borrow().as_ptr(),
        self.weights.as_view().as_ptr(),
        self.bias.as_ptr(),
        self.tmp_buf.as_mut_ptr(),
        self.nnp_pool.as_raw(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }

    //activate_fwd(self.cfg.act_kind, &self.tmp_buf, &mut *self.out.out_buf.borrow_mut());
    self.act_kern.forward(batch_size, &self.tmp_buf, &mut *self.out.out_buf.borrow_mut());

    let in_loss = *self.in_.out_loss.borrow();
    *self.out.out_loss.borrow_mut() = in_loss;
  }

  fn fwd_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        // FIXME(20161002): regularize the bias too?
        let w_norm = self.weights.as_view().reshape(self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan).l2_norm();
        *self.out.out_loss.borrow_mut() = 0.5 * lambda * w_norm * w_norm;
      }
    }
  }

  fn backward(&mut self) {
    //assert!(self.in_.batch_size <= self.cfg.batch_sz);
    //assert_eq!(self.out.batch_size, self.in_.batch_size);
    let batch_size = *self.out.batch_size.borrow();

    //activate_bwd(self.cfg.act_kind, &self.tmp_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);
    self.act_kern.backward(batch_size, &self.tmp_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    let status = unsafe { nnp_convolution_kernel_gradient(
        nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
        batch_size,
        self.cfg.in_dim.2,
        self.cfg.out_chan,
        nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
        //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
        nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
        nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
        self.in_.out_buf.borrow().as_ptr(),
        self.tmp_grad.as_ptr(),
        self.w_grad.as_view_mut().as_mut_ptr(),
        self.nnp_pool.as_raw(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }

    let out_dim = self.cfg.out_dim();
    unsafe { neuralops_conv2d_bias_bwd(
        batch_size,
        out_dim.0,
        out_dim.1,
        out_dim.2,
        self.in_.out_buf.borrow().as_ptr(),
        self.b_grad.as_mut_ptr(),
    ) };

    if let Some(in_grad) = self.in_.out_grad.as_ref() {
      let status = unsafe { nnp_convolution_input_gradient(
          nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
          batch_size,
          self.cfg.in_dim.2,
          self.cfg.out_chan,
          nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
          //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
          nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
          nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
          self.tmp_grad.as_ptr(),
          self.weights.as_view().as_ptr(),
          in_grad.borrow_mut().as_mut_ptr(),
          self.nnp_pool.as_raw(),
          null_mut(),
      ) };
      if status.is_err() {
        panic!("nnpack convolution failed: {:?}", status);
      }
    }
  }

  fn bwd_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        // FIXME(20161002)
        unimplemented!();
      }
    }
  }
}

#[derive(Clone, Copy)]
pub struct BatchNormConv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub kernel_w: usize,
  pub kernel_h: usize,
  pub stride_w: usize,
  pub stride_h: usize,
  pub pad_w:    usize,
  pub pad_h:    usize,
  pub out_chan: usize,
  pub avg_rate: f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

impl BatchNormConv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    let (in_w, in_h, in_chan) = self.in_dim;
    let out_w = (2 * self.pad_w + in_w - self.kernel_w + 1) / self.stride_w;
    let out_h = (2 * self.pad_h + in_h - self.kernel_h + 1) / self.stride_h;
    (out_w, out_h, self.out_chan)
  }
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
  nnp_h:    NnpackHandle,
  nnp_pool: Rc<NnpackPthreadPool>,
}

impl BatchNormConv2dOperator {
  pub fn new(cfg: BatchNormConv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> BatchNormConv2dOperator {
    assert_eq!(1, cfg.stride_w);
    assert_eq!(1, cfg.stride_h);
    let mut tmp_buf = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp_buf.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    let mut tmp_grad = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp_grad.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    let mut tmp2_buf = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp2_buf.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    let mut tmp2_grad = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp2_grad.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    let mut tmp3_buf = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp3_buf.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    let mut tmp3_grad = Vec::with_capacity(cfg.batch_sz * cfg.out_dim().flat_len());
    unsafe { tmp3_grad.set_len(cfg.batch_sz * cfg.out_dim().flat_len()) };
    BatchNormConv2dOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      weights:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      zerobias: Array1d::zeros(cfg.out_chan),
      tmp_buf:      tmp_buf,
      tmp_grad:     tmp_grad,
      tmp2_buf:     tmp2_buf,
      tmp2_grad:    tmp2_grad,
      tmp3_buf:     tmp3_buf,
      tmp3_grad:    tmp3_grad,
      bnorm_k:  BatchNorm2dKernel::new(cfg.batch_sz, cfg.out_dim()),
      scale_k:  ConvScale2dKernel::new(cfg.batch_sz, cfg.out_dim()),
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind, res.nnp_pool.clone()),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      nnp_h:    NnpackHandle::new(),
      nnp_pool: res.nnp_pool,
    }
  }
}

impl DiffOperator<f32> for BatchNormConv2dOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn param_len(&self) -> usize {
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan +
        2 * self.cfg.out_chan
  }

  fn diff_param_sz(&self) -> usize {
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan +
        2 * self.cfg.out_chan
  }

  fn nondiff_param_sz(&self) -> usize {
    2 * self.cfg.out_chan
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    match self.cfg.w_init {
      ParamInitKind::Disabled => {
        panic!("parameter initialization explicitly disabled");
      }
      ParamInitKind::Uniform{lo, hi} => {
        let dist = Range::new(lo, hi);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Xavier => {
        let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    self.scale_k.scale.as_view_mut().set_constant(1.0);
    self.scale_k.bias.as_view_mut().set_constant(0.0);
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read(offset, self.weights.as_mut_slice());
    offset += param_reader.read(offset, self.scale_k.scale.as_mut_slice());
    offset += param_reader.read(offset, self.scale_k.bias.as_mut_slice());
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write(offset, self.weights.as_slice());
    offset += param_writer.write(offset, self.scale_k.scale.as_slice());
    offset += param_writer.write(offset, self.scale_k.bias.as_slice());
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.weights.as_mut_slice());
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.scale_k.scale.as_mut_slice());
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.scale_k.bias.as_mut_slice());
    offset - init_offset
  }

  fn update_nondiff_param(&mut self) {
    self.bnorm_k.update(self.cfg.avg_rate);
  }

  fn reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0);
    self.scale_k.scale_grad.as_view_mut().set_constant(0.0);
    self.scale_k.bias_grad.as_view_mut().set_constant(0.0);
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        let weights_len = self.weights.dim().flat_len();
        let weights = self.weights.as_mut_slice();
        let w_grad = self.w_grad.as_mut_slice();
        for j in 0 .. weights_len {
          w_grad[j] += lambda * weights[j];
        }
        /*for j in 0 .. self.bias.len() {
          self.b_grad[j] += lambda * self.bias[j];
        }*/
      }
    }
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.w_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, self.scale_k.scale_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, self.scale_k.bias_grad.as_slice());
    offset - init_offset
  }

  fn forward(&mut self, _phase: OpPhase) {
    //assert!(self.in_.batch_size <= self.cfg.batch_sz);
    //self.out.batch_size = self.in_.batch_size;
    let batch_size = *self.in_.batch_size.borrow();
    *self.out.batch_size.borrow_mut() = batch_size;
    assert!(batch_size <= self.cfg.batch_sz);

    let status = unsafe { nnp_convolution_output(
        nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
        batch_size,
        self.cfg.in_dim.2,
        self.cfg.out_chan,
        nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
        //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
        nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
        nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
        self.in_.out_buf.borrow().as_ptr(),
        self.weights.as_view().as_ptr(),
        self.zerobias.as_view().as_ptr(),
        self.tmp_buf.as_mut_ptr(),
        self.nnp_pool.as_raw(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }

    self.bnorm_k.forward(batch_size, &self.tmp_buf, &mut self.tmp2_buf, 1.0);
    self.scale_k.forward(batch_size, &self.tmp2_buf, &mut self.tmp3_buf);

    //activate_fwd(self.cfg.act_kind, &self.tmp3_buf, &mut *self.out.out_buf.borrow_mut());
    self.act_kern.forward(batch_size, &self.tmp3_buf, &mut *self.out.out_buf.borrow_mut());

    let in_loss = *self.in_.out_loss.borrow();
    *self.out.out_loss.borrow_mut() = in_loss;
  }

  fn fwd_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        let w_norm = self.weights.as_view().reshape(self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan).l2_norm();
        *self.out.out_loss.borrow_mut() = 0.5 * lambda * w_norm * w_norm;
      }
    }
  }

  fn backward(&mut self) {
    //assert!(self.in_.batch_size <= self.cfg.batch_sz);
    //assert_eq!(self.out.batch_size, self.in_.batch_size);
    let batch_size = *self.out.batch_size.borrow();

    //activate_bwd(self.cfg.act_kind, &self.tmp3_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp3_grad);
    self.act_kern.backward(batch_size, &self.tmp3_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp3_grad);

    self.scale_k.backward(batch_size, &self.tmp2_buf, &self.tmp3_grad, &mut self.tmp2_grad);
    self.bnorm_k.backward(batch_size, &self.tmp_buf, &self.tmp2_grad, &mut self.tmp_grad, 1.0);

    let status = unsafe { nnp_convolution_kernel_gradient(
        nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
        batch_size,
        self.cfg.in_dim.2,
        self.cfg.out_chan,
        nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
        //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
        nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
        nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
        self.in_.out_buf.borrow().as_ptr(),
        self.tmp_grad.as_ptr(),
        self.w_grad.as_view_mut().as_mut_ptr(),
        self.nnp_pool.as_raw(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }

    if let Some(in_grad) = self.in_.out_grad.as_ref() {
      let status = unsafe { nnp_convolution_input_gradient(
          nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
          batch_size,
          self.cfg.in_dim.2,
          self.cfg.out_chan,
          nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
          //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
          nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
          nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
          self.tmp_grad.as_ptr(),
          self.weights.as_view().as_ptr(),
          in_grad.borrow_mut().as_mut_ptr(),
          self.nnp_pool.as_raw(),
          null_mut(),
      ) };
      if status.is_err() {
        panic!("nnpack convolution failed: {:?}", status);
      }
    }
  }

  fn bwd_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        // FIXME(20161002)
        unimplemented!();
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct ResidualConv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub avg_rate: f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

pub struct ResidualConv2dOperator {
  split:    CopySplitOperator,
  conv1:    BatchNormConv2dOperator,
  conv2:    BatchNormConv2dOperator,
  join:     AddJoinOperator,
  //act_k:    ActivateKernel,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjResidualConv2dOperatorConfig {
  pub batch_sz: usize,
  pub in_dim:   (usize, usize, usize),
  pub out_chan: usize,
  pub avg_rate: f32,
  pub act_kind: ActivationKind,
  pub w_init:   ParamInitKind,
}

pub struct ProjResidualConv2dOperator {
  split:    CopySplitOperator,
  conv0:    BatchNormConv2dOperator,
  conv1:    BatchNormConv2dOperator,
  conv2:    BatchNormConv2dOperator,
  join:     AddJoinOperator,
  //act_k:    ActivateKernel,
}
