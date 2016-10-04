use prelude::*;
use common::{CommonResources, CommonOperatorOutput};
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
  _nnp_h:   NnpackHandle,
  nnp_pool: Rc<NnpackPthreadPool>,
}

impl Conv2dOperator {
  pub fn new(cfg: Conv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> Conv2dOperator {
    assert_eq!(1, cfg.stride_w);
    assert_eq!(1, cfg.stride_h);
    let mut bias = Vec::with_capacity(cfg.out_chan);
    for _ in 0 .. cfg.out_chan {
      bias.push(0.0);
    }
    let mut b_grad = Vec::with_capacity(cfg.out_chan);
    for _ in 0 .. cfg.out_chan {
      b_grad.push(0.0);
    }
    let out_len = cfg.batch_sz * cfg.out_dim().flat_len();
    let mut tmp_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_buf.push(0.0);
    }
    let mut tmp_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_grad.push(0.0);
    }
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
      _nnp_h:   NnpackHandle::new(),
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

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read(offset, self.weights.as_mut_slice());
    offset += param_reader.read(offset, &mut self.bias);
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write(offset, self.weights.as_slice());
    offset += param_writer.write(offset, &self.bias);
    offset - init_offset
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

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write(offset, self.w_grad.as_slice());
    offset += grad_writer.write(offset, &self.b_grad);
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.w_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, &self.b_grad);
    offset - init_offset
  }

  fn forward(&mut self, _phase: OpPhase) {
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
        //self.nnp_pool.as_raw(),
        null_mut(),
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

    let out_dim = self.cfg.out_dim();
    unsafe { neuralops_conv2d_bias_bwd(
        batch_size,
        out_dim.0,
        out_dim.1,
        out_dim.2,
        self.tmp_grad.as_ptr(),
        self.b_grad.as_mut_ptr(),
    ) };

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
        //self.nnp_pool.as_raw(),
        null_mut(),
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
          //self.nnp_pool.as_raw(),
          null_mut(),
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
  _nnp_h:   NnpackHandle,
  nnp_pool: Rc<NnpackPthreadPool>,
}

impl BatchNormConv2dOperator {
  pub fn new(cfg: BatchNormConv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> BatchNormConv2dOperator {
    assert_eq!(1, cfg.stride_w);
    assert_eq!(1, cfg.stride_h);
    let out_len = cfg.batch_sz * cfg.out_dim().flat_len();
    let mut tmp_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_buf.push(0.0);
    }
    let mut tmp_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_grad.push(0.0);
    }
    let mut tmp2_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp2_buf.push(0.0);
    }
    let mut tmp2_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp2_grad.push(0.0);
    }
    let mut tmp3_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp3_buf.push(0.0);
    }
    let mut tmp3_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp3_grad.push(0.0);
    }
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
      bnorm_k:  BatchNorm2dKernel::new(cfg.batch_sz, 1.0e-6, cfg.out_dim()),
      scale_k:  ConvScale2dKernel::new(cfg.batch_sz, cfg.out_dim()),
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind, res.nnp_pool.clone()),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      _nnp_h:   NnpackHandle::new(),
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
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan
        + 2 * self.cfg.out_chan
  }

  fn diff_param_sz(&self) -> usize {
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan
        + 2 * self.cfg.out_chan
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
    self.zerobias.as_view_mut().set_constant(0.0);
    self.bnorm_k.run_mean.as_view_mut().set_constant(0.0);
    self.bnorm_k.run_var.as_view_mut().set_constant(1.0);
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
    assert_eq!(self.param_len(), offset - init_offset);
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.weights.as_mut_slice());
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.scale_k.scale.as_mut_slice());
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.scale_k.bias.as_mut_slice());
    offset - init_offset
  }

  fn update_nondiff_param(&mut self, iter: usize) {
    if iter == 0 {
      self.bnorm_k.update(1.0);
    } else {
      self.bnorm_k.update(self.cfg.avg_rate);
    }
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
        //self.nnp_pool.as_raw(),
        null_mut(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }
    //println!("DEBUG: bnorm conv: tmp_buf[100]: {:e}", self.tmp_buf[100]);

    let out_len = batch_size * self.cfg.out_dim().flat_len();
    self.bnorm_k.forward(batch_size, &self.tmp_buf[ .. out_len], &mut self.tmp2_buf[ .. out_len], 1.0);
    self.scale_k.forward(batch_size, &self.tmp2_buf[ .. out_len], &mut self.tmp3_buf[ .. out_len]);

    //activate_fwd(self.cfg.act_kind, &self.tmp3_buf, &mut *self.out.out_buf.borrow_mut());
    self.act_kern.forward(batch_size, &self.tmp3_buf[ .. out_len], &mut self.out.out_buf.borrow_mut()[ .. out_len]);

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
    let out_len = batch_size * self.cfg.out_dim().flat_len();

    //activate_bwd(self.cfg.act_kind, &self.tmp3_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp3_grad);
    self.act_kern.backward(batch_size, &self.tmp3_buf[ .. out_len], &self.out.out_grad.as_ref().unwrap().borrow()[ .. out_len], &mut self.tmp3_grad[ .. out_len]);

    //self.scale_k.backward(batch_size, &self.tmp_buf, &self.tmp3_grad, &mut self.tmp_grad);
    //self.bnorm_k.backward(batch_size, &self.tmp_buf, &self.tmp3_grad, &mut self.tmp_grad, 1.0);

    self.scale_k.backward(batch_size, &self.tmp2_buf[ .. out_len], &self.tmp3_grad[ .. out_len], &mut self.tmp2_grad[ .. out_len]);
    self.bnorm_k.backward(batch_size, &self.tmp_buf[ .. out_len], &self.tmp2_grad[ .. out_len], &mut self.tmp_grad[ .. out_len], 1.0);

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
        //self.nnp_pool.as_raw(),
        null_mut(),
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
          //self.nnp_pool.as_raw(),
          null_mut(),
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
  cfg:      ResidualConv2dOperatorConfig,
  split:    CopySplitOperator,
  conv1:    BatchNormConv2dOperator,
  conv2:    BatchNormConv2dOperator,
  join:     AddJoinOperator,
  act_k:    ActivateKernel,
  out:      CommonOperatorOutput<f32>,
}

impl ResidualConv2dOperator {
  pub fn new(cfg: ResidualConv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> ResidualConv2dOperator {
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      cfg.in_dim.flat_len(),
    };
    let conv1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,
      kernel_h: 3,
      stride_w: 1,
      stride_h: 1,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.in_dim.2,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv2_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,
      kernel_h: 3,
      stride_w: 1,
      stride_h: 1,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.in_dim.2,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.in_dim.flat_len(),
    };
    let split = CopySplitOperator::new(split_cfg, cap, prev_op, prev_arm, res.clone());
    let conv1 = BatchNormConv2dOperator::new(conv1_cfg, cap, &split, 1, res.clone());
    let conv2 = BatchNormConv2dOperator::new(conv2_cfg, cap, &conv1, 0, res.clone());
    let join = AddJoinOperator::new(join_cfg, cap, &[(&split, 0), (&conv2, 0)], res.clone());
    let act_k = ActivateKernel::new(cfg.batch_sz, cfg.in_dim.flat_len(), cfg.act_kind, res.nnp_pool.clone());
    ResidualConv2dOperator{
      cfg:      cfg,
      split:    split,
      conv1:    conv1,
      conv2:    conv2,
      join:     join,
      act_k:    act_k,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.in_dim.flat_len(), cap),
    }
  }
}

impl DiffOperator<f32> for ResidualConv2dOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn param_len(&self) -> usize {
    self.conv1.param_len()
        + self.conv2.param_len()
  }

  fn diff_param_sz(&self) -> usize {
    self.conv1.diff_param_sz()
        + self.conv2.diff_param_sz()
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    self.conv1.init_param(rng);
    self.conv2.init_param(rng);
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.load_param(param_reader, offset);
    offset += self.conv2.load_param(param_reader, offset);
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.store_param(param_writer, offset);
    offset += self.conv2.store_param(param_writer, offset);
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.update_param(alpha, beta, grad_reader, offset);
    offset += self.conv2.update_param(alpha, beta, grad_reader, offset);
    offset - init_offset
  }

  fn update_nondiff_param(&mut self, iter: usize) {
    self.conv1.update_nondiff_param(iter);
    self.conv2.update_nondiff_param(iter);
  }

  fn reset_grad(&mut self) {
    self.conv1.reset_grad();
    self.conv2.reset_grad();
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    self.conv1.apply_grad_reg(reg);
    self.conv2.apply_grad_reg(reg);
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.store_grad(grad_writer, offset);
    offset += self.conv2.store_grad(grad_writer, offset);
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv1.accumulate_grad(alpha, beta, grad_accum, offset);
    offset += self.conv2.accumulate_grad(alpha, beta, grad_accum, offset);
    offset - init_offset
  }

  fn forward(&mut self, phase: OpPhase) {
    let split_out = self.split._output(0);
    let batch_size = *split_out.batch_size.borrow();
    *self.out.batch_size.borrow_mut() = batch_size;
    self.split.forward(phase);
    self.conv1.forward(phase);
    self.conv2.forward(phase);
    self.join.forward(phase);
    let join_out = self.join._output(0);
    self.act_k.forward(batch_size, &*join_out.out_buf.borrow(), &mut *self.out.out_buf.borrow_mut());
    *self.out.batch_size.borrow_mut() = batch_size;
  }

  fn backward(&mut self) {
    let batch_size = *self.out.batch_size.borrow();
    let join_out = self.join._output(0);
    self.act_k.backward(batch_size, &*join_out.out_buf.borrow(), &*self.out.out_grad.as_ref().unwrap().borrow(), &mut *join_out.out_grad.as_ref().unwrap().borrow_mut());
    self.join.backward();
    self.conv2.backward();
    self.conv1.backward();
    self.split.backward();
  }
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

impl ProjResidualConv2dOperatorConfig {
  pub fn out_dim(&self) -> (usize, usize, usize) {
    (self.in_dim.0, self.in_dim.1, self.out_chan)
  }
}

pub struct ProjResidualConv2dOperator {
  cfg:      ProjResidualConv2dOperatorConfig,
  split:    CopySplitOperator,
  conv0:    BatchNormConv2dOperator,
  conv1:    BatchNormConv2dOperator,
  conv2:    BatchNormConv2dOperator,
  join:     AddJoinOperator,
  act_k:    ActivateKernel,
  out:      CommonOperatorOutput<f32>,
}

impl ProjResidualConv2dOperator {
  pub fn new(cfg: ProjResidualConv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> ProjResidualConv2dOperator {
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      cfg.in_dim.flat_len(),
    };
    let conv0_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 1,
      kernel_h: 1,
      stride_w: 1,
      stride_h: 1,
      pad_w:    0,
      pad_h:    0,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let conv1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,
      kernel_h: 3,
      stride_w: 1,
      stride_h: 1,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv2_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.out_dim(),
      kernel_w: 3,
      kernel_h: 3,
      stride_w: 1,
      stride_h: 1,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.out_dim().flat_len(),
    };
    let split = CopySplitOperator::new(split_cfg, cap, prev_op, prev_arm, res.clone());
    let conv0 = BatchNormConv2dOperator::new(conv0_cfg, cap, &split, 0, res.clone());
    let conv1 = BatchNormConv2dOperator::new(conv1_cfg, cap, &split, 1, res.clone());
    let conv2 = BatchNormConv2dOperator::new(conv2_cfg, cap, &conv1, 0, res.clone());
    let join = AddJoinOperator::new(join_cfg, cap, &[(&conv0, 0), (&conv2, 0)], res.clone());
    let act_k = ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind, res.nnp_pool.clone());
    ProjResidualConv2dOperator{
      cfg:      cfg,
      split:    split,
      conv0:    conv0,
      conv1:    conv1,
      conv2:    conv2,
      join:     join,
      act_k:    act_k,
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
    }
  }
}

impl DiffOperator<f32> for ProjResidualConv2dOperator {
  type Output = CommonOperatorOutput<f32>;
  type Rng = Xorshiftplus128Rng;

  fn _output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn param_len(&self) -> usize {
    self.conv0.param_len()
        + self.conv1.param_len()
        + self.conv2.param_len()
  }

  fn diff_param_sz(&self) -> usize {
    self.conv0.diff_param_sz()
        + self.conv1.diff_param_sz()
        + self.conv2.diff_param_sz()
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    self.conv0.init_param(rng);
    self.conv1.init_param(rng);
    self.conv2.init_param(rng);
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv0.load_param(param_reader, offset);
    offset += self.conv1.load_param(param_reader, offset);
    offset += self.conv2.load_param(param_reader, offset);
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv0.store_param(param_writer, offset);
    offset += self.conv1.store_param(param_writer, offset);
    offset += self.conv2.store_param(param_writer, offset);
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv0.update_param(alpha, beta, grad_reader, offset);
    offset += self.conv1.update_param(alpha, beta, grad_reader, offset);
    offset += self.conv2.update_param(alpha, beta, grad_reader, offset);
    offset - init_offset
  }

  fn update_nondiff_param(&mut self, iter: usize) {
    self.conv0.update_nondiff_param(iter);
    self.conv1.update_nondiff_param(iter);
    self.conv2.update_nondiff_param(iter);
  }

  fn reset_grad(&mut self) {
    self.conv0.reset_grad();
    self.conv1.reset_grad();
    self.conv2.reset_grad();
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    self.conv0.apply_grad_reg(reg);
    self.conv1.apply_grad_reg(reg);
    self.conv2.apply_grad_reg(reg);
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv0.store_grad(grad_writer, offset);
    offset += self.conv1.store_grad(grad_writer, offset);
    offset += self.conv2.store_grad(grad_writer, offset);
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += self.conv0.accumulate_grad(alpha, beta, grad_accum, offset);
    offset += self.conv1.accumulate_grad(alpha, beta, grad_accum, offset);
    offset += self.conv2.accumulate_grad(alpha, beta, grad_accum, offset);
    offset - init_offset
  }

  fn forward(&mut self, phase: OpPhase) {
    let split_out = self.split._output(0);
    let batch_size = *split_out.batch_size.borrow();
    *self.out.batch_size.borrow_mut() = batch_size;
    self.split.forward(phase);
    self.conv0.forward(phase);
    self.conv1.forward(phase);
    self.conv2.forward(phase);
    self.join.forward(phase);
    let join_out = self.join._output(0);
    self.act_k.forward(batch_size, &*join_out.out_buf.borrow(), &mut *self.out.out_buf.borrow_mut());
    *self.out.batch_size.borrow_mut() = batch_size;
  }

  fn backward(&mut self) {
    let batch_size = *self.out.batch_size.borrow();
    let join_out = self.join._output(0);
    self.act_k.backward(batch_size, &*join_out.out_buf.borrow(), &*self.out.out_grad.as_ref().unwrap().borrow(), &mut *join_out.out_grad.as_ref().unwrap().borrow_mut());
    self.join.backward();
    self.conv2.backward();
    self.conv1.backward();
    self.conv0.backward();
    self.split.backward();
  }
}
