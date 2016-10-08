use prelude::*;
use common::{CommonResources};
use kernels::*;
use kernels::activate::{ActivateKernel};
use kernels::batchnorm::{BatchNorm2dKernel};
use kernels::conv::*;
use ops::*;

use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array4d};
use densearray::linalg::{Transpose};
use mkl_dnn::*;
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

pub struct Conv2dOperator {
  cfg:      Conv2dOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  weights:  Array4d<f32>,
  w_g_tmp:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  bias:     Array1d<f32>,
  b_g_tmp:  Array1d<f32>,
  b_grad:   Array1d<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  fwd:      MklDnnConv2dFwd<f32>,
  bwd_w:    MklDnnConv2dBwdKernel<f32>,
  bwd_b:    MklDnnConv2dBwdBias<f32>,
  bwd_in:   MklDnnConv2dBwdInput<f32>,
  act_kern: ActivateKernel,
  out:      CommonOperatorOutput<f32>,
}

impl Conv2dOperator {
  pub fn new(cfg: Conv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> Conv2dOperator {
    let out_len = cfg.batch_sz * cfg.out_dim().flat_len();
    let mut tmp_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_buf.push(0.0);
    }
    let mut tmp_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_grad.push(0.0);
    }
    // FIXME(20161006): how to resize batch size?
    let out_dim = cfg.out_dim();
    let conv_cfg = MklDnnConv2dConfig{
      algo:     MklDnnConvAlgo::Direct,
      in_dim:   vec![cfg.in_dim.0, cfg.in_dim.1, cfg.in_dim.2, cfg.batch_sz],
      out_dim:  vec![out_dim.0, out_dim.1, out_dim.2, cfg.batch_sz],
      w_dim:    vec![cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan],
      stride:   vec![cfg.stride_w, cfg.stride_h],
      pad:      vec![cfg.pad_w, cfg.pad_h],
      /*in_dim:   vec![cfg.batch_sz, cfg.in_dim.2, cfg.in_dim.1, cfg.in_dim.0],
      out_dim:  vec![cfg.batch_sz, out_dim.2, out_dim.1, out_dim.0],
      w_dim:    vec![cfg.out_chan, cfg.in_dim.2, cfg.kernel_h, cfg.kernel_w],
      stride:   vec![cfg.stride_h, cfg.stride_w],
      pad:      vec![cfg.pad_h, cfg.pad_w],*/
    };
    let fwd = MklDnnConv2dFwd::create(conv_cfg.clone()).unwrap();
    let bwd_w = MklDnnConv2dBwdKernel::create(conv_cfg.clone()).unwrap();
    let bwd_b = MklDnnConv2dBwdBias::create(conv_cfg.clone()).unwrap();
    let bwd_in = MklDnnConv2dBwdInput::create(conv_cfg).unwrap();
    Conv2dOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      weights:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_g_tmp:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      bias:     Array1d::zeros(cfg.out_chan),
      b_g_tmp:  Array1d::zeros(cfg.out_chan),
      b_grad:   Array1d::zeros(cfg.out_chan),
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      fwd:      fwd,
      bwd_w:    bwd_w,
      bwd_b:    bwd_b,
      bwd_in:   bwd_in,
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind, res.nnp_pool.clone()),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
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
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    self.bias.as_view_mut().set_constant(0.0);
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read(offset, self.weights.as_mut_slice());
    offset += param_reader.read(offset, self.bias.as_mut_slice());
    assert_eq!(offset - init_offset, self.diff_param_sz());
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write(offset, self.weights.as_slice());
    offset += param_writer.write(offset, self.bias.as_slice());
    assert_eq!(offset - init_offset, self.diff_param_sz());
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.weights.as_mut_slice());
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.bias.as_mut_slice());
    assert_eq!(offset - init_offset, self.diff_param_sz());
    offset - init_offset
  }

  fn reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0);
    self.b_grad.as_view_mut().set_constant(0.0);
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        let w_sz = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
        self.w_grad.as_view_mut().reshape_mut(w_sz).vector_add(lambda, self.weights.as_view().reshape(w_sz));
        self.b_grad.as_view_mut().vector_add(lambda, self.bias.as_view());
      }
    }
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write(offset, self.w_grad.as_slice());
    offset += grad_writer.write(offset, self.b_grad.as_slice());
    assert_eq!(offset - init_offset, self.diff_param_sz());
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.w_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, self.b_grad.as_slice());
    assert_eq!(offset - init_offset, self.diff_param_sz());
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
        self.bias.as_view().as_ptr(),
        self.tmp_buf.as_mut_ptr(),
        //self.nnp_pool.as_raw(),
        null_mut(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }
    /*let status = self.fwd.execute(
        self.in_.out_buf.borrow().as_ptr(),
        self.weights.as_view().as_ptr(),
        self.bias.as_view().as_ptr(),
        self.tmp_buf.as_mut_ptr(),
    );
    if status.is_err() {
      panic!("mkl convolution failed: {:?}", status);
    }*/
    //println!("DEBUG: mkl fwd success");

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
        self.b_grad.as_view_mut().as_mut_ptr(),
    ) };

    /*self.b_g_tmp.as_view_mut().set_constant(0.0);
    let status = self.bwd_b.execute(
        self.tmp_grad.as_ptr(),
        self.b_g_tmp.as_view_mut().as_mut_ptr(),
    );
    if status.is_err() {
      panic!("mkl convolution failed: {:?}", status);
    }
    self.b_grad.as_view_mut().vector_add(1.0, self.b_g_tmp.as_view());
    //println!("DEBUG: mkl bwd b grad success");*/

    let w_dim = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
    self.w_g_tmp.as_view_mut().reshape_mut(w_dim).set_constant(0.0);
    let status = unsafe { nnp_convolution_kernel_gradient(
        nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
        //nnp_convolution_algorithm::nnp_convolution_algorithm_implicit_gemm,
        batch_size,
        self.cfg.in_dim.2,
        self.cfg.out_chan,
        nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
        //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
        nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
        nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
        self.in_.out_buf.borrow().as_ptr(),
        self.tmp_grad.as_ptr(),
        self.w_g_tmp.as_view_mut().as_mut_ptr(),
        //self.nnp_pool.as_raw(),
        null_mut(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }
    /*let status = self.bwd_w.execute(
        self.in_.out_buf.borrow().as_ptr(),
        self.tmp_grad.as_ptr(),
        self.w_g_tmp.as_view_mut().as_mut_ptr(),
    );
    if status.is_err() {
      panic!("mkl convolution failed: {:?}", status);
    }*/
    self.w_grad.as_view_mut().reshape_mut(w_dim).vector_add(1.0, self.w_g_tmp.as_view().reshape(w_dim));
    //println!("DEBUG: mkl bwd w grad success");

    if let Some(in_grad) = self.in_.out_grad.as_ref() {
      let in_len = batch_size * self.cfg.in_dim.flat_len();
      in_grad.borrow_mut().reshape_mut(in_len).set_constant(0.0);
      let status = unsafe { nnp_convolution_input_gradient(
          nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
          //nnp_convolution_algorithm::nnp_convolution_algorithm_implicit_gemm,
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
      /*let status = self.bwd_in.execute(
          self.in_.out_buf.borrow().as_ptr(),
          self.weights.as_view().as_ptr(),
          self.tmp_grad.as_ptr(),
          in_grad.borrow_mut().as_mut_ptr(),
      );
      if status.is_err() {
        panic!("mkl convolution failed: {:?}", status);
      }*/
      //println!("DEBUG: mkl bwd in grad success");
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

pub struct BatchNormConv2dOperator {
  cfg:      BatchNormConv2dOperatorConfig,
  in_:      CommonOperatorOutput<f32>,
  weights:  Array4d<f32>,
  w_g_tmp:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  bias:     Array1d<f32>,
  tmp3_buf:  Vec<f32>,
  tmp3_grad: Vec<f32>,
  tmp2_buf:  Vec<f32>,
  tmp2_grad: Vec<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  fwd:      MklDnnConv2dFwdNoBias<f32>,
  bwd_w:    MklDnnConv2dBwdKernel<f32>,
  bwd_b:    MklDnnConv2dBwdBias<f32>,
  bwd_in:   MklDnnConv2dBwdInput<f32>,
  bnorm_k:  BatchNorm2dKernel,
  scale_k:  ConvScale2dKernel,
  act_kern: ActivateKernel,
  out:      CommonOperatorOutput<f32>,
}

impl BatchNormConv2dOperator {
  pub fn new(cfg: BatchNormConv2dOperatorConfig, cap: OpCapability, prev_op: &DiffOperator<f32, Output=CommonOperatorOutput<f32>, Rng=Xorshiftplus128Rng>, prev_arm: usize, res: CommonResources) -> BatchNormConv2dOperator {
    let bias = Array1d::zeros(cfg.out_chan);
    let out_len = cfg.batch_sz * cfg.out_dim().flat_len();
    let mut tmp3_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp3_buf.push(0.0);
    }
    let mut tmp3_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp3_grad.push(0.0);
    }
    let mut tmp2_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp2_buf.push(0.0);
    }
    let mut tmp2_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp2_grad.push(0.0);
    }
    let mut tmp_buf = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_buf.push(0.0);
    }
    let mut tmp_grad = Vec::with_capacity(out_len);
    for _ in 0 .. out_len {
      tmp_grad.push(0.0);
    }
    let out_dim = cfg.out_dim();
    let conv_cfg = MklDnnConv2dConfig{
      algo:     MklDnnConvAlgo::Direct,
      /*in_dim:   vec![cfg.in_dim.0, cfg.in_dim.1, cfg.in_dim.2, cfg.batch_sz],
      out_dim:  vec![out_dim.0, out_dim.1, out_dim.2, cfg.batch_sz],
      w_dim:    vec![cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan],*/
      in_dim:   vec![cfg.batch_sz, cfg.in_dim.2, cfg.in_dim.1, cfg.in_dim.0],
      out_dim:  vec![cfg.batch_sz, out_dim.2, out_dim.1, out_dim.0],
      w_dim:    vec![cfg.out_chan, cfg.in_dim.2, cfg.kernel_h, cfg.kernel_w],
      stride:   vec![cfg.stride_h, cfg.stride_w],
      pad:      vec![cfg.pad_h, cfg.pad_w],
    };
    let fwd = MklDnnConv2dFwdNoBias::create(conv_cfg.clone()).unwrap();
    let bwd_w = MklDnnConv2dBwdKernel::create(conv_cfg.clone()).unwrap();
    let bwd_b = MklDnnConv2dBwdBias::create(conv_cfg.clone()).unwrap();
    let bwd_in = MklDnnConv2dBwdInput::create(conv_cfg).unwrap();
    BatchNormConv2dOperator{
      cfg:      cfg,
      in_:      prev_op._output(prev_arm),
      weights:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_g_tmp:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      bias:     bias,
      tmp3_buf:  tmp3_buf,
      tmp3_grad: tmp3_grad,
      tmp2_buf:  tmp2_buf,
      tmp2_grad: tmp2_grad,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      fwd:      fwd,
      bwd_w:    bwd_w,
      bwd_b:    bwd_b,
      bwd_in:   bwd_in,
      bnorm_k:  BatchNorm2dKernel::new(cfg.batch_sz, cfg.out_dim(), 1.0e-6),
      scale_k:  ConvScale2dKernel::new(cfg.batch_sz, cfg.out_dim()),
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind, res.nnp_pool.clone()),
      out:      CommonOperatorOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      //_nnp_h:   NnpackHandle::new(),
      //nnp_pool: res.nnp_pool,
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
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    self.bias.as_view_mut().set_constant(0.0);
    self.bnorm_k.run_mean.as_view_mut().set_constant(0.0);
    self.bnorm_k.run_var.as_view_mut().set_constant(1.0);
    self.scale_k.scale.as_view_mut().set_constant(1.0);
    self.scale_k.bias.as_view_mut().set_constant(0.0);
  }

  fn load_param(&mut self, param_reader: &mut ReadBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read(offset, self.weights.as_mut_slice());
    //offset += param_reader.read(offset, self.bias.as_mut_slice());
    offset += param_reader.read(offset, self.scale_k.scale.as_mut_slice());
    offset += param_reader.read(offset, self.scale_k.bias.as_mut_slice());
    offset - init_offset
  }

  fn store_param(&mut self, param_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write(offset, self.weights.as_slice());
    //offset += param_writer.write(offset, self.bias.as_slice());
    offset += param_writer.write(offset, self.scale_k.scale.as_slice());
    offset += param_writer.write(offset, self.scale_k.bias.as_slice());
    offset - init_offset
  }

  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadAccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_reader.read_accumulate(alpha, beta, offset, self.weights.as_mut_slice());
    //offset += grad_reader.read_accumulate(alpha, beta, offset, self.bias.as_mut_slice());
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
    //self.b_grad.as_view_mut().set_constant(0.0);
    self.scale_k.scale_grad.as_view_mut().set_constant(0.0);
    self.scale_k.bias_grad.as_view_mut().set_constant(0.0);
  }

  fn apply_grad_reg(&mut self, reg: Regularization) {
    match reg {
      Regularization::L2(lambda) => {
        let w_sz = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
        self.w_grad.as_view_mut().reshape_mut(w_sz).vector_add(lambda, self.weights.as_view().reshape(w_sz));
        //self.b_grad.as_view_mut().vector_add(lambda, self.bias.as_view());
      }
    }
  }

  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write(offset, self.w_grad.as_slice());
    //offset += grad_writer.write(offset, self.b_grad.as_slice());
    offset += grad_writer.write(offset, self.scale_k.scale_grad.as_slice());
    offset += grad_writer.write(offset, self.scale_k.bias_grad.as_slice());
    offset - init_offset
  }

  fn accumulate_grad(&mut self, alpha: f32, beta: f32, grad_accum: &mut AccumulateBuffer<f32>, init_offset: usize) -> usize {
    let mut offset = init_offset;
    offset += grad_accum.accumulate(alpha, beta, offset, self.w_grad.as_slice());
    //offset += grad_accum.accumulate(alpha, beta, offset, self.b_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, self.scale_k.scale_grad.as_slice());
    offset += grad_accum.accumulate(alpha, beta, offset, self.scale_k.bias_grad.as_slice());
    offset - init_offset
  }

  fn forward(&mut self, _phase: OpPhase) {
    let batch_size = *self.in_.batch_size.borrow();
    *self.out.batch_size.borrow_mut() = batch_size;
    assert!(batch_size <= self.cfg.batch_sz);

    let status = unsafe { nnp_convolution_output(
        nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
        //nnp_convolution_algorithm::nnp_convolution_algorithm_implicit_gemm,
        batch_size,
        self.cfg.in_dim.2,
        self.cfg.out_chan,
        nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
        //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
        nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
        nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
        self.in_.out_buf.borrow().as_ptr(),
        self.weights.as_view().as_ptr(),
        self.bias.as_view().as_ptr(),
        self.tmp_buf.as_mut_ptr(),
        //self.tmp_buf.as_mut_ptr(),
        //self.nnp_pool.as_raw(),
        null_mut(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }

    let out_len = batch_size * self.cfg.out_dim().flat_len();
    self.bnorm_k.forward(batch_size, &self.tmp_buf[ .. out_len], &mut self.tmp2_buf[ .. out_len], 1.0);
    self.scale_k.forward(batch_size, &self.tmp2_buf[ .. out_len], &mut self.tmp3_buf[ .. out_len]);
    //activate_fwd(self.cfg.act_kind, &self.tmp3_buf, &mut *self.out.out_buf.borrow_mut());
    self.act_kern.forward(batch_size, &self.tmp3_buf, &mut *self.out.out_buf.borrow_mut());

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

    let out_len = batch_size * self.cfg.out_dim().flat_len();
    //activate_bwd(self.cfg.act_kind, &self.tmp3_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);
    self.act_kern.backward(batch_size, &self.tmp3_buf, &self.out.out_grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);
    self.scale_k.backward(batch_size, &self.tmp2_buf[ .. out_len], &self.tmp3_grad[ .. out_len], &mut self.tmp2_grad[ .. out_len]);
    self.bnorm_k.backward(batch_size, &self.tmp_buf[ .. out_len], &self.tmp2_grad[ .. out_len], &mut self.tmp_grad[ .. out_len], 1.0);

    let w_dim = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
    self.w_g_tmp.as_view_mut().reshape_mut(w_dim).set_constant(0.0);
    let status = unsafe { nnp_convolution_kernel_gradient(
        nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
        //nnp_convolution_algorithm::nnp_convolution_algorithm_implicit_gemm,
        batch_size,
        self.cfg.in_dim.2,
        self.cfg.out_chan,
        nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
        //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
        nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
        nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
        self.in_.out_buf.borrow().as_ptr(),
        self.tmp_grad.as_ptr(),
        self.w_g_tmp.as_view_mut().as_mut_ptr(),
        //self.nnp_pool.as_raw(),
        null_mut(),
        null_mut(),
    ) };
    if status.is_err() {
      panic!("nnpack convolution failed: {:?}", status);
    }
    self.w_grad.as_view_mut().reshape_mut(w_dim).vector_add(1.0, self.w_g_tmp.as_view().reshape(w_dim));

    if let Some(in_grad) = self.in_.out_grad.as_ref() {
      let in_len = batch_size * self.cfg.in_dim.flat_len();
      in_grad.borrow_mut().reshape_mut(in_len).set_constant(0.0);
      let status = unsafe { nnp_convolution_input_gradient(
          nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
          //nnp_convolution_algorithm::nnp_convolution_algorithm_implicit_gemm,
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
