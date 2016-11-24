use prelude::*;
//use conv::{Conv2dOperatorConfig, BatchNorm2dOperatorConfig};
//use join::{AddJoinOperator};
//use kernels::*;
use kernels::activate::{ActivateKernel};
use kernels::batchnorm::{BatchNorm2dKernel};
use kernels::conv::*;
use kernels::ffi::*;
//use ops::*;
//use split::{CopySplitOperator};

//use densearray::{ArrayIndex, Reshape, ReshapeMut, View, ViewMut, AsView, AsViewMut, Array1d, Array4d};
//use densearray::linalg::{Transpose};
use densearray::prelude::*;
/*use nnpack::{NnpackHandle, NnpackPthreadPool};
use nnpack::ffi::*;*/
use operator::prelude::*;
use operator::io::{IoBuffer};
//use operator::rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
//use std::cmp::{max};
//use std::ptr::{null_mut};
use std::rc::{Rc};

pub struct NewConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      Conv2dOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  weights:  Array4d<f32>,
  w_g_tmp:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  bias:     Array1d<f32>,
  b_grad:   Array1d<f32>,
  col_buf:  Option<Vec<f32>>,
  col_grad: Option<Vec<f32>>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  act_kern: ActivateKernel,
}

impl<S, IoBuf: ?Sized> NewConv2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: Conv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let (col_buf, col_grad) =
        if cfg.prefer_gemm_conv() {
          let w_in_len = cfg.kernel_w * cfg.kernel_h * cfg.in_dim.2;
          let out_len = cfg.out_dim().flat_len();
          let col_len = w_in_len * out_len;
          let mut col_buf = Vec::with_capacity(col_len);
          col_buf.resize(col_len, 0.0);
          let mut col_grad = Vec::with_capacity(col_len);
          col_grad.resize(col_len, 0.0);
          (Some(col_buf), Some(col_grad))
        } else {
          (None, None)
        };
    let out_len = cfg.batch_sz * cfg.out_dim().flat_len();
    let mut tmp_buf = Vec::with_capacity(out_len);
    tmp_buf.resize(out_len, 0.0);
    let mut tmp_grad = Vec::with_capacity(out_len);
    tmp_grad.resize(out_len, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(NewConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      weights:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_g_tmp:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      bias:     Array1d::zeros(cfg.out_chan),
      b_grad:   Array1d::zeros(cfg.out_chan),
      col_buf:  col_buf,
      col_grad: col_grad,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for NewConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for NewConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for NewConv2dOperator<S, IoBuf> {
  default fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }
}

impl<S> DiffOperatorIo<[f32]> for NewConv2dOperator<S, [f32]> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.weights.as_mut_slice());
    offset += param_reader.read_buf(offset, self.bias.as_mut_slice());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write_buf(offset, self.weights.as_slice());
    offset += param_writer.write_buf(offset, self.bias.as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write_buf(offset, self.w_grad.as_slice());
    offset += grad_writer.write_buf(offset, self.b_grad.as_slice());
    offset - init_offset
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for NewConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _diff_param_sz(&self) -> usize {
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan + self.cfg.out_chan
  }

  fn _init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
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
        //let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let half_range = (3.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        //let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let std = (2.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
    }
    self.bias.as_view_mut().set_constant(0.0);
  }

  fn _reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0);
    self.b_grad.as_view_mut().set_constant(0.0);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    if !self.cfg.prefer_gemm_conv() {
      unimplemented!();
      /*let status = unsafe { nnp_convolution_output(
          nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
          batch_size,
          self.cfg.in_dim.2,
          self.cfg.out_chan,
          nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
          //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
          nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
          nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
          self.in_.buf.borrow().as_ptr(),
          self.weights.as_view().as_ptr(),
          self.bias.as_view().as_ptr(),
          self.tmp_buf.as_mut_ptr(),
          //self.nnp_pool.as_raw(),
          null_mut(),
          null_mut(),
      ) };
      if status.is_err() {
        panic!("nnpack convolution failed: {:?}", status);
      }*/
    } else {
      let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
      let in_len = self.cfg.in_dim.flat_len();
      let out_len = self.cfg.out_dim().flat_len();
      let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
      for idx in 0 .. batch_size {
        unsafe { neuralops_caffe_im2col(
            self.in_.buf.borrow()[idx * in_len .. (idx+1) * in_len].as_ptr(),
            self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
            self.cfg.kernel_h as _, self.cfg.kernel_w as _,
            self.cfg.pad_h as _, self.cfg.pad_w as _,
            self.cfg.stride_h as _, self.cfg.stride_w as _,
            1, 1,
            self.col_buf.as_mut().unwrap().as_mut_ptr(),
        ) };
        self.tmp_buf[idx * out_len .. (idx+1) * out_len]
          .reshape_mut((out_space_len, self.cfg.out_chan))
          .matrix_prod(
              1.0,
              self.col_buf.as_ref().unwrap().reshape((out_space_len, w_in_len)), Transpose::N,
              self.weights.as_view().reshape((w_in_len, self.cfg.out_chan)), Transpose::N,
              0.0);
      }
      let out_dim = self.cfg.out_dim();
      unsafe { neuralops_conv2d_bias_fwd(
          batch_size,
          out_dim.0,
          out_dim.1,
          out_dim.2,
          self.in_.buf.borrow().as_ptr(),
          self.bias.as_view().as_ptr(),
          self.tmp_grad.as_mut_ptr(),
      ) };
    }

    self.act_kern.forward(batch_size, &self.tmp_buf, &mut *self.out.buf.borrow_mut());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    self.act_kern.backward(batch_size, &self.out.buf.borrow(), &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    let out_dim = self.cfg.out_dim();
    unsafe { neuralops_conv2d_bias_bwd(
        batch_size,
        out_dim.0,
        out_dim.1,
        out_dim.2,
        self.tmp_grad.as_ptr(),
        self.b_grad.as_view_mut().as_mut_ptr(),
    ) };

    if !self.cfg.prefer_gemm_conv() {
      unimplemented!();
      /*let w_dim = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
      self.w_g_tmp.as_view_mut().reshape_mut(w_dim).set_constant(0.0);
      /*let status = unsafe { nnp_convolution_kernel_gradient(
          nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
          batch_size,
          self.cfg.in_dim.2,
          self.cfg.out_chan,
          nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
          //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
          nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
          nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
          self.in_.buf.borrow().as_ptr(),
          self.tmp_grad.as_ptr(),
          self.w_g_tmp.as_view_mut().as_mut_ptr(),
          //self.nnp_pool.as_raw(),
          null_mut(),
          null_mut(),
      ) };
      if status.is_err() {
        panic!("nnpack convolution failed: {:?}", status);
      }*/
      self.w_grad.as_view_mut().reshape_mut(w_dim).vector_add(1.0, self.w_g_tmp.as_view().reshape(w_dim));*/
    } else {
      let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
      let in_len = self.cfg.in_dim.flat_len();
      let out_len = self.cfg.out_dim().flat_len();
      let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
      for idx in 0 .. batch_size {
        unsafe { neuralops_caffe_im2col(
            self.in_.buf.borrow()[idx * in_len .. (idx+1) * in_len].as_ptr(),
            self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
            self.cfg.kernel_h as _, self.cfg.kernel_w as _,
            self.cfg.pad_h as _, self.cfg.pad_w as _,
            self.cfg.stride_h as _, self.cfg.stride_w as _,
            1, 1,
            self.col_buf.as_mut().unwrap().as_mut_ptr(),
        ) };
        self.w_grad.as_view_mut().reshape_mut((w_in_len, self.cfg.out_chan))
          .matrix_prod(
              1.0,
              self.col_buf.as_ref().unwrap().reshape((out_space_len, w_in_len)), Transpose::T,
              self.tmp_grad[idx * out_len .. (idx+1) * out_len].reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
              1.0,
          );
      }
    }

    if let Some(in_grad) = self.in_.grad.as_ref() {
      let in_len = batch_size * self.cfg.in_dim.flat_len();
      in_grad.borrow_mut().reshape_mut(in_len).set_constant(0.0);
      if !self.cfg.prefer_gemm_conv() {
        unimplemented!();
        /*let status = unsafe { nnp_convolution_input_gradient(
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
        }*/
      } else {
        let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
        let in_len = self.cfg.in_dim.flat_len();
        let out_len = self.cfg.out_dim().flat_len();
        let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
        in_grad.borrow_mut().reshape_mut(batch_size * in_len).set_constant(0.0);
        for idx in 0 .. batch_size {
          self.col_grad.as_mut().unwrap().reshape_mut((out_space_len, w_in_len))
            .matrix_prod(
                1.0,
                self.tmp_grad[idx * out_len .. (idx+1) * out_len].reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
                self.weights.as_view().reshape((w_in_len, self.cfg.out_chan)), Transpose::T,
                0.0);
          unsafe { neuralops_caffe_col2im(
              self.col_grad.as_ref().unwrap().as_ptr(),
              self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
              self.cfg.kernel_h as _, self.cfg.kernel_w as _,
              self.cfg.pad_h as _, self.cfg.pad_w as _,
              self.cfg.stride_h as _, self.cfg.stride_w as _,
              1, 1,
              in_grad.borrow_mut()[idx * in_len .. (idx+1) * in_len].as_mut_ptr(),
          ) };
        }
      }
    }
  }
}

pub struct NewBatchNormConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      BatchNormConv2dOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  weights:  Array4d<f32>,
  w_g_tmp:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  bias:     Array1d<f32>,
  b_grad:   Array1d<f32>,
  col_buf:  Option<Vec<f32>>,
  col_grad: Option<Vec<f32>>,
  tmp3_buf:  Vec<f32>,
  tmp3_grad: Vec<f32>,
  tmp2_buf:  Vec<f32>,
  tmp2_grad: Vec<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  bnorm_k:  BatchNorm2dKernel,
  scale_k:  ConvScale2dKernel,
  act_kern: ActivateKernel,
}

impl<S, IoBuf: ?Sized> NewBatchNormConv2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: BatchNormConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<NewBatchNormConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let (col_buf, col_grad) =
        if cfg.prefer_gemm_conv() {
          let w_in_len = cfg.kernel_w * cfg.kernel_h * cfg.in_dim.2;
          let out_len = cfg.out_dim().flat_len();
          let col_len = w_in_len * out_len;
          let mut col_buf = Vec::with_capacity(col_len);
          col_buf.resize(col_len, 0.0);
          let mut col_grad = Vec::with_capacity(col_len);
          col_grad.resize(col_len, 0.0);
          (Some(col_buf), Some(col_grad))
        } else {
          (None, None)
        };
    let out_len = cfg.batch_sz * cfg.out_dim().flat_len();
    let mut tmp3_buf = Vec::with_capacity(out_len);
    tmp3_buf.resize(out_len, 0.0);
    let mut tmp3_grad = Vec::with_capacity(out_len);
    tmp3_grad.resize(out_len, 0.0);
    let mut tmp2_buf = Vec::with_capacity(out_len);
    tmp2_buf.resize(out_len, 0.0);
    let mut tmp2_grad = Vec::with_capacity(out_len);
    tmp2_grad.resize(out_len, 0.0);
    let mut tmp_buf = Vec::with_capacity(out_len);
    tmp_buf.resize(out_len, 0.0);
    let mut tmp_grad = Vec::with_capacity(out_len);
    tmp_grad.resize(out_len, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(NewBatchNormConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      weights:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_g_tmp:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      bias:     Array1d::zeros(cfg.out_chan),
      b_grad:   Array1d::zeros(cfg.out_chan),
      col_buf:  col_buf,
      col_grad: col_grad,
      tmp3_buf:  tmp3_buf,
      tmp3_grad: tmp3_grad,
      tmp2_buf:  tmp2_buf,
      tmp2_grad: tmp2_grad,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      bnorm_k:  BatchNorm2dKernel::new(cfg.batch_sz, cfg.out_dim(), cfg.epsilon),
      scale_k:  ConvScale2dKernel::new(cfg.batch_sz, cfg.out_dim()),
      act_kern: ActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for NewBatchNormConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for NewBatchNormConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for NewBatchNormConv2dOperator<S, IoBuf> {
  default fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }

  default fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut IoBuf) -> usize {
    unimplemented!();
  }
}

impl<S> DiffOperatorIo<[f32]> for NewBatchNormConv2dOperator<S, [f32]> {
  fn _load_diff_param(&mut self, init_offset: usize, param_reader: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_reader.read_buf(offset, self.weights.as_mut_slice());
    offset += param_reader.read_buf(offset, self.scale_k.scale.as_mut_slice());
    offset += param_reader.read_buf(offset, self.scale_k.bias.as_mut_slice());
    offset - init_offset
  }

  fn _store_diff_param(&mut self, init_offset: usize, param_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += param_writer.write_buf(offset, self.weights.as_slice());
    offset += param_writer.write_buf(offset, self.scale_k.scale.as_slice());
    offset += param_writer.write_buf(offset, self.scale_k.bias.as_slice());
    offset - init_offset
  }

  fn _store_grad(&mut self, init_offset: usize, grad_writer: &mut [f32]) -> usize {
    let mut offset = init_offset;
    offset += grad_writer.write_buf(offset, self.w_grad.as_slice());
    offset += grad_writer.write_buf(offset, self.scale_k.scale_grad.as_slice());
    offset += grad_writer.write_buf(offset, self.scale_k.bias_grad.as_slice());
    offset - init_offset
  }
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for NewBatchNormConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.in_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.in_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _diff_param_sz(&self) -> usize {
    self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan + 2 * self.cfg.out_chan
  }

  fn _nondiff_param_sz(&self) -> usize {
    2 * self.cfg.out_chan
  }

  fn _init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
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
        //let half_range = (6.0 / (self.cfg.in_dim.2 + self.cfg.out_chan) as f64).sqrt();
        let half_range = (3.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        //let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let std = (2.0 / (self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2) as f64).sqrt();
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

  fn _reset_grad(&mut self) {
    self.w_grad.as_view_mut().set_constant(0.0);
    self.scale_k.scale_grad.as_view_mut().set_constant(0.0);
    self.scale_k.bias_grad.as_view_mut().set_constant(0.0);
  }

  fn _forward(&mut self, _phase: OpPhase) {
    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    if !self.cfg.prefer_gemm_conv() {
      unimplemented!();
      /*let status = unsafe { nnp_convolution_output(
          nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
          //nnp_convolution_algorithm::nnp_convolution_algorithm_implicit_gemm,
          batch_size,
          self.cfg.in_dim.2,
          self.cfg.out_chan,
          nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
          //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
          nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
          nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
          self.in_.buf.borrow().as_ptr(),
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
      }*/
    } else {
      let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
      let in_len = self.cfg.in_dim.flat_len();
      let out_len = self.cfg.out_dim().flat_len();
      let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
      for idx in 0 .. batch_size {
        unsafe { neuralops_caffe_im2col(
            self.in_.buf.borrow()[idx * in_len .. (idx+1) * in_len].as_ptr(),
            self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
            self.cfg.kernel_h as _, self.cfg.kernel_w as _,
            self.cfg.pad_h as _, self.cfg.pad_w as _,
            self.cfg.stride_h as _, self.cfg.stride_w as _,
            1, 1,
            self.col_buf.as_mut().unwrap().as_mut_ptr(),
        ) };
        self.tmp_buf[idx * out_len .. (idx+1) * out_len]
          .reshape_mut((out_space_len, self.cfg.out_chan))
          .matrix_prod(
              1.0,
              self.col_buf.as_ref().unwrap().reshape((out_space_len, w_in_len)), Transpose::N,
              self.weights.as_view().reshape((w_in_len, self.cfg.out_chan)), Transpose::N,
              0.0);
      }
    }

    let out_len = batch_size * self.cfg.out_dim().flat_len();
    self.bnorm_k.forward(batch_size, &self.tmp_buf[ .. out_len], &mut self.tmp2_buf[ .. out_len], 1.0);
    self.scale_k.forward(batch_size, &self.tmp2_buf[ .. out_len], &mut self.tmp3_buf[ .. out_len]);
    self.act_kern.forward(batch_size, &self.tmp3_buf, &mut *self.out.buf.borrow_mut());
  }

  fn _backward(&mut self) {
    let batch_size = self.out.batch_sz.get();

    let out_len = batch_size * self.cfg.out_dim().flat_len();
    self.act_kern.backward(batch_size, &self.out.buf.borrow(), &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp3_grad);
    self.scale_k.backward(batch_size, &self.tmp2_buf[ .. out_len], &self.tmp3_grad[ .. out_len], &mut self.tmp2_grad[ .. out_len]);
    self.bnorm_k.backward(batch_size, &self.tmp_buf[ .. out_len], &self.tmp2_grad[ .. out_len], &mut self.tmp_grad[ .. out_len], 1.0);

    if !self.cfg.prefer_gemm_conv() {
      unimplemented!();
      /*let w_dim = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
      self.w_g_tmp.as_view_mut().reshape_mut(w_dim).set_constant(0.0);
      /*let status = unsafe { nnp_convolution_kernel_gradient(
          nnp_convolution_algorithm::nnp_convolution_algorithm_auto,
          //nnp_convolution_algorithm::nnp_convolution_algorithm_implicit_gemm,
          batch_size,
          self.cfg.in_dim.2,
          self.cfg.out_chan,
          nnp_size{width: self.cfg.in_dim.0, height: self.cfg.in_dim.1},
          //nnp_padding{left: self.cfg.pad_left, right: self.cfg.pad_right, bottom: self.cfg.pad_bot, top: self.cfg.pad_top},
          nnp_padding{left: self.cfg.pad_w, right: self.cfg.pad_w, bottom: self.cfg.pad_h, top: self.cfg.pad_h},
          nnp_size{width: self.cfg.kernel_w, height: self.cfg.kernel_h},
          self.in_.buf.borrow().as_ptr(),
          self.tmp_grad.as_ptr(),
          self.w_g_tmp.as_view_mut().as_mut_ptr(),
          //self.nnp_pool.as_raw(),
          null_mut(),
          null_mut(),
      ) };
      if status.is_err() {
        panic!("nnpack convolution failed: {:?}", status);
      }*/
      self.w_grad.as_view_mut().reshape_mut(w_dim).vector_add(1.0, self.w_g_tmp.as_view().reshape(w_dim));*/
    } else {
      let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
      let in_len = self.cfg.in_dim.flat_len();
      let out_len = self.cfg.out_dim().flat_len();
      let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
      for idx in 0 .. batch_size {
        unsafe { neuralops_caffe_im2col(
            self.in_.buf.borrow()[idx * in_len .. (idx+1) * in_len].as_ptr(),
            self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
            self.cfg.kernel_h as _, self.cfg.kernel_w as _,
            self.cfg.pad_h as _, self.cfg.pad_w as _,
            self.cfg.stride_h as _, self.cfg.stride_w as _,
            1, 1,
            self.col_buf.as_mut().unwrap().as_mut_ptr(),
        ) };
        self.w_grad.as_view_mut().reshape_mut((w_in_len, self.cfg.out_chan))
          .matrix_prod(
              1.0,
              self.col_buf.as_ref().unwrap().reshape((out_space_len, w_in_len)), Transpose::T,
              self.tmp_grad[idx * out_len .. (idx+1) * out_len].reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
              1.0,
          );
      }
    }

    if let Some(in_grad) = self.in_.grad.as_ref() {
      if !self.cfg.prefer_gemm_conv() {
        let in_len = batch_size * self.cfg.in_dim.flat_len();
        in_grad.borrow_mut().reshape_mut(in_len).set_constant(0.0);
        unimplemented!();
        /*let status = unsafe { nnp_convolution_input_gradient(
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
        }*/
      } else {
        let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
        let in_len = self.cfg.in_dim.flat_len();
        let out_len = self.cfg.out_dim().flat_len();
        let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
        in_grad.borrow_mut().reshape_mut(batch_size * in_len).set_constant(0.0);
        for idx in 0 .. batch_size {
          self.col_grad.as_mut().unwrap().reshape_mut((out_space_len, w_in_len))
            .matrix_prod(
                1.0,
                self.tmp_grad[idx * out_len .. (idx+1) * out_len].reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
                self.weights.as_view().reshape((w_in_len, self.cfg.out_chan)), Transpose::T,
                0.0);
          unsafe { neuralops_caffe_col2im(
              self.col_grad.as_ref().unwrap().as_ptr(),
              self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
              self.cfg.kernel_h as _, self.cfg.kernel_w as _,
              self.cfg.pad_h as _, self.cfg.pad_w as _,
              self.cfg.stride_h as _, self.cfg.stride_w as _,
              1, 1,
              in_grad.borrow_mut()[idx * in_len .. (idx+1) * in_len].as_mut_ptr(),
          ) };
        }
      }
    }
  }
}
