use prelude::*;
use kernels::activate::{ActivateKernel, ParallelActivateKernel};
use kernels::batchnorm::{BatchNorm2dKernel};
use kernels::conv::*;
use kernels::ffi::*;

use densearray::prelude::*;
use mkl_dnn::prelude::*;
use operator::prelude::*;
use operator::io::{IoBuffer};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
//use std::cmp::{max};
//use std::ptr::{null_mut};
use std::rc::{Rc};

pub struct MklConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      Conv2dOperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  weights:  Array4d<f32>,
  w_g_tmp:  Array4d<f32>,
  w_grad:   Array4d<f32>,
  bias:     Array1d<f32>,
  b_g_tmp:  Array1d<f32>,
  b_grad:   Array1d<f32>,
  //col_buf:  Vec<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  conv_fwd:     MklDnnConv2dFwd<f32>,
  conv_bwd_w:   MklDnnConv2dBwdKernel<f32>,
  conv_bwd_b:   MklDnnConv2dBwdBias<f32>,
  conv_bwd_in:  MklDnnConv2dBwdInput<f32>,
  act_kern: ParallelActivateKernel,
  watch:    Stopwatch,
}

impl<S, IoBuf: ?Sized> MklConv2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: Conv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<MklConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    /*let col_buf = {
      let w_in_len = cfg.kernel_w * cfg.kernel_h * cfg.in_dim.2;
      let out_len = cfg.out_dim().flat_len();
      let col_len = w_in_len * out_len;
      let mut col_buf = Vec::with_capacity(col_len);
      col_buf.resize(col_len, 0.0);
      col_buf
    };*/
    let out_len = cfg.batch_sz * cfg.out_dim().flat_len();
    let mut tmp_buf = Vec::with_capacity(out_len);
    tmp_buf.resize(out_len, 0.0);
    let mut tmp_grad = Vec::with_capacity(out_len);
    tmp_grad.resize(out_len, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    let in_dim = cfg.in_dim;
    let out_dim = cfg.out_dim();
    let conv_cfg = MklDnnConv2dConfig{
      algo:     MklDnnConvAlgo::Direct,
      in_dim:   vec![in_dim.0, in_dim.1, in_dim.2, cfg.batch_sz],
      out_dim:  vec![out_dim.0, out_dim.1, out_dim.2, cfg.batch_sz],
      w_dim:    vec![cfg.kernel_w, cfg.kernel_h, in_dim.2, out_dim.2],
      stride:   vec![cfg.stride_w, cfg.stride_h],
      pad:      vec![cfg.pad_w, cfg.pad_h],
      bias:     cfg.bias,
    };
    Rc::new(RefCell::new(MklConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      weights:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_g_tmp:  Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array4d::zeros((cfg.kernel_w, cfg.kernel_h, cfg.in_dim.2, cfg.out_chan)),
      bias:     Array1d::zeros(cfg.out_chan),
      b_g_tmp:  Array1d::zeros(cfg.out_chan),
      b_grad:   Array1d::zeros(cfg.out_chan),
      //col_buf:  col_buf,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      conv_fwd:     MklDnnConv2dFwd::create(conv_cfg.clone()).unwrap(),
      conv_bwd_w:   MklDnnConv2dBwdKernel::create(conv_cfg.clone()).unwrap(),
      conv_bwd_b:   MklDnnConv2dBwdBias::create(conv_cfg.clone()).unwrap(),
      conv_bwd_in:  MklDnnConv2dBwdInput::create(conv_cfg.clone()).unwrap(),
      act_kern: ParallelActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind),
      watch:    Stopwatch::new(),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for MklConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for MklConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for MklConv2dOperator<S, IoBuf> {
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

impl<S> DiffOperatorIo<[f32]> for MklConv2dOperator<S, [f32]> {
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

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for MklConv2dOperator<S, IoBuf> {
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
    self.watch.lap();

    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    unsafe { self.conv_fwd.execute(
        self.in_.buf.borrow().as_ptr(),
        self.weights.as_view().as_ptr(),
        if self.cfg.bias {
          Some(self.bias.as_view().as_ptr())
        } else {
          None
        },
        self.tmp_buf.as_mut_ptr(),
    ).unwrap() };

    self.act_kern.forward(batch_size, &self.tmp_buf, &mut *self.out.buf.borrow_mut());

    self.watch.lap();
    println!("DEBUG: conv2d: fwd: {:.6}", self.watch.elapsed());
  }

  fn _backward(&mut self) {
    self.watch.lap();

    let batch_size = self.out.batch_sz.get();

    self.act_kern.backward(batch_size, &self.out.buf.borrow(), &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    if self.cfg.bias {
      self.b_g_tmp.as_view_mut().set_constant(0.0);
      unsafe { self.conv_bwd_b.execute(
          self.tmp_grad.as_ptr(),
          self.b_g_tmp.as_view_mut().as_mut_ptr(),
      ).unwrap() };
      self.b_grad.as_view_mut().parallel_add(1.0, self.b_g_tmp.as_view());
    }

    let w_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
    self.w_g_tmp.as_view_mut().reshape_mut(w_len).set_constant(0.0);
    unsafe { self.conv_bwd_w.execute(
        self.in_.buf.borrow().as_ptr(),
        self.tmp_grad.as_ptr(),
        self.w_g_tmp.as_view_mut().as_mut_ptr(),
    ).unwrap() };
    self.w_grad.as_view_mut().reshape_mut(w_len).parallel_add(1.0, self.w_g_tmp.as_view().reshape(w_len));

    if let Some(in_grad) = self.in_.grad.as_ref() {
      let in_len = self.cfg.in_dim.flat_len();
      in_grad.borrow_mut().reshape_mut(batch_size * in_len).set_constant(0.0);
      unsafe { self.conv_bwd_in.execute(
          self.weights.as_view().as_ptr(),
          self.tmp_grad.as_ptr(),
          in_grad.borrow_mut().as_mut_ptr(),
      ).unwrap() };
    }

    self.watch.lap();
    println!("DEBUG: conv2d: bwd: {:.6}", self.watch.elapsed());
  }
}

pub struct ParallelConv2d1x1Operator<S, IoBuf: ?Sized> {
  cfg:      Conv2d1x1OperatorConfig,
  node:     OperatorNode,
  in_op:    Rc<RefCell<DiffOperator<S, IoBuf>>>,
  in_:      CommonOutput,
  out:      CommonOutput,
  weights:  Array2d<f32>,
  w_g_tmp:  Array2d<f32>,
  w_grad:   Array2d<f32>,
  bias:     Array1d<f32>,
  b_g_tmp:  Array1d<f32>,
  b_grad:   Array1d<f32>,
  //col_buf:  Vec<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  act_kern: ParallelActivateKernel,
  watch:    Stopwatch,
}

impl<S, IoBuf: ?Sized> ParallelConv2d1x1Operator<S, IoBuf> {
  pub fn new<InOp>(cfg: Conv2d1x1OperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<ParallelConv2d1x1Operator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    /*let col_buf = {
      let w_in_len = cfg.kernel_w * cfg.kernel_h * cfg.in_dim.2;
      let out_len = cfg.out_dim().flat_len();
      let col_len = w_in_len * out_len;
      let mut col_buf = Vec::with_capacity(col_len);
      col_buf.resize(col_len, 0.0);
      col_buf
    };*/
    let out_len = cfg.batch_sz * cfg.out_dim().flat_len();
    let mut tmp_buf = Vec::with_capacity(out_len);
    tmp_buf.resize(out_len, 0.0);
    let mut tmp_grad = Vec::with_capacity(out_len);
    tmp_grad.resize(out_len, 0.0);
    let in_ = prev_op.borrow()._output(prev_arm);
    Rc::new(RefCell::new(ParallelConv2d1x1Operator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      in_op:    prev_op,
      in_:      in_,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      weights:  Array2d::zeros((cfg.in_dim.2, cfg.out_chan)),
      w_g_tmp:  Array2d::zeros((cfg.in_dim.2, cfg.out_chan)),
      w_grad:   Array2d::zeros((cfg.in_dim.2, cfg.out_chan)),
      bias:     Array1d::zeros(cfg.out_chan),
      b_g_tmp:  Array1d::zeros(cfg.out_chan),
      b_grad:   Array1d::zeros(cfg.out_chan),
      //col_buf:  col_buf,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      act_kern: ParallelActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind),
      watch:    Stopwatch::new(),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for ParallelConv2d1x1Operator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for ParallelConv2d1x1Operator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for ParallelConv2d1x1Operator<S, IoBuf> {
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

impl<S> DiffOperatorIo<[f32]> for ParallelConv2d1x1Operator<S, [f32]> {
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

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for ParallelConv2d1x1Operator<S, IoBuf> {
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
    self.cfg.in_dim.2 * self.cfg.out_chan + self.cfg.out_chan
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
        let half_range = (3.0 / (self.cfg.in_dim.2) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for e in self.weights.as_mut_slice().iter_mut() {
          *e = dist.ind_sample(rng) as f32;
        }
      }
      ParamInitKind::Kaiming => {
        //let std = (2.0 / max(self.cfg.in_dim.2, self.cfg.out_chan) as f64).sqrt();
        //let std = (2.0 / self.cfg.in_dim.2 as f64).sqrt();
        let std = (2.0 / (self.cfg.in_dim.2) as f64).sqrt();
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
    self.watch.lap();

    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    {
      let in_chan = self.cfg.in_dim.2;
      let in_len = self.cfg.in_dim.flat_len();
      let out_len = self.cfg.out_dim().flat_len();
      let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
      for idx in 0 .. batch_size {
        self.tmp_buf[idx * out_len .. (idx+1) * out_len].reshape_mut((out_space_len, self.cfg.out_chan))
          .matrix_prod(
              1.0,
              self.in_.buf.borrow()[idx * in_len .. (idx+1) * in_len].reshape((out_space_len, in_chan)), Transpose::N,
              self.weights.as_view(), Transpose::N,
              0.0);
      }
    }

    if self.cfg.bias {
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

    self.watch.lap();
    println!("DEBUG: conv2d 1x1: fwd: {:.6}", self.watch.elapsed());
  }

  fn _backward(&mut self) {
    self.watch.lap();

    let batch_size = self.out.batch_sz.get();

    self.act_kern.backward(batch_size, &self.out.buf.borrow(), &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp_grad);

    if self.cfg.bias {
      let out_dim = self.cfg.out_dim();
      unsafe { neuralops_conv2d_bias_bwd(
          batch_size,
          out_dim.0,
          out_dim.1,
          out_dim.2,
          self.tmp_grad.as_ptr(),
          self.b_grad.as_view_mut().as_mut_ptr(),
      ) };
    }

    {
      let in_chan = self.cfg.in_dim.2;
      let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
      let in_len = self.cfg.in_dim.flat_len();
      let out_len = self.cfg.out_dim().flat_len();
      let in_buf = self.in_.buf.borrow();
      for idx in 0 .. batch_size {
        self.w_grad.as_view_mut()
          .matrix_prod(
              1.0,
              in_buf[idx * in_len .. (idx+1) * in_len].reshape((out_space_len, in_chan)), Transpose::T,
              self.tmp_grad[idx * out_len .. (idx+1) * out_len].reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
              1.0,
          );
      }
    }

    if let Some(in_grad) = self.in_.grad.as_ref() {
      {
        let in_chan = self.cfg.in_dim.2;
        let in_len = self.cfg.in_dim.flat_len();
        let out_len = self.cfg.out_dim().flat_len();
        let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
        //in_grad.borrow_mut().reshape_mut(batch_size * in_len).set_constant(0.0);
        for idx in 0 .. batch_size {
          in_grad.borrow_mut()[idx * in_len .. (idx+1) * in_len].reshape_mut((out_space_len, in_chan))
            .matrix_prod(
                1.0,
                self.tmp_grad[idx * out_len .. (idx+1) * out_len].reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
                self.weights.as_view(), Transpose::T,
                0.0);
        }
      }
    }

    self.watch.lap();
    println!("DEBUG: conv2d 1x1: bwd: {:.6}", self.watch.elapsed());
  }
}

pub struct MklBatchNormConv2dOperator<S, IoBuf: ?Sized> {
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
  //col_buf:  Vec<f32>,
  tmp3_buf:  Vec<f32>,
  tmp3_grad: Vec<f32>,
  tmp2_buf:  Vec<f32>,
  tmp2_grad: Vec<f32>,
  tmp_buf:  Vec<f32>,
  tmp_grad: Vec<f32>,
  /*conv_src_buf: MklDnnBuffer<f32>,
  conv_w_buf:   MklDnnBuffer<f32>,
  conv_dst_buf: MklDnnBuffer<f32>,*/
  conv_fwd:     MklDnnConv2dFwd<f32>,
  conv_bwd_w:   MklDnnConv2dBwdKernel<f32>,
  conv_bwd_in:  MklDnnConv2dBwdInput<f32>,
  // FIXME(20161128): parallel versions of batchnorm and convscale.
  bnorm_k:  BatchNorm2dKernel,
  scale_k:  ConvScale2dKernel,
  act_kern: ParallelActivateKernel,
  watch:    Stopwatch,
}

impl<S, IoBuf: ?Sized> MklBatchNormConv2dOperator<S, IoBuf> {
  pub fn new<InOp>(cfg: BatchNormConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<MklBatchNormConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    /*let col_buf = {
      let w_in_len = cfg.kernel_w * cfg.kernel_h * cfg.in_dim.2;
      let out_len = cfg.out_dim().flat_len();
      let col_len = w_in_len * out_len;
      let mut col_buf = Vec::with_capacity(col_len);
      col_buf.resize(col_len, 0.0);
      col_buf
    };*/
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
    //let batch_sz = cfg.batch_sz;
    let in_dim = cfg.in_dim;
    let out_dim = cfg.out_dim();
    let conv_cfg = MklDnnConv2dConfig{
      algo:     MklDnnConvAlgo::Direct,
      in_dim:   vec![in_dim.0, in_dim.1, in_dim.2, cfg.batch_sz],
      out_dim:  vec![out_dim.0, out_dim.1, out_dim.2, cfg.batch_sz],
      w_dim:    vec![cfg.kernel_w, cfg.kernel_h, in_dim.2, out_dim.2],
      stride:   vec![cfg.stride_w, cfg.stride_h],
      pad:      vec![cfg.pad_w, cfg.pad_h],
      bias:     false,
    };
    let conv_src_buf = MklDnnBuffer::create(MklDnnLayout::create(vec![in_dim.0, in_dim.1, in_dim.2, cfg.batch_sz]).unwrap()).unwrap();
    let conv_w_buf = MklDnnBuffer::create(MklDnnLayout::create(vec![cfg.kernel_w, cfg.kernel_h, in_dim.2, out_dim.2]).unwrap()).unwrap();
    let conv_dst_buf = MklDnnBuffer::create(MklDnnLayout::create(vec![out_dim.0, out_dim.1, out_dim.2, cfg.batch_sz]).unwrap()).unwrap();
    Rc::new(RefCell::new(MklBatchNormConv2dOperator{
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
      //col_buf:  col_buf,
      tmp3_buf:  tmp3_buf,
      tmp3_grad: tmp3_grad,
      tmp2_buf:  tmp2_buf,
      tmp2_grad: tmp2_grad,
      tmp_buf:  tmp_buf,
      tmp_grad: tmp_grad,
      // FIXME(20161128): parallel versions of batchnorm and convscale.
      /*conv_src_buf: conv_src_buf,
      conv_w_buf:   conv_w_buf,
      conv_dst_buf: conv_dst_buf,*/
      conv_fwd:     MklDnnConv2dFwd::create(conv_cfg.clone()).unwrap(),
      conv_bwd_w:   MklDnnConv2dBwdKernel::create(conv_cfg.clone()).unwrap(),
      conv_bwd_in:  MklDnnConv2dBwdInput::create(conv_cfg.clone()).unwrap(),
      bnorm_k:  BatchNorm2dKernel::new(cfg.batch_sz, cfg.out_dim(), cfg.epsilon),
      scale_k:  ConvScale2dKernel::new(cfg.batch_sz, cfg.out_dim()),
      act_kern: ParallelActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind),
      watch:    Stopwatch::new(),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for MklBatchNormConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for MklBatchNormConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for MklBatchNormConv2dOperator<S, IoBuf> {
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

impl<S> DiffOperatorIo<[f32]> for MklBatchNormConv2dOperator<S, [f32]> {
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

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for MklBatchNormConv2dOperator<S, IoBuf> {
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
    self.watch.lap();

    let batch_size = self.in_.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    assert!(batch_size <= self.cfg.batch_sz);

    unsafe { self.conv_fwd.execute(
        self.in_.buf.borrow().as_ptr(),
        self.weights.as_view().as_ptr(),
        None,
        self.tmp_buf.as_mut_ptr(),
    ).unwrap() };

    /*self.conv_src_buf.copy_from(&*self.in_.buf.borrow());
    self.conv_w_buf.copy_from(self.weights.as_slice());
    unsafe { self.conv_fwd.execute(
        self.conv_src_buf.as_ptr(),
        self.conv_w_buf.as_ptr(),
        None,
        self.conv_dst_buf.as_mut_ptr(),
    ).unwrap() };
    self.conv_dst_buf.copy_to(&mut self.tmp_buf);*/

    /*{
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
            self.col_buf.as_mut_ptr(),
        ) };
        self.tmp_buf[idx * out_len .. (idx+1) * out_len]
          .reshape_mut((out_space_len, self.cfg.out_chan))
          .matrix_prod(
              1.0,
              self.col_buf.reshape((out_space_len, w_in_len)), Transpose::N,
              self.weights.as_view().reshape((w_in_len, self.cfg.out_chan)), Transpose::N,
              0.0);
      }
    }*/

    let out_len = batch_size * self.cfg.out_dim().flat_len();
    self.bnorm_k.forward(batch_size, &self.tmp_buf[ .. out_len], &mut self.tmp2_buf[ .. out_len], 1.0);
    self.scale_k.forward(batch_size, &self.tmp2_buf[ .. out_len], &mut self.tmp3_buf[ .. out_len]);
    self.act_kern.forward(batch_size, &self.tmp3_buf, &mut *self.out.buf.borrow_mut());

    self.watch.lap();
    //println!("DEBUG: conv2d: fwd: {:.6}", self.watch.elapsed());
  }

  fn _backward(&mut self) {
    self.watch.lap();

    let batch_size = self.out.batch_sz.get();

    let out_len = batch_size * self.cfg.out_dim().flat_len();
    self.act_kern.backward(batch_size, &self.out.buf.borrow(), &self.out.grad.as_ref().unwrap().borrow(), &mut self.tmp3_grad);
    self.scale_k.backward(batch_size, &self.tmp2_buf[ .. out_len], &self.tmp3_grad[ .. out_len], &mut self.tmp2_grad[ .. out_len]);
    self.bnorm_k.backward(batch_size, &self.tmp_buf[ .. out_len], &self.tmp2_grad[ .. out_len], &mut self.tmp_grad[ .. out_len], 1.0);

    let w_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2 * self.cfg.out_chan;
    self.w_g_tmp.as_view_mut().reshape_mut(w_len).set_constant(0.0);
    unsafe { self.conv_bwd_w.execute(
        self.in_.buf.borrow().as_ptr(),
        self.tmp_grad.as_ptr(),
        self.w_grad.as_view_mut().as_mut_ptr(),
    ).unwrap() };
    self.w_grad.as_view_mut().reshape_mut(w_len).parallel_add(1.0, self.w_g_tmp.as_view().reshape(w_len));

    /*self.conv_src_buf.copy_from(&*self.in_.buf.borrow());
    self.conv_dst_buf.copy_from(&self.tmp_grad);
    unsafe { self.conv_bwd_w.execute(
        self.conv_src_buf.as_ptr(),
        self.conv_dst_buf.as_ptr(),
        self.conv_w_buf.as_mut_ptr(),
    ).unwrap() };
    self.conv_w_buf.copy_to(self.weights.as_mut_slice());*/

    /*{
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
            self.col_buf.as_mut_ptr(),
        ) };
        self.w_grad.as_view_mut().reshape_mut((w_in_len, self.cfg.out_chan))
          .matrix_prod(
              1.0,
              self.col_buf.reshape((out_space_len, w_in_len)), Transpose::T,
              self.tmp_grad[idx * out_len .. (idx+1) * out_len].reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
              1.0,
          );
      }
    }*/

    if let Some(in_grad) = self.in_.grad.as_ref() {
      let in_len = self.cfg.in_dim.flat_len();
      in_grad.borrow_mut().reshape_mut(batch_size * in_len).set_constant(0.0);

      unsafe { self.conv_bwd_in.execute(
          self.weights.as_view().as_ptr(),
          self.tmp_grad.as_ptr(),
          in_grad.borrow_mut().as_mut_ptr(),
      ).unwrap() };

      /*self.conv_w_buf.copy_from(self.weights.as_slice());
      self.conv_dst_buf.copy_from(&self.tmp_grad);
      unsafe { self.conv_bwd_in.execute(
          self.conv_w_buf.as_ptr(),
          self.conv_dst_buf.as_ptr(),
          self.conv_src_buf.as_mut_ptr(),
      ).unwrap() };
      self.conv_src_buf.copy_to(&mut *in_grad.borrow_mut());*/

      /*{
        let w_in_len = self.cfg.kernel_w * self.cfg.kernel_h * self.cfg.in_dim.2;
        let in_len = self.cfg.in_dim.flat_len();
        let out_len = self.cfg.out_dim().flat_len();
        let out_space_len = self.cfg.out_dim().0 * self.cfg.out_dim().1;
        in_grad.borrow_mut().reshape_mut(batch_size * in_len).set_constant(0.0);
        for idx in 0 .. batch_size {
          self.col_buf.reshape_mut((out_space_len, w_in_len))
            .matrix_prod(
                1.0,
                self.tmp_grad[idx * out_len .. (idx+1) * out_len].reshape((out_space_len, self.cfg.out_chan)), Transpose::N,
                self.weights.as_view().reshape((w_in_len, self.cfg.out_chan)), Transpose::T,
                0.0);
          unsafe { neuralops_caffe_col2im(
              self.col_buf.as_ptr(),
              self.cfg.in_dim.2 as _, self.cfg.in_dim.1 as _, self.cfg.in_dim.0 as _,
              self.cfg.kernel_h as _, self.cfg.kernel_w as _,
              self.cfg.pad_h as _, self.cfg.pad_w as _,
              self.cfg.stride_h as _, self.cfg.stride_w as _,
              1, 1,
              in_grad.borrow_mut()[idx * in_len .. (idx+1) * in_len].as_mut_ptr(),
          ) };
        }
      }*/
    }

    self.watch.lap();
    //println!("DEBUG: conv2d: bwd: {:.6}", self.watch.elapsed());
  }
}

pub struct MklResidualConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      ResidualConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<NewAddJoinOperator<S, IoBuf>>>,
  out:      CommonOutput,
  act_k:    ParallelActivateKernel,
}

impl<S, IoBuf: ?Sized> MklResidualConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: ResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<MklResidualConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
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
      epsilon:  cfg.epsilon,
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
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.in_dim.flat_len(),
    };
    let split_op = NewCopySplitOperator::new(split_cfg, cap, prev_op, prev_arm);
    let conv1_op = MklBatchNormConv2dOperator::new(conv1_cfg, cap, split_op.clone(), 0);
    let conv2_op = MklBatchNormConv2dOperator::new(conv2_cfg, cap, conv1_op, 0);
    let join_op = NewAddJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv2_op, 0);
    join_op.borrow_mut().append_input(split_op, 1);
    Rc::new(RefCell::new(MklResidualConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
      out:      CommonOutput::new(cfg.batch_sz, cfg.in_dim.flat_len(), cap),
      act_k:    ParallelActivateKernel::new(cfg.batch_sz, cfg.in_dim.flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for MklResidualConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for MklResidualConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for MklResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for MklResidualConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
    let join_out = self.join_op.borrow()._output(0);
    let batch_size = join_out.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    self.act_k.forward(batch_size, &*join_out.buf.borrow(), &mut *self.out.buf.borrow_mut());
  }

  fn _backward(&mut self) {
    let join_out = self.join_op.borrow()._output(0);
    if let Some(ref join_grad) = join_out.grad.as_ref() {
      let batch_size = self.out.batch_sz.get();
      self.act_k.backward(batch_size, &*join_out.buf.borrow(), &*self.out.grad.as_ref().unwrap().borrow(), &mut *join_grad.borrow_mut());
    }
  }
}

pub struct MklProjResidualConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      ProjResidualConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<NewAddJoinOperator<S, IoBuf>>>,
  out:      CommonOutput,
  act_k:    ParallelActivateKernel,
}

impl<S, IoBuf: ?Sized> MklProjResidualConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: ProjResidualConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<MklProjResidualConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
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
      stride_w: cfg.stride_w,
      stride_h: cfg.stride_h,
      pad_w:    1,
      pad_h:    1,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      epsilon:  cfg.epsilon,
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
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let conv1x1_cfg = BatchNormConv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 1,
      kernel_h: 1,
      stride_w: cfg.stride_w,
      stride_h: cfg.stride_h,
      pad_w:    0,
      pad_h:    0,
      out_chan: cfg.out_chan,
      avg_rate: cfg.avg_rate,
      epsilon:  cfg.epsilon,
      act_kind: ActivationKind::Identity,
      w_init:   cfg.w_init,
    };
    let join_cfg = JoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      dim:      cfg.out_dim().flat_len(),
    };
    let split_op = NewCopySplitOperator::new(split_cfg, cap, prev_op, prev_arm);
    let conv1_op = MklBatchNormConv2dOperator::new(conv1_cfg, cap, split_op.clone(), 0);
    let conv2_op = MklBatchNormConv2dOperator::new(conv2_cfg, cap, conv1_op, 0);
    let conv1x1_op = MklBatchNormConv2dOperator::new(conv1x1_cfg, cap, split_op, 1);
    let join_op = NewAddJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv2_op, 0);
    join_op.borrow_mut().append_input(conv1x1_op, 0);
    Rc::new(RefCell::new(MklProjResidualConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
      out:      CommonOutput::new(cfg.batch_sz, cfg.out_dim().flat_len(), cap),
      act_k:    ParallelActivateKernel::new(cfg.batch_sz, cfg.out_dim().flat_len(), cfg.act_kind),
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for MklProjResidualConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for MklProjResidualConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    self.out.clone()
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for MklProjResidualConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for MklProjResidualConv2dOperator<S, IoBuf> {
  //type IoBuf = [f32];

  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
    let join_out = self.join_op.borrow()._output(0);
    let batch_size = join_out.batch_sz.get();
    self.out.batch_sz.set(batch_size);
    self.act_k.forward(batch_size, &*join_out.buf.borrow(), &mut *self.out.buf.borrow_mut());
  }

  fn _backward(&mut self) {
    let join_out = self.join_op.borrow()._output(0);
    if let Some(ref join_grad) = join_out.grad.as_ref() {
      let batch_size = self.out.batch_sz.get();
      self.act_k.backward(batch_size, &*join_out.buf.borrow(), &*self.out.grad.as_ref().unwrap().borrow(), &mut *join_grad.borrow_mut());
    }
  }
}

pub struct MklSqueezeConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      SqueezeConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<ConcatJoinOperator<S, IoBuf>>>,
}

impl<S, IoBuf: ?Sized> MklSqueezeConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: SqueezeConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<MklSqueezeConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let squeeze_dim = cfg.squeeze_dim();
    let expand_chan = cfg.out_chan / 2;
    assert_eq!(0, cfg.out_chan % 2);
    assert_eq!(1, cfg.stride_w);
    assert_eq!(1, cfg.stride_h);
    /*let conv1_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 1,            kernel_h: 1,
      stride_w: cfg.stride_w, stride_h: cfg.stride_h,
      pad_w:    0,            pad_h:    0,
      out_chan: cfg.squeeze,
      bias:     false,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };*/
    let conv1_cfg = Conv2d1x1OperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      out_chan: cfg.squeeze,
      bias:     true,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      squeeze_dim.flat_len(),
    };
    /*let conv1x1_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      kernel_w: 1,  kernel_h: 1,
      stride_w: 1,  stride_h: 1,
      pad_w:    0,  pad_h:    0,
      out_chan: expand_chan,
      bias:     false,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };*/
    let conv1x1_cfg = Conv2d1x1OperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      out_chan: expand_chan,
      bias:     true,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv3x3_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      kernel_w: 3,  kernel_h: 3,
      stride_w: 1,  stride_h: 1,
      pad_w:    1,  pad_h:    1,
      out_chan: expand_chan,
      bias:     true,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let join_cfg = ConcatJoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      in_dims:  vec![conv1x1_cfg.out_dim().flat_len(), conv3x3_cfg.out_dim().flat_len()],
    };
    let conv1_op = ParallelConv2d1x1Operator::new(conv1_cfg, cap, prev_op, prev_arm);
    let split_op = NewCopySplitOperator::new(split_cfg, cap, conv1_op, 0);
    let conv1x1_op = ParallelConv2d1x1Operator::new(conv1x1_cfg, cap, split_op.clone(), 0);
    let conv3x3_op = MklConv2dOperator::new(conv3x3_cfg, cap, split_op.clone(), 1);
    let join_op = ConcatJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv1x1_op, 0);
    join_op.borrow_mut().append_input(conv3x3_op, 0);
    Rc::new(RefCell::new(MklSqueezeConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for MklSqueezeConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for MklSqueezeConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    let join_out = self.join_op.borrow()._output(0);
    join_out
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for MklSqueezeConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for MklSqueezeConv2dOperator<S, IoBuf> {
  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
  }

  fn _backward(&mut self) {
  }
}

pub struct MklProjSqueezeConv2dOperator<S, IoBuf: ?Sized> {
  cfg:      SqueezeConv2dOperatorConfig,
  node:     OperatorNode,
  join_op:  Rc<RefCell<ConcatJoinOperator<S, IoBuf>>>,
}

impl<S, IoBuf: ?Sized> MklProjSqueezeConv2dOperator<S, IoBuf> where S: 'static, IoBuf: 'static {
  pub fn new<InOp>(cfg: SqueezeConv2dOperatorConfig, cap: OpCapability, prev_op: Rc<RefCell<InOp>>, prev_arm: usize) -> Rc<RefCell<MklProjSqueezeConv2dOperator<S, IoBuf>>> where InOp: 'static + CommonOperator + DiffOperator<S, IoBuf> {
    let squeeze_dim = cfg.squeeze_dim();
    let expand_chan = cfg.out_chan / 2;
    assert_eq!(0, cfg.out_chan % 2);
    let conv1_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   cfg.in_dim,
      kernel_w: 3,            kernel_h: 3,
      stride_w: cfg.stride_w, stride_h: cfg.stride_h,
      pad_w:    1,            pad_h:    1,
      out_chan: cfg.squeeze,
      bias:     true,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let split_cfg = SplitOperatorConfig{
      batch_sz: cfg.batch_sz,
      out_arms: 2,
      dim:      squeeze_dim.flat_len(),
    };
    let conv1x1_cfg = Conv2d1x1OperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      out_chan: expand_chan,
      bias:     true,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let conv3x3_cfg = Conv2dOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_dim:   squeeze_dim,
      kernel_w: 3,  kernel_h: 3,
      stride_w: 1,  stride_h: 1,
      pad_w:    1,  pad_h:    1,
      out_chan: expand_chan,
      bias:     true,
      act_kind: ActivationKind::Rect,
      w_init:   cfg.w_init,
    };
    let join_cfg = ConcatJoinOperatorConfig{
      batch_sz: cfg.batch_sz,
      in_arms:  2,
      in_dims:  vec![conv1x1_cfg.out_dim().flat_len(), conv3x3_cfg.out_dim().flat_len()],
    };
    let conv1_op = MklConv2dOperator::new(conv1_cfg, cap, prev_op, prev_arm);
    let split_op = NewCopySplitOperator::new(split_cfg, cap, conv1_op, 0);
    let conv1x1_op = ParallelConv2d1x1Operator::new(conv1x1_cfg, cap, split_op.clone(), 0);
    let conv3x3_op = MklConv2dOperator::new(conv3x3_cfg, cap, split_op.clone(), 1);
    let join_op = ConcatJoinOperator::new(join_cfg, cap);
    join_op.borrow_mut().append_input(conv1x1_op, 0);
    join_op.borrow_mut().append_input(conv3x3_op, 0);
    Rc::new(RefCell::new(MklProjSqueezeConv2dOperator{
      cfg:      cfg,
      node:     OperatorNode::default(),
      join_op:  join_op,
    }))
  }
}

impl<S, IoBuf: ?Sized> Operator for MklProjSqueezeConv2dOperator<S, IoBuf> {
  fn _next(&self) -> u64 {
    self.node._next()
  }
}

impl<S, IoBuf: ?Sized> CommonOperator for MklProjSqueezeConv2dOperator<S, IoBuf> {
  fn _output(&self, arm: usize) -> CommonOutput {
    assert_eq!(0, arm);
    let join_out = self.join_op.borrow()._output(0);
    join_out
  }
}

impl<S, IoBuf: ?Sized> DiffOperatorIo<IoBuf> for MklProjSqueezeConv2dOperator<S, IoBuf> {
}

impl<S, IoBuf: ?Sized> DiffOperator<S, IoBuf> for MklProjSqueezeConv2dOperator<S, IoBuf> {
  fn _traverse_fwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    self.join_op.borrow_mut()._traverse_fwd(epoch, apply);
    apply(self);
    self.node.pop(epoch);
  }

  fn _traverse_bwd(&mut self, epoch: u64, apply: &mut FnMut(&mut DiffOperator<S, IoBuf>)) {
    self.node.push(epoch);
    assert!(self.node.limit(1));
    apply(self);
    self.join_op.borrow_mut()._traverse_bwd(epoch, apply);
    self.node.pop(epoch);
  }

  fn _forward(&mut self, phase: OpPhase) {
  }

  fn _backward(&mut self) {
  }
}
