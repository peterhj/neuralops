use densearray::{ArrayIndex, Reshape, ReshapeMut, AsView, AsViewMut, Array1d};

use libc::*;

#[link(name = "neuralops_extkernels", kind = "static")]
extern "C" {
  pub fn neuralops_conv2d_bias_fwd(
      batch_sz: usize,
      out_width: usize,
      out_height: usize,
      out_chan: usize,
      in_buf: *const f32,
      bias: *const f32,
      out_buf: *mut f32);
  pub fn neuralops_conv2d_bias_bwd(
      batch_sz: usize,
      out_width: usize,
      out_height: usize,
      out_chan: usize,
      out_grad: *const f32,
      bias_grad: *mut f32);
      //in_grad: *mut f32);
  pub fn neuralops_conv2d_scale_bias_fwd(
      batch_sz: usize,
      out_width: usize,
      out_height: usize,
      out_chan: usize,
      in_buf: *const f32,
      scale: *const f32,
      bias: *const f32,
      out_buf: *mut f32);
  pub fn neuralops_conv2d_scale_bias_bwd(
      batch_sz: usize,
      out_width: usize,
      out_height: usize,
      out_chan: usize,
      in_buf: *const f32,
      scale: *const f32,
      out_grad: *const f32,
      scale_grad: *mut f32,
      bias_grad: *mut f32,
      in_grad: *mut f32);
  pub fn neuralops_caffe_im2col(
      data_im: *const f32,
      channels: c_int, height: c_int, width: c_int,
      kernel_h: c_int, kernel_w: c_int,
      pad_h: c_int, pad_w: c_int,
      stride_h: c_int, stride_w: c_int,
      dilation_h: c_int, dilation_w: c_int,
      data_col: *mut f32);
  pub fn neuralops_caffe_col2im(
      data_col: *const f32,
      channels: c_int, height: c_int, width: c_int,
      kernel_h: c_int, kernel_w: c_int,
      pad_h: c_int, pad_w: c_int,
      stride_h: c_int, stride_w: c_int,
      dilation_h: c_int, dilation_w: c_int,
      data_im: *mut f32);
}

pub struct ConvScale2dKernel {
  pub batch_sz:     usize,
  pub dim:          (usize, usize, usize),
  pub scale:        Array1d<f32>,
  pub scale_grad:   Array1d<f32>,
  pub bias:         Array1d<f32>,
  pub bias_grad:    Array1d<f32>,
}

impl ConvScale2dKernel {
  pub fn new(batch_sz: usize, dim: (usize, usize, usize)) -> ConvScale2dKernel {
    let mut scale = Array1d::zeros(dim.2);
    scale.as_view_mut().set_constant(1.0);
    let bias = Array1d::zeros(dim.2);
    let scale_grad = Array1d::zeros(dim.2);
    let bias_grad = Array1d::zeros(dim.2);
    ConvScale2dKernel{
      batch_sz:     batch_sz,
      dim:          dim,
      scale:        scale,
      scale_grad:   scale_grad,
      bias:         bias,
      bias_grad:    bias_grad,
    }
  }

  pub fn forward(&mut self, batch_sz: usize, in_buf: &[f32], out_buf: &mut [f32]) {
    assert!(batch_sz <= self.batch_sz);
    unsafe { neuralops_conv2d_scale_bias_fwd(
        batch_sz,
        self.dim.0,
        self.dim.1,
        self.dim.2,
        in_buf.as_ptr(),
        self.scale.as_view().as_ptr(),
        self.bias.as_view().as_ptr(),
        out_buf.as_mut_ptr(),
    ) };
  }

  pub fn backward(&mut self, batch_sz: usize, in_buf: &[f32], out_grad: &[f32], in_grad: &mut [f32]) {
    assert!(batch_sz <= self.batch_sz);
    unsafe { neuralops_conv2d_scale_bias_bwd(
        batch_sz,
        self.dim.0,
        self.dim.1,
        self.dim.2,
        in_buf.as_ptr(),
        self.scale.as_view().as_ptr(),
        out_grad.as_ptr(),
        self.scale_grad.as_view_mut().as_mut_ptr(),
        self.bias_grad.as_view_mut().as_mut_ptr(),
        in_grad.as_mut_ptr(),
    ) };
  }
}
