use densearray::{ArrayIndex, Reshape, ReshapeMut, AsView, AsViewMut, Array1d};

#[link(name = "neuralops_extkernels", kind = "static")]
extern "C" {
  pub fn neuralops_conv2d_bias_fwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      bias: *const f32,
      out_buf: *mut f32);
  pub fn neuralops_conv2d_scale_bias_fwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      scale: *const f32,
      bias: *const f32,
      out_buf: *mut f32);
  pub fn neuralops_conv2d_bias_bwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      out_grad: *const f32,
      bias_grad: *mut f32);
      //in_grad: *mut f32);
  pub fn neuralops_conv2d_scale_bias_bwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      scale: *const f32,
      out_grad: *const f32,
      scale_grad: *mut f32,
      bias_grad: *mut f32,
      in_grad: *mut f32);
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
    let mut scale = Array1d::zeros(dim.flat_len());
    scale.as_view_mut().set_constant(1.0);
    let bias = Array1d::zeros(dim.flat_len());
    let scale_grad = Array1d::zeros(dim.flat_len());
    let bias_grad = Array1d::zeros(dim.flat_len());
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
