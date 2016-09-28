use densearray::{Reshape, ReshapeMut, AsView, AsViewMut, Array1d};

#[link(name = "neuralops_extkernels", kind = "static")]
extern "C" {
  pub fn neuralops_batchnorm2d_fwd_mean(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *mut f32);
  pub fn neuralops_batchnorm2d_fwd_var(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      var: *mut f32);
  pub fn neuralops_batchnorm2d_fwd_output(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      mean_acc: *const f32,
      var: *const f32,
      var_acc: *const f32,
      out_buf: *mut f32,
      gamma: f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_bwd_var(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      mean_acc: *const f32,
      var: *const f32,
      var_acc: *const f32,
      out_grad: *const f32,
      var_grad: *mut f32,
      gamma: f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_bwd_mean(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      mean_acc: *const f32,
      var: *const f32,
      var_acc: *const f32,
      var_grad: *const f32,
      out_grad: *const f32,
      mean_grad: *mut f32,
      gamma: f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_bwd_input(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      mean_acc: *const f32,
      mean_grad: *const f32,
      var: *const f32,
      var_acc: *const f32,
      var_grad: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
      gamma: f32,
      epsilon: f32);
}

pub struct BatchMean2dKernel {
  batch_sz: usize,
  dim:      (usize, usize, usize),
  mean:     Array1d<f32>,
  run_mean: Array1d<f32>,
}

pub struct BatchNorm2dKernel {
  batch_sz: usize,
  dim:      (usize, usize, usize),
  mean:     Array1d<f32>,
  var:      Array1d<f32>,
  run_mean: Array1d<f32>,
  run_var:  Array1d<f32>,
}

impl BatchNorm2dKernel {
  pub fn forward(&mut self, batch_sz: usize, in_buf: &[f32], out_buf: &mut [f32], gamma: f32) {
    assert!(batch_sz <= self.batch_sz);
    unimplemented!();
  }

  pub fn backward(&mut self, batch_sz: usize, in_buf: &[f32], out_grad: &[f32], in_grad: &mut [f32], gamma: f32) {
    assert!(batch_sz <= self.batch_sz);
    unimplemented!();
  }

  pub fn update(&mut self, gamma: f32) {
    let chan = self.dim.2;
    self.run_mean.as_view_mut().vector_scale(1.0 - gamma);
    self.run_mean.as_view_mut().vector_add(gamma, self.mean.as_view());
    self.run_var.as_view_mut().vector_scale(1.0 - gamma);
    self.run_var.as_view_mut().vector_add(gamma, self.var.as_view());
  }
}
