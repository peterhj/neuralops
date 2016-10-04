use densearray::{ArrayIndex, Reshape, ReshapeMut, AsView, AsViewMut, Array1d};

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
      run_mean: *const f32,
      var: *const f32,
      run_var: *const f32,
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
      run_mean: *const f32,
      var: *const f32,
      run_var: *const f32,
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
      run_mean: *const f32,
      var: *const f32,
      run_var: *const f32,
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
      run_mean: *const f32,
      mean_grad: *const f32,
      var: *const f32,
      run_var: *const f32,
      var_grad: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
      gamma: f32,
      epsilon: f32);
}

pub struct BatchMean2dKernel {
  pub batch_sz: usize,
  pub dim:      (usize, usize, usize),
  pub mean:     Array1d<f32>,
  pub run_mean: Array1d<f32>,
}

pub struct BatchNorm2dKernel {
  pub batch_sz: usize,
  pub dim:      (usize, usize, usize),
  //pub gamma:    f32,
  pub epsilon:  f32,
  pub mean:     Array1d<f32>,
  pub mean_g:   Array1d<f32>,
  pub var:      Array1d<f32>,
  pub var_g:    Array1d<f32>,
  pub run_mean: Array1d<f32>,
  pub run_var:  Array1d<f32>,
}

impl BatchNorm2dKernel {
  pub fn new(batch_sz: usize, /*gamma: f32,*/ epsilon: f32, dim: (usize, usize, usize)) -> BatchNorm2dKernel {
    let mean = Array1d::zeros(dim.2);
    let mean_g = Array1d::zeros(dim.2);
    let var = Array1d::zeros(dim.2);
    let var_g = Array1d::zeros(dim.2);
    let run_mean = Array1d::zeros(dim.2);
    let mut run_var = Array1d::zeros(dim.2);
    run_var.as_view_mut().set_constant(1.0);
    BatchNorm2dKernel{
      batch_sz: batch_sz,
      dim:      dim,
      //gamma:    gamma,
      epsilon:  epsilon,
      mean:     mean,
      mean_g:   mean_g,
      var:      var,
      var_g:    var_g,
      run_mean: run_mean,
      run_var:  run_var,
    }
  }

  pub fn forward(&mut self, batch_sz: usize, in_buf: &[f32], out_buf: &mut [f32], gamma: f32) {
    assert!(batch_sz <= self.batch_sz);
    //unimplemented!();
    //self.mean.as_view_mut().set_constant(0.0);
    //self.var.as_view_mut().set_constant(0.0);
    assert_eq!(in_buf.len(), self.dim.flat_len() * batch_sz);
    assert_eq!(out_buf.len(), self.dim.flat_len() * batch_sz);
    unsafe { neuralops_batchnorm2d_fwd_mean(
        batch_sz,
        self.dim.0, self.dim.1, self.dim.2,
        in_buf.as_ptr(),
        self.mean.as_view_mut().as_mut_ptr(),
    ) };
    unsafe { neuralops_batchnorm2d_fwd_var(
        batch_sz,
        self.dim.0, self.dim.1, self.dim.2,
        in_buf.as_ptr(),
        self.mean.as_view().as_ptr(),
        self.var.as_view_mut().as_mut_ptr(),
    ) };
    unsafe { neuralops_batchnorm2d_fwd_output(
        batch_sz,
        self.dim.0, self.dim.1, self.dim.2,
        in_buf.as_ptr(),
        self.mean.as_view().as_ptr(),
        self.run_mean.as_view().as_ptr(),
        self.var.as_view().as_ptr(),
        self.run_var.as_view().as_ptr(),
        out_buf.as_mut_ptr(),
        gamma,
        self.epsilon,
    ) };
  }

  pub fn backward(&mut self, batch_sz: usize, in_buf: &[f32], out_grad: &[f32], in_grad: &mut [f32], gamma: f32) {
    assert!(batch_sz <= self.batch_sz);
    //unimplemented!();
    //self.mean_grad.as_view_mut().set_constant(0.0);
    //self.var_grad.as_view_mut().set_constant(0.0);
    assert_eq!(in_buf.len(), self.dim.flat_len() * batch_sz);
    assert_eq!(out_grad.len(), self.dim.flat_len() * batch_sz);
    assert_eq!(in_grad.len(), self.dim.flat_len() * batch_sz);
    unsafe { neuralops_batchnorm2d_bwd_var(
        batch_sz,
        self.dim.0, self.dim.1, self.dim.2,
        in_buf.as_ptr(),
        self.mean.as_view().as_ptr(),
        self.run_mean.as_view().as_ptr(),
        self.var.as_view().as_ptr(),
        self.run_var.as_view().as_ptr(),
        out_grad.as_ptr(),
        self.var_g.as_view_mut().as_mut_ptr(),
        gamma,
        self.epsilon,
    ) };
    unsafe { neuralops_batchnorm2d_bwd_mean(
        batch_sz,
        self.dim.0, self.dim.1, self.dim.2,
        in_buf.as_ptr(),
        self.mean.as_view().as_ptr(),
        self.run_mean.as_view().as_ptr(),
        self.var.as_view().as_ptr(),
        self.run_var.as_view().as_ptr(),
        self.var_g.as_view().as_ptr(),
        out_grad.as_ptr(),
        self.mean_g.as_view_mut().as_mut_ptr(),
        gamma,
        self.epsilon,
    ) };
    unsafe { neuralops_batchnorm2d_bwd_input(
        batch_sz,
        self.dim.0, self.dim.1, self.dim.2,
        in_buf.as_ptr(),
        self.mean.as_view().as_ptr(),
        self.run_mean.as_view().as_ptr(),
        self.mean_g.as_view().as_ptr(),
        self.var.as_view().as_ptr(),
        self.run_var.as_view().as_ptr(),
        self.var_g.as_view().as_ptr(),
        out_grad.as_ptr(),
        in_grad.as_mut_ptr(),
        gamma,
        self.epsilon,
    ) };
  }

  pub fn update(&mut self, gamma: f32) {
    self.run_mean.as_view_mut().vector_scale(1.0 - gamma);
    self.run_mean.as_view_mut().vector_add(gamma, self.mean.as_view());
    self.run_var.as_view_mut().vector_scale(1.0 - gamma);
    self.run_var.as_view_mut().vector_add(gamma, self.var.as_view());
  }
}
