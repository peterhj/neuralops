#[link(name = "neuralops_extkernels", kind = "static")]
extern "C" {
  pub fn neuralops_batchnorm2d_mean_fwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *mut f32);
  pub fn neuralops_batchnorm2d_var_fwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      var: *mut f32);
  pub fn neuralops_batchnorm2d_output_fwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      var: *const f32,
      out_buf: *mut f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_var_bwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      var: *const f32,
      out_grad: *const f32,
      var_grad: *mut f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_mean_bwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      var: *const f32,
      var_grad: *const f32,
      out_grad: *const f32,
      mean_grad: *mut f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_input_bwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      in_buf: *const f32,
      mean: *const f32,
      mean_grad: *const f32,
      var: *const f32,
      var_grad: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
      epsilon: f32);
}

pub struct BatchMean2dKernel {
  batch_sz: usize,
  dim:      (usize, usize, usize),
}

pub struct BatchNorm2dKernel {
  batch_sz: usize,
  dim:      (usize, usize, usize),
  mean:     Vec<f32>,
  var:      Vec<f32>,
  run_mean: Vec<f32>,
  run_var:  Vec<f32>,
}

impl BatchNorm2dKernel {
  pub fn forward(&mut self, ) {
  }

  pub fn backward(&mut self, ) {
  }
}
