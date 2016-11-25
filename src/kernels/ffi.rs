//pub use neuralops_kernels::*;
//pub use neuralops_omp_kernels::*;

use libc::*;

#[cfg(not(feature = "iomp"))]
#[link(name = "gomp")]
extern "C" {}

#[cfg(feature = "iomp")]
#[link(name = "iomp5")]
extern "C" {}

#[link(name = "neuralops_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_rect_fwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_buf: *mut f32,
  );
  pub fn neuralops_rect_bwd(
      batch_sz: size_t,
      dim: size_t,
      out_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
  );
  pub fn neuralops_logistic_fwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_buf: *mut f32,
  );
  pub fn neuralops_logistic_bwd(
      batch_sz: size_t,
      dim: size_t,
      out_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
  );
}

#[link(name = "neuralops_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_batchnorm2d_fwd_mean(
      batch_sz: size_t,
      width: size_t,
      height: size_t,
      chan: size_t,
      in_buf: *const f32,
      mean: *mut f32);
  pub fn neuralops_batchnorm2d_fwd_var(
      batch_sz: size_t,
      width: size_t,
      height: size_t,
      chan: size_t,
      in_buf: *const f32,
      mean: *const f32,
      var: *mut f32);
  pub fn neuralops_batchnorm2d_fwd_output(
      batch_sz: size_t,
      width: size_t,
      height: size_t,
      chan: size_t,
      in_buf: *const f32,
      mean: *const f32,
      //run_mean: *const f32,
      var: *const f32,
      //run_var: *const f32,
      out_buf: *mut f32,
      //gamma: f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_bwd_var(
      batch_sz: size_t,
      width: size_t,
      height: size_t,
      chan: size_t,
      in_buf: *const f32,
      mean: *const f32,
      //run_mean: *const f32,
      var: *const f32,
      //run_var: *const f32,
      out_grad: *const f32,
      var_grad: *mut f32,
      //gamma: f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_bwd_mean(
      batch_sz: size_t,
      width: size_t,
      height: size_t,
      chan: size_t,
      in_buf: *const f32,
      mean: *const f32,
      //run_mean: *const f32,
      var: *const f32,
      //run_var: *const f32,
      var_grad: *const f32,
      out_grad: *const f32,
      mean_grad: *mut f32,
      //gamma: f32,
      epsilon: f32);
  pub fn neuralops_batchnorm2d_bwd_input(
      batch_sz: size_t,
      width: size_t,
      height: size_t,
      chan: size_t,
      in_buf: *const f32,
      mean: *const f32,
      //run_mean: *const f32,
      mean_grad: *const f32,
      var: *const f32,
      //run_var: *const f32,
      var_grad: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
      //gamma: f32,
      epsilon: f32);
}

#[link(name = "neuralops_kernels", kind = "static")]
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

#[link(name = "neuralops_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_image_crop(
      in_width: size_t,
      in_height: size_t,
      chan: size_t,
      crop_w: size_t,
      crop_h: size_t,
      offset_x: ptrdiff_t,
      offset_y: ptrdiff_t,
      in_pixels: *const f32,
      out_pixels: *mut f32,
  );
  pub fn neuralops_image_flip(
      width: size_t,
      height: size_t,
      chan: size_t,
      in_pixels: *const f32,
      out_pixels: *mut f32,
  );
}

#[link(name = "neuralops_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_interpolate2d_catmullrom(
      in_width: size_t,
      in_height: size_t,
      chan: size_t,
      out_width: size_t,
      out_height: size_t,
      in_pixels: *const f32,
      out_pixels: *mut f32,
  );
}

#[link(name = "neuralops_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_avgpool2d_2x2_fwd(
      batch_sz: usize,
      in_width: usize,
      in_height: usize,
      chan: usize,
      in_buf: *const f32,
      out_buf: *mut f32);
  pub fn neuralops_avgpool2d_2x2_bwd(
      batch_sz: usize,
      in_width: usize,
      in_height: usize,
      chan: usize,
      in_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32);
  pub fn neuralops_avgpool2d_fwd(
      batch_sz: usize,
      in_width: usize,
      in_height: usize,
      chan: usize,
      in_buf: *const f32,
      out_buf: *mut f32,
      pool_w: usize,
      pool_h: usize);
  pub fn neuralops_avgpool2d_bwd(
      batch_sz: usize,
      in_width: usize,
      in_height: usize,
      chan: usize,
      in_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
      pool_w: usize,
      pool_h: usize);
}

#[link(name = "neuralops_omp_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_omp_rect_fwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_buf: *mut f32,
  );
  pub fn neuralops_omp_rect_bwd(
      batch_sz: size_t,
      dim: size_t,
      out_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
  );
  pub fn neuralops_omp_logistic_fwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_buf: *mut f32,
  );
  pub fn neuralops_omp_logistic_bwd(
      batch_sz: size_t,
      dim: size_t,
      out_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
  );
}
