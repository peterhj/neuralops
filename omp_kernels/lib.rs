extern crate libc;

use libc::*;

#[cfg(not(feature = "iomp"))]
#[link(name = "gomp")]
extern "C" {}

#[cfg(feature = "iomp")]
#[link(name = "iomp")]
extern "C" {}

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
