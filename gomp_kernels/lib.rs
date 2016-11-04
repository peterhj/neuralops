extern crate libc;

use libc::*;

#[link(name = "gomp")]
extern "C" {}

#[link(name = "neuralops_gomp_kernels", kind = "static")]
extern "C" {
  pub fn neuralops_gomp_rect_fwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_buf: *mut f32,
  );
  pub fn neuralops_gomp_rect_bwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
  );
  pub fn neuralops_gomp_logistic_fwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_buf: *mut f32,
  );
  pub fn neuralops_gomp_logistic_bwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
  );
}
