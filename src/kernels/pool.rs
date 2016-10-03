use densearray::{Reshape, ReshapeMut, AsView, AsViewMut, Array1d};

#[link(name = "neuralops_extkernels", kind = "static")]
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
