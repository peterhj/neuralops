#[link(name = "neuralops_extkernels", kind = "static")]
extern "C" {
  pub fn neuralops_conv2d_bias_bwd(
      batch_sz: usize,
      width: usize,
      height: usize,
      chan: usize,
      out_grad: *const f32,
      bias_grad: *mut f32);
}
