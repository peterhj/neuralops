use libc::*;

#[link(name = "neuralops_extkernels", kind = "static")]
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
