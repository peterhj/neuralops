use libc::*;

#[link(name = "neuralops_extkernels", kind = "static")]
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
