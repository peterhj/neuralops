#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void neuralops_omp_image_crop(
    size_t in_width,
    size_t in_height,
    size_t chan,
    size_t crop_w,
    size_t crop_h,
    ptrdiff_t offset_x,
    ptrdiff_t offset_y,
    const float *in_pixels,
    float *out_pixels)
{
  size_t p_limit = crop_w * crop_h * chan;
  #pragma omp parallel for
  for (size_t p = 0; p < p_limit; p++) {
    size_t u = p % crop_w;
    size_t v = (p / crop_w) % crop_h;
    size_t a = p / (crop_w * crop_h);
    ptrdiff_t x = offset_x + u;
    ptrdiff_t y = offset_y + v;
    if (x < 0 || x >= in_width || y < 0 || y >= in_height) {
      out_pixels[p] = 0.0f;
    } else {
      out_pixels[p] = in_pixels[x + in_width * (y + in_height * a)];
    }
  }
}

void neuralops_omp_image_flip(
    size_t width,
    size_t height,
    size_t chan,
    const float *in_pixels,
    float *out_pixels)
{
  size_t p_limit = width * height * chan;
  #pragma omp parallel for
  for (size_t p = 0; p < p_limit; p++) {
    size_t x = p % width;
    size_t y = (p / width) % height;
    size_t a = p / (width * height);
    out_pixels[p] = in_pixels[(width - x - 1) + width * (y + height * a)];
  }
}
