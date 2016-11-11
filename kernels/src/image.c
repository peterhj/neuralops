#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void neuralops_image_crop(
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
  size_t p = 0;
  for (size_t a = 0; a < chan; a++) {
    for (size_t v = 0; v < crop_h; v++) {
      for (size_t u = 0; u < crop_w; u++) {
        ptrdiff_t x = offset_x + u;
        ptrdiff_t y = offset_y + v;
        if (x < 0 || x >= in_width || y < 0 || y >= in_height) {
          out_pixels[p] = 0.0f;
        } else {
          out_pixels[p] = in_pixels[x + in_width * (y + in_height * a)];
        }
        p += 1;
      }
    }
  }
}

void neuralops_image_flip(
    size_t width,
    size_t height,
    size_t chan,
    const float *in_pixels,
    float *out_pixels)
{
  size_t p = 0;
  for (size_t a = 0; a < chan; a++) {
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        out_pixels[p] = in_pixels[(width - x - 1) + width * (y + height * a)];
        p += 1;
      }
    }
  }
}
