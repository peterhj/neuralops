#include <stdint.h>
#include <stdlib.h>

void neuralops_conv2d_bias_bwd(
    size_t batch_sz,
    size_t out_width,
    size_t out_height,
    size_t out_chan,
    const float *restrict out_grad,
    float *bias_grad)
{
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < out_chan; a += 1) {
      for (size_t y = 0; y < out_height; y += 1) {
        for (size_t x = 0; x < out_width; x += 1) {
          bias_grad[a] += out_grad[p];
          p += 1;
        }
      }
    }
  }
}
