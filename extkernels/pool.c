#include <stdint.h>
#include <stdlib.h>

void neuralops_avgpool2d_2x2_fwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *restrict in_buf,
    float *restrict out_buf)
{
  // FIXME(20161002): fast case when in_dims are even, for simplicity.
  size_t out_width = (in_width + 1) / 2;
  size_t out_height = (in_height + 1) / 2;
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      for (size_t y = 0; y < out_height; y += 1) {
        for (size_t x = 0; x < out_width; x += 1) {
          float pool_value =
                in_buf[x*2   + in_width * (y*2   + in_height * (a + chan * idx))]
              + in_buf[x*2+1 + in_width * (y*2   + in_height * (a + chan * idx))]
              + in_buf[x*2   + in_width * (y*2+1 + in_height * (a + chan * idx))]
              + in_buf[x*2+1 + in_width * (y*2+1 + in_height * (a + chan * idx))];
          out_buf[p] = pool_value;
          p += 1;
        }
      }
    }
  }
}

void neuralops_avgpool2d_2x2_bwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict out_grad,
    float *restrict in_grad)
{
  // FIXME(20161002): fast case when in_dims are even, for simplicity.
  size_t out_width = (in_width + 1) / 2;
  size_t out_height = (in_height + 1) / 2;
  size_t p = 0;
  // FIXME: unimplemented.
}

void neuralops_avgpool2d_fwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *restrict in_buf,
    float *restrict out_buf,
    size_t pool_w,
    size_t pool_h)
{
  // FIXME: unimplemented.
}
