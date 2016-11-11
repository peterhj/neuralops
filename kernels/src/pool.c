#include <stdint.h>
#include <stdlib.h>

void neuralops_maxpool2d_2x2_fwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *in_buf,
    float *out_buf)
{
  // FIXME(20161005): unimplemented.
}

void neuralops_maxpool2d_2x2_bwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *in_buf,
    const float *out_grad,
    float *in_grad)
{
  // FIXME(20161005): unimplemented.
}

void neuralops_avgpool2d_2x2_fwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *in_buf,
    float *out_buf)
{
  // XXX(20161002): fast case when in_dims are even, for simplicity.
  size_t out_width = (in_width + 1) / 2;
  size_t out_height = (in_height + 1) / 2;
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      for (size_t y = 0; y < out_height; y += 1) {
        for (size_t x = 0; x < out_width; x += 1) {
          float pool_value =
              0.25f * (
                in_buf[x*2   + in_width * (y*2   + in_height * (a + chan * idx))]
              + in_buf[x*2+1 + in_width * (y*2   + in_height * (a + chan * idx))]
              + in_buf[x*2   + in_width * (y*2+1 + in_height * (a + chan * idx))]
              + in_buf[x*2+1 + in_width * (y*2+1 + in_height * (a + chan * idx))]);
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
    const float *in_buf,
    const float *out_grad,
    float *in_grad)
{
  // XXX(20161002): fast case when in_dims are even, for simplicity.
  size_t out_width = (in_width + 1) / 2;
  size_t out_height = (in_height + 1) / 2;
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      for (size_t y = 0; y < in_height; y += 1) {
        for (size_t x = 0; x < in_width; x += 1) {
          in_grad[p] = 0.25f * out_grad[x/2 + out_width * (y/2 + out_height * (a + chan * idx))];
          p += 1;
        }
      }
    }
  }
}

void neuralops_avgpool2d_fwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *in_buf,
    float *out_buf,
    size_t pool_w,
    size_t pool_h)
{
  size_t out_width = (in_width + pool_w - 1) / pool_w;
  size_t out_height = (in_height + pool_h - 1) / pool_h;
  float normalizer = 1.0f / ((float)(pool_w * pool_h));
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      for (size_t y = 0; y < out_height; y += 1) {
        for (size_t x = 0; x < out_width; x += 1) {
          float pool_value = 0.0f;
          for (size_t v = 0; v < pool_h; v += 1) {
            for (size_t u = 0; u < pool_w; u += 1) {
              pool_value +=
                  in_buf[x*pool_w + u + in_width * (y*pool_h + v + in_height * (a + chan * idx))];
            }
          }
          out_buf[p] = normalizer * pool_value;
          p += 1;
        }
      }
    }
  }
}

void neuralops_avgpool2d_bwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *in_buf,
    const float *out_grad,
    float *in_grad,
    size_t pool_w,
    size_t pool_h)
{
  size_t out_width = (in_width + pool_w - 1) / pool_w;
  size_t out_height = (in_height + pool_h - 1) / pool_h;
  float normalizer = 1.0f / ((float)(pool_w * pool_h));
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      for (size_t y = 0; y < in_height; y += 1) {
        for (size_t x = 0; x < in_width; x += 1) {
          in_grad[p] = normalizer * out_grad[x/pool_w + out_width * (y/pool_h + out_height * (a + chan * idx))];
          p += 1;
        }
      }
    }
  }
}
