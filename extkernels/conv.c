#include <stdint.h>
#include <stdlib.h>

void neuralops_conv2d_bias_fwd(
    size_t batch_sz,
    size_t out_width,
    size_t out_height,
    size_t out_chan,
    const float *restrict in_buf,
    const float *restrict bias,
    float *restrict out_buf)
{
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < out_chan; a += 1) {
      float b = bias[a];
      for (size_t y = 0; y < out_height; y += 1) {
        for (size_t x = 0; x < out_width; x += 1) {
          out_buf[p] = in_buf[p] + b;
          p += 1;
        }
      }
    }
  }
}

void neuralops_conv2d_bias_bwd(
    size_t batch_sz,
    size_t out_width,
    size_t out_height,
    size_t out_chan,
    const float *restrict out_grad,
    float *restrict bias_grad)
{
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < out_chan; a += 1) {
      for (size_t y = 0; y < out_height; y += 1) {
        for (size_t x = 0; x < out_width; x += 1) {
          float g = out_grad[p];
          bias_grad[a] += g;
          p += 1;
        }
      }
    }
  }
}

void neuralops_conv2d_scale_bias_fwd(
    size_t batch_sz,
    size_t out_width,
    size_t out_height,
    size_t out_chan,
    const float *restrict in_buf,
    const float *restrict scale,
    const float *restrict bias,
    float *restrict out_buf)
{
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < out_chan; a += 1) {
      float s = scale[a];
      float b = bias[a];
      for (size_t y = 0; y < out_height; y += 1) {
        for (size_t x = 0; x < out_width; x += 1) {
          out_buf[p] = in_buf[p] * s + b;
          p += 1;
        }
      }
    }
  }
}

void neuralops_conv2d_scale_bias_bwd(
    size_t batch_sz,
    size_t out_width,
    size_t out_height,
    size_t out_chan,
    const float *restrict in_buf,
    const float *restrict scale,
    const float *restrict out_grad,
    float *restrict scale_grad,
    float *restrict bias_grad,
    float *restrict in_grad)
{
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < out_chan; a += 1) {
      float s = scale[a];
      for (size_t y = 0; y < out_height; y += 1) {
        for (size_t x = 0; x < out_width; x += 1) {
          float g = out_grad[p];
          scale_grad[a] += g * in_buf[p];
          bias_grad[a] += g;
          in_grad[p] = g * s;
          p += 1;
        }
      }
    }
  }
}
