#include <stdint.h>
#include <stdlib.h>

void neuralops_batchnorm2d_mean_fwd(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    float *restrict mean)
{
  for (size_t a = 0; a < chan; a += 1) {
    mean[a] = 0.0f;
  }
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      for (size_t y = 0; y < height; y += 1) {
        for (size_t x = 0; x < width; x += 1) {
          mean[a] += in_buf[p];
          p += 1;
        }
      }
    }
  }
  for (size_t a = 0; a < chan; a += 1) {
    mean[a] /= (float)(width * height * batch_sz);
  }
}

void neuralops_batchnorm2d_var_fwd(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    float *restrict var)
{
  for (size_t a = 0; a < chan; a += 1) {
    var[a] = 0.0f;
  }
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      float m = mean[a];
      for (size_t y = 0; y < height; y += 1) {
        for (size_t x = 0; x < width; x += 1) {
          float val = in_buf[p];
          var[a] += (val - m) * (val - m);
          p += 1;
        }
      }
    }
  }
  for (size_t a = 0; a < chan; a += 1) {
    var[a] /= (float)(width * height * batch_sz - 1);
  }
}

void neuralops_batchnorm2d_output_fwd(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    const float *restrict var,
    float *restrict out_buf,
    float epsilon)
{
}

void neuralops_batchnorm2d_var_bwd(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    const float *restrict var,
    const float *restrict out_grad,
    float *restrict var_grad,
    float epsilon)
{
}

void neuralops_batchnorm2d_mean_bwd(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    const float *restrict var,
    const float *restrict var_grad,
    const float *restrict out_grad,
    float *restrict mean_grad,
    float epsilon)
{
}

void neuralops_batchnorm2d_input_bwd(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    const float *restrict mean_grad,
    const float *restrict var,
    const float *restrict var_grad,
    const float *restrict out_grad,
    float *restrict in_grad,
    float epsilon)
{
}
