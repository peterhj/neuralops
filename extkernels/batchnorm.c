#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void neuralops_batchnorm2d_fwd_mean(
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

void neuralops_batchnorm2d_fwd_var(
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

void neuralops_batchnorm2d_fwd_output(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    const float *restrict run_mean,
    const float *restrict var,
    const float *restrict run_var,
    float *restrict out_buf,
    float gamma,
    float epsilon)
{
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      /*float m = run_mean[a] * (1.0f - gamma) + mean[a] * gamma;
      float v = run_var[a] * (1.0f - gamma) + var[a] * gamma;*/
      float m = mean[a];
      float v = var[a];
      // FIXME(20160928): try rsqrtps intrinsic here.
      float rs = 1.0f / sqrtf(v + epsilon);
      for (size_t y = 0; y < height; y += 1) {
        for (size_t x = 0; x < width; x += 1) {
          out_buf[p] = (in_buf[p] - m) * rs;
          p += 1;
        }
      }
    }
  }
}

void neuralops_batchnorm2d_bwd_var(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    const float *restrict run_mean,
    const float *restrict var,
    const float *restrict run_var,
    const float *restrict out_grad,
    float *restrict var_grad,
    float gamma,
    float epsilon)
{
  for (size_t a = 0; a < chan; a += 1) {
    var_grad[a] = 0.0f;
  }
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      /*float m = run_mean[a] * (1.0f - gamma) + mean[a] * gamma;
      float v = run_var[a] * (1.0f - gamma) + var[a] * gamma;*/
      float m = mean[a];
      float v = var[a];
      float rs = -0.5f / ((v + epsilon) * sqrtf(v + epsilon));
      for (size_t y = 0; y < height; y += 1) {
        for (size_t x = 0; x < width; x += 1) {
          var_grad[a] += out_grad[p] * rs * (in_buf[p] - m);
          p += 1;
        }
      }
    }
  }
  /*for (size_t a = 0; a < chan; a += 1) {
    var_grad[a] *= -0.5f * gamma;
  }*/
}

void neuralops_batchnorm2d_bwd_mean(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    const float *restrict run_mean,
    const float *restrict var,
    const float *restrict run_var,
    const float *restrict var_grad,
    const float *restrict out_grad,
    float *restrict mean_grad,
    float gamma,
    float epsilon)
{
  for (size_t a = 0; a < chan; a += 1) {
    mean_grad[a] = 0.0f;
  }
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      /*float m = run_mean[a] * (1.0f - gamma) + mean[a] * gamma;
      float v = run_var[a] * (1.0f - gamma) + var[a] * gamma;*/
      float m = mean[a];
      float v = var[a];
      float rs = -1.0f / sqrtf(v + epsilon);
      float dv = var_grad[a];
      float c = -2.0f / ((float)(width * height * batch_sz - 1));
      for (size_t y = 0; y < height; y += 1) {
        for (size_t x = 0; x < width; x += 1) {
          mean_grad[a] += out_grad[p] * rs + dv * c * (in_buf[p] - m);
          p += 1;
        }
      }
    }
  }
  /*for (size_t a = 0; a < chan; a += 1) {
    mean_grad[a] *= -gamma;
  }*/
}

void neuralops_batchnorm2d_bwd_input(
    size_t batch_sz,
    size_t width,
    size_t height,
    size_t chan,
    const float *restrict in_buf,
    const float *restrict mean,
    const float *restrict run_mean,
    const float *restrict mean_grad,
    const float *restrict var,
    const float *restrict run_var,
    const float *restrict var_grad,
    const float *restrict out_grad,
    float *restrict in_grad,
    float gamma,
    float epsilon)
{
  size_t p = 0;
  for (size_t idx = 0; idx < batch_sz; idx += 1) {
    for (size_t a = 0; a < chan; a += 1) {
      /*float m = run_mean[a] * (1.0f - gamma) + mean[a] * gamma;
      float v = run_var[a] * (1.0f - gamma) + var[a] * gamma;*/
      float m = mean[a];
      float v = var[a];
      float dm = mean_grad[a];
      float cm = 1.0f / ((float)(width * height * batch_sz));
      float rs = 1.0f / sqrtf(v + epsilon);
      float dv = var_grad[a];
      float cv = 2.0f / ((float)(width * height * batch_sz - 1));
      for (size_t y = 0; y < height; y += 1) {
        for (size_t x = 0; x < width; x += 1) {
          in_grad[p] = out_grad[p] * rs + dm * cm + dv * cv * (in_buf[p] - m);
          p += 1;
        }
      }
    }
  }
}
