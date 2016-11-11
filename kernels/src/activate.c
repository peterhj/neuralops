#include "lib.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void NEURALOPS_SYMBOL(rect_fwd)(
    size_t batch_sz,
    size_t dim,
    const float *in_buf,
    float *out_buf)
{
  #pragma omp parallel for
  for (size_t p = 0; p < batch_sz * dim; p += 1) {
    float x = in_buf[p];
    out_buf[p] = x * (x > 0.0f);
  }
}

void NEURALOPS_SYMBOL(rect_bwd)(
    size_t batch_sz,
    size_t dim,
    const float *out_buf,
    const float *out_grad,
    float *in_grad)
{
  #pragma omp parallel for
  for (size_t p = 0; p < batch_sz * dim; p += 1) {
    float y = out_buf[p];
    float dy = out_grad[p];
    in_grad[p] = dy * (y > 0.0f);
  }
}

void NEURALOPS_SYMBOL(logistic_fwd)(
    size_t batch_sz,
    size_t dim,
    const float *in_buf,
    float *out_buf)
{
  #pragma omp parallel for
  for (size_t p = 0; p < batch_sz * dim; p += 1) {
    float x = in_buf[p];
    out_buf[p] = 1.0f / (1.0f + expf(-x));
  }
}

void NEURALOPS_SYMBOL(logistic_bwd)(
    size_t batch_sz,
    size_t dim,
    const float *out_buf,
    const float *out_grad,
    float *in_grad)
{
  #pragma omp parallel for
  for (size_t p = 0; p < batch_sz * dim; p += 1) {
    float y = out_buf[p];
    float dy = out_grad[p];
    in_grad[p] = y * (1.0f - y) * dy;
  }
}
