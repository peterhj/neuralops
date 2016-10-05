#include <stdint.h>
#include <stdlib.h>

void neuralops_rect_fwd(
    size_t batch_sz,
    size_t dim,
    const float *in_buf,
    float *out_buf)
{
  for (size_t p = 0; p < batch_sz * dim; p += 1) {
    float x = in_buf[p];
    if (x > 0.0f) {
      out_buf[p] = x;
    } else {
      out_buf[p] = 0.0f;
    }
  }
}

void neuralops_rect_bwd(
    size_t batch_sz,
    size_t dim,
    const float *in_buf,
    const float *out_grad,
    float *in_grad)
{
  for (size_t p = 0; p < batch_sz * dim; p += 1) {
    float x = in_buf[p];
    if (x > 0.0f) {
      in_grad[p] = out_grad[p];
    } else {
      in_grad[p] = 0.0f;
    }
  }
}
