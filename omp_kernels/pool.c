#include "lib.h"
#include <stdint.h>
#include <stdlib.h>

void neuralops_omp_caffe_avgpool2d_fwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    //size_t width_,
    //size_t height_,
    size_t chan,
    const float *in_buf,
    size_t out_width,
    size_t out_height,
    float *out_buf,
    size_t kernel_w_,
    size_t kernel_h_,
    size_t stride_w_,
    size_t stride_h_,
    size_t pad_w_,
    size_t pad_h_)
{
  size_t top_count = out_width * out_height * chan * batch_sz;
  #pragma omp parallel for
  for (size_t i = 0; i < top_count; ++i) {
    out_buf[i] = 0.0f;
  }
  #pragma omp parallel for
  for (size_t n = 0; n < batch_sz; ++n) {
    const float *bottom_data = in_buf + n * in_width * in_height * chan;
    float *top_data = out_buf + n * in_width * in_height * chan;
    for (size_t c = 0; c < chan; ++c) {
      for (size_t ph = 0; ph < out_height; ++ph) {
        for (size_t pw = 0; pw < out_width; ++pw) {
          size_t hstart = ph * stride_h_ - pad_h_;
          size_t wstart = pw * stride_w_ - pad_w_;
          size_t hend = min(hstart + kernel_h_, in_height + pad_h_);
          size_t wend = min(wstart + kernel_w_, in_width + pad_w_);
          size_t pool_size = (hend - hstart) * (wend - wstart);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, in_height);
          wend = min(wend, in_width);
          for (size_t h = hstart; h < hend; ++h) {
            for (size_t w = wstart; w < wend; ++w) {
              top_data[ph * out_width + pw] +=
                  bottom_data[h * in_width + w];
            }
          }
          top_data[ph * out_width + pw] /= ((float)pool_size);
        }
      }
      bottom_data += in_width * in_height * chan;
      top_data += out_width * out_height * chan;
    }
  }
}

void neuralops_omp_caffe_avgpool2d_bwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    //const float *in_buf,
    size_t out_width,
    size_t out_height,
    const float *out_grad,
    float *in_grad,
    size_t kernel_w_,
    size_t kernel_h_,
    size_t stride_w_,
    size_t stride_h_,
    size_t pad_w_,
    size_t pad_h_)
{
  size_t bottom_count = in_width * in_height * chan * batch_sz;
  #pragma omp parallel for
  for (size_t i = 0; i < bottom_count; ++i) {
    in_grad[i] = 0.0f;
  }
  #pragma omp parallel for
  for (size_t n = 0; n < batch_sz; ++n) {
    float *bottom_diff = in_grad + n * in_width * in_height * chan;
    const float *top_diff = out_grad + n * in_width * in_height * chan;
    for (size_t c = 0; c < chan; ++c) {
      for (size_t ph = 0; ph < out_height; ++ph) {
        for (size_t pw = 0; pw < out_width; ++pw) {
          size_t hstart = ph * stride_h_ - pad_h_;
          size_t wstart = pw * stride_w_ - pad_w_;
          size_t hend = min(hstart + kernel_h_, in_height + pad_h_);
          size_t wend = min(wstart + kernel_w_, in_width + pad_w_);
          size_t pool_size = (hend - hstart) * (wend - wstart);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, in_height);
          wend = min(wend, in_width);
          for (size_t h = hstart; h < hend; ++h) {
            for (size_t w = wstart; w < wend; ++w) {
              bottom_diff[h * in_width + w] +=
                top_diff[ph * out_width + pw] / ((float)pool_size);
            }
          }
        }
      }
    }
  }
}
