#include "lib.h"
#include <float.h>
#include <stdint.h>
#include <stdlib.h>

void neuralops_omp_caffe_avgpool2d_fwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
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
    float *top_data = out_buf + n * out_width * out_height * chan;
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
      bottom_data += in_width * in_height;
      top_data += out_width * out_height;
    }
  }
}

void neuralops_omp_caffe_avgpool2d_bwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
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
    const float *top_diff = out_grad + n * out_width * out_height * chan;
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
      bottom_diff += in_width * in_height;
      top_diff += out_width * out_height;
    }
  }
}

void neuralops_omp_caffe_maxpool2d_fwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    const float *in_buf,
    size_t out_width,
    size_t out_height,
    uint32_t *mask_buf,
    float *out_buf,
    size_t kernel_w_,
    size_t kernel_h_,
    size_t stride_w_,
    size_t stride_h_,
    size_t pad_w_,
    size_t pad_h_)
{
  // Initialize
  size_t top_count = out_width * out_height * chan * batch_sz;
  #pragma omp parallel for
  for (size_t p = 0; p < top_count; p++) {
    mask_buf[p] = 0xffffffff;
  }
  #pragma omp parallel for
  for (size_t p = 0; p < top_count; p++) {
    out_buf[p] = (float)(-FLT_MAX);
  }
  // The main loop
  #pragma omp parallel for
  for (size_t n = 0; n < batch_sz; ++n) {
    const float *bottom_data = in_buf + n * in_width * in_height * chan;
    uint32_t *top_mask = mask_buf + n * out_width * out_height * chan;
    float *top_data = out_buf + n * out_width * out_height * chan;
    for (size_t c = 0; c < chan; ++c) {
      for (size_t ph = 0; ph < out_height; ++ph) {
        for (size_t pw = 0; pw < out_width; ++pw) {
          size_t hstart = ph * stride_h_ - pad_h_;
          size_t wstart = pw * stride_w_ - pad_w_;
          size_t hend = min(hstart + kernel_h_, in_height);
          size_t wend = min(wstart + kernel_w_, in_width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          size_t pool_index = ph * out_width + pw;
          for (size_t h = hstart; h < hend; ++h) {
            for (size_t w = wstart; w < wend; ++w) {
              size_t index = h * in_width + w;
              if (bottom_data[index] > top_data[pool_index]) {
                top_data[pool_index] = bottom_data[index];
                top_mask[pool_index] = (uint32_t)(index);
              }
            }
          }
        }
      }
      // compute offset
      bottom_data += in_width * in_height;
      top_mask += out_width * out_height;
      top_data += out_width * out_height;
    }
  }
}

void neuralops_omp_caffe_maxpool2d_bwd(
    size_t batch_sz,
    size_t in_width,
    size_t in_height,
    size_t chan,
    size_t out_width,
    size_t out_height,
    const uint32_t *mask_buf,
    const float *out_grad,
    float *in_grad,
    size_t kernel_w_,
    size_t kernel_h_,
    size_t stride_w_,
    size_t stride_h_,
    size_t pad_w_,
    size_t pad_h_)
{
  // The main loop
  size_t bottom_count = in_width * in_height * chan * batch_sz;
  #pragma omp parallel for
  for (size_t i = 0; i < bottom_count; i++) {
    in_grad[i] = 0.0f;
  }
  #pragma omp parallel for private(in_grad)
  for (size_t n = 0; n < batch_sz; n++) {
    float *bottom_diff = in_grad + n * in_width * in_height * chan;
    const uint32_t *top_mask = mask_buf + n * out_width * out_height * chan;
    const float *top_diff = out_grad + n * out_width * out_height * chan;
    for (size_t c = 0; c < chan; ++c) {
      for (size_t ph = 0; ph < out_height; ++ph) {
        for (size_t pw = 0; pw < out_width; ++pw) {
          size_t index = ph * out_width + pw;
          uint32_t bottom_index = top_mask[index];
          bottom_diff[bottom_index] += top_diff[index];
        }
      }
      bottom_diff += in_width * in_height;
      top_mask += out_width * out_height;
      top_diff += out_width * out_height;
    }
  }
  /*size_t top_count = out_width * out_height * chan * batch_sz;
  #pragma omp parallel for
  for (size_t p = 0; p < top_count; p++) {
    size_t pw = p % out_width;
    size_t ph = (p / out_width) % out_height;
    size_t c = (p / (out_width * out_height)) % chan;
    size_t n = p / (out_width * out_height * chan);

    size_t index = pw + out_width * (ph + out_height * (c + chan * n));
    uint32_t bottom_index = top_mask[index];
    bottom_diff[bottom_index] += top_diff[index];
  }*/
}
