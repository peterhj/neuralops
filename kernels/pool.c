#include "lib.h"
#include <float.h>
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

void neuralops_caffe_avgpool2d_fwd(
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
  for (size_t i = 0; i < top_count; ++i) {
    out_buf[i] = 0.0f;
  }
  const float *bottom_data = in_buf;
  float *top_data = out_buf;
  for (size_t n = 0; n < batch_sz; ++n) {
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

void neuralops_caffe_avgpool2d_bwd(
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
  for (size_t i = 0; i < bottom_count; ++i) {
    in_grad[i] = 0.0f;
  }
  float *bottom_diff = in_grad;
  const float *top_diff = out_grad;
  for (size_t n = 0; n < batch_sz; ++n) {
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
      bottom_diff += in_width * in_height * chan;
      top_diff += out_width * out_height * chan;
    }
  }
}

void neuralops_caffe_maxpool2d_fwd(
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
  for (size_t p = 0; p < top_count; p++) {
    mask_buf[p] = 0xffffffff;
  }
  for (size_t p = 0; p < top_count; p++) {
    out_buf[p] = (float)(-FLT_MAX);
  }
  // The main loop
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

void neuralops_caffe_maxpool2d_bwd(
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
  for (size_t i = 0; i < bottom_count; i++) {
    in_grad[i] = 0.0f;
  }
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
}
