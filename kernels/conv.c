#include <stdint.h>
#include <stdlib.h>

void neuralops_conv2d_bias_fwd(
    size_t batch_sz,
    size_t out_width,
    size_t out_height,
    size_t out_chan,
    const float *in_buf,
    const float *bias,
    float *out_buf)
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
    const float *out_grad,
    float *bias_grad)
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
    const float *in_buf,
    const float *scale,
    const float *bias,
    float *out_buf)
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
    const float *in_buf,
    const float *scale,
    const float *out_grad,
    float *scale_grad,
    float *bias_grad,
    float *in_grad)
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

void neuralops_caffe_im2col(
    const float *in_buf,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float *col_buf)
{
  const float *data_im = in_buf;
  float *data_col = col_buf;
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          //if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
          if (!(input_row >= 0 && input_row < height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0.0f;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              //if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
              if (input_col >= 0 && input_col < width) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0.0f;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void neuralops_caffe_col2im(
    const float *col_buf,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float *out_buf)
{
  const float *data_col = col_buf;
  float *data_im = out_buf;
  //caffe_set(height * width * channels, float(0), data_im);
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          //if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
          if (!(input_row >= 0 && input_row < height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              //if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
              if (input_col >= 0 && input_col < width) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}
