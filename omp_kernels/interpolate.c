#define _GNU_SOURCE

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#undef max
#undef min
#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); ((_a) > (_b) ? (_a) : (_b)); })
#define min(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); ((_a) < (_b) ? (_a) : (_b)); })

static float clamp2d(
    const float *pixels,
    size_t width, size_t height,
    size_t x, size_t y, size_t a)
{
  size_t clamp_x = min(max(0, x), width-1);
  size_t clamp_y = min(max(0, y), height-1);
  return pixels[clamp_x + width * (clamp_y + height * a)];
}

static float catmullrom_w0(float x) {
  //return -0.5f*a + a*a - 0.5f*a*a*a;
  return x*(-0.5f + x*(1.0f - 0.5f*x));
}

static float catmullrom_w1(float x) {
  //return 1.0f - 2.5f*x*x + 1.5f*x*x*x;
  return 1.0f + x*x*(-2.5f + 1.5f*x);
}

static float catmullrom_w2(float x) {
  //return 0.5f*x + 2.0f*x*x - 1.5f*x*x*x;
  return x*(0.5f + x*(2.0f - 1.5f*x));
}

static float catmullrom_w3(float x) {
  //return -0.5f*x*x + 0.5f*x*x*x;
  return x*x*(-0.5f + 0.5f*x);
}

static float catmullrom_filter(
    float x,
    float c0,
    float c1,
    float c2,
    float c3)
{
  float r = c0 * catmullrom_w0(x);
  r += c1 * catmullrom_w1(x);
  r += c2 * catmullrom_w2(x);
  r += c3 * catmullrom_w3(x);
  return r;
}

static float catmullrom_interp(
    const float *pixels,
    size_t width, size_t height,
    float u, float v, size_t a)
{
  u -= 0.5f;
  v -= 0.5f;
  float px = floorf(u);
  float py = floorf(v);
  float fx = u - px;
  float fy = v - py;
  size_t ipx = (size_t)px;
  size_t ipy = (size_t)py;
  return catmullrom_filter(fy,
      catmullrom_filter(fx,
          clamp2d(pixels, width, height, ipx-1, ipy-1, a),
          clamp2d(pixels, width, height, ipx,   ipy-1, a),
          clamp2d(pixels, width, height, ipx+1, ipy-1, a),
          clamp2d(pixels, width, height, ipx+2, ipy-1, a)),
      catmullrom_filter(fx,
          clamp2d(pixels, width, height, ipx-1, ipy,   a),
          clamp2d(pixels, width, height, ipx,   ipy,   a),
          clamp2d(pixels, width, height, ipx+1, ipy,   a),
          clamp2d(pixels, width, height, ipx+2, ipy,   a)),
      catmullrom_filter(fx,
          clamp2d(pixels, width, height, ipx-1, ipy+1, a),
          clamp2d(pixels, width, height, ipx,   ipy+1, a),
          clamp2d(pixels, width, height, ipx+1, ipy+1, a),
          clamp2d(pixels, width, height, ipx+2, ipy+1, a)),
      catmullrom_filter(fx,
          clamp2d(pixels, width, height, ipx-1, ipy+2, a),
          clamp2d(pixels, width, height, ipx,   ipy+2, a),
          clamp2d(pixels, width, height, ipx+1, ipy+2, a),
          clamp2d(pixels, width, height, ipx+2, ipy+2, a)));
}

void neuralops_omp_interpolate2d_catmullrom(
    size_t in_width,
    size_t in_height,
    size_t chan,
    size_t out_width,
    size_t out_height,
    const float *in_pixels,
    float *out_pixels)
{
  float scale_x = ((float)in_width) / ((float)(out_width));
  float scale_y = ((float)in_height) / ((float)(out_height));
  size_t p_limit = out_width * out_height * chan;
  #pragma omp parallel for
  for (size_t p = 0; p < p_limit; p++) {
    size_t x = p % out_width;
    size_t y = (p / out_width) % out_height;
    size_t a = p / (out_width * out_height);
    float u = (float)x * scale_x;
    float v = (float)y * scale_y;
    /*float u = ((float)x + 0.5f) * scale_x - 0.5f;
    float v = ((float)y + 0.5f) * scale_y - 0.5f;*/
    out_pixels[p] = catmullrom_interp(in_pixels, in_width, in_height, u, v, a);
  }
}
