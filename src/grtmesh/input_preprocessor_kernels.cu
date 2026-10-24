// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <climits>

#include "grtmesh/input_preprocessor_kernels.cuh"

namespace grtmesh::preprocess_kernels {
namespace {

constexpr int kThreads = 256;

int Blocks(int n) { return (n + kThreads - 1) / kThreads; }

// cv::remap INTER_LINEAR on 8-bit data: weights of the 32x32 sub-pixel table
// in 1/32768 (INTER_REMAP_COEF_BITS = 15), rounded with + 2^14 >> 15.
__global__ void RemapColorKernel(const uint8_t* __restrict__ src,
                                 uint8_t* __restrict__ dst, int width,
                                 int height, const short2* __restrict__ map1,
                                 const uint16_t* __restrict__ map2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height) return;
  uint8_t* d = dst + 3 * i;
  if (map1 == nullptr) {
    d[0] = src[3 * i];
    d[1] = src[3 * i + 1];
    d[2] = src[3 * i + 2];
    return;
  }
  const short2 xy = map1[i];
  const int sx = xy.x, sy = xy.y;
  if (sx >= width || sx + 1 < 0 || sy >= height || sy + 1 < 0) {
    d[0] = d[1] = d[2] = 0;
    return;
  }
  const int a = map2[i] & 1023;
  const int fx = a & 31, fy = a >> 5;
  const int w[4] = {(32 - fx) * (32 - fy) * 32, fx * (32 - fy) * 32,
                    (32 - fx) * fy * 32, fx * fy * 32};
  const int px[4] = {sx, sx + 1, sx, sx + 1};
  const int py[4] = {sy, sy, sy + 1, sy + 1};
  int sum[3] = {0, 0, 0};
  for (int k = 0; k < 4; ++k) {
    if (px[k] < 0 || px[k] >= width || py[k] < 0 || py[k] >= height) continue;
    const uint8_t* s = src + 3 * (py[k] * width + px[k]);
    sum[0] += s[0] * w[k];
    sum[1] += s[1] * w[k];
    sum[2] += s[2] * w[k];
  }
  for (int c = 0; c < 3; ++c) {
    const int v = (sum[c] + (1 << 14)) >> 15;
    d[c] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
  }
}

// cv::remap INTER_NEAREST with CV_16SC2 + CV_16UC1 maps: OpenCV adds
// NNDeltaTab_i (1 when the sub-pixel index is below one half) to map1.
__global__ void RemapDepthKernel(const float* __restrict__ src,
                                 float* __restrict__ dst, int width, int height,
                                 const short2* __restrict__ map1,
                                 const uint16_t* __restrict__ map2,
                                 float far_threshold) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height) return;
  float v;
  if (map1 == nullptr) {
    v = src[i];
  } else {
    const short2 xy = map1[i];
    const int a = map2[i] & 1023;
    const int sx = xy.x + ((a & 31) < 16 ? 1 : 0);
    const int sy = xy.y + ((a >> 5) < 16 ? 1 : 0);
    v = (static_cast<unsigned>(sx) < static_cast<unsigned>(width) &&
         static_cast<unsigned>(sy) < static_cast<unsigned>(height))
            ? src[sy * width + sx]
            : 0.f;
  }
  dst[i] = v > far_threshold ? 0.f : v;
}

__global__ void FillKernel(int* buffer, int n, int value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buffer[i] = value;
}

// Same float operations, in the same order, as ProjectDepthToColor in
// util.cc (no FMA contraction): x = (u - cx) * z / fx, p_c = t + R p_d,
// u_c = round(x_c / z * fx_c * s + cx_c * s). Positive floats order like
// their bit patterns, so atomicMin keeps the nearest surface.
__global__ void ZBufferKernel(const float* __restrict__ depth, int width,
                              int height, DepthToColorParams p, int* zbuffer,
                              int zw, int zh) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height) return;
  const float val = depth[i];
  if (val < 1.17549435e-38f) return;  // FLT_MIN
  const int u = i % width, v = i / width;
  const float dx = __fdiv_rn(
      __fmul_rn(__fsub_rn(static_cast<float>(u), p.depth_cx), val), p.depth_fx);
  const float dy = __fdiv_rn(
      __fmul_rn(__fsub_rn(static_cast<float>(v), p.depth_cy), val), p.depth_fy);
  float c[3];
  for (int r = 0; r < 3; ++r) {
    const float* row = p.rotation + 3 * r;
    const float lin =
        __fadd_rn(__fadd_rn(__fmul_rn(row[0], dx), __fmul_rn(row[1], dy)),
                  __fmul_rn(row[2], val));
    c[r] = __fadd_rn(p.translation[r], lin);
  }
  const float cz = c[2];
  if (!(cz > 0.f)) return;
  const float cx = roundf(
      __fadd_rn(__fmul_rn(__fmul_rn(__fdiv_rn(c[0], val), p.color_fx), p.scale),
                __fmul_rn(p.color_cx, p.scale)));
  const float cy = roundf(
      __fadd_rn(__fmul_rn(__fmul_rn(__fdiv_rn(c[1], val), p.color_fy), p.scale),
                __fmul_rn(p.color_cy, p.scale)));
  if (cy < 0 || zh - 1 < cy || cx < 0 || zw - 1 < cx) return;
  atomicMin(zbuffer + static_cast<int>(cy) * zw + static_cast<int>(cx),
            __float_as_int(cz));
}

__global__ void UpsampleKernel(const int* __restrict__ zbuffer, int zw,
                               const int* __restrict__ x_offsets,
                               const int* __restrict__ y_offsets, float* out,
                               int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height) return;
  const int key = zbuffer[y_offsets[i / width] * zw + x_offsets[i % width]];
  out[i] = key == INT_MAX ? 0.f : __int_as_float(key);
}

}  // namespace

void RemapColor(const uint8_t* src, uint8_t* dst, int width, int height,
                RemapMaps maps, cudaStream_t stream) {
  RemapColorKernel<<<Blocks(width * height), kThreads, 0, stream>>>(
      src, dst, width, height, maps.map1, maps.map2);
}

void RemapDepth(const float* src, float* dst, int width, int height,
                RemapMaps maps, float far_threshold, cudaStream_t stream) {
  RemapDepthKernel<<<Blocks(width * height), kThreads, 0, stream>>>(
      src, dst, width, height, maps.map1, maps.map2, far_threshold);
}

void ProjectDepthToColor(const float* depth, int depth_width, int depth_height,
                         const DepthToColorParams& params, int* zbuffer,
                         int zbuffer_width, int zbuffer_height,
                         const int* x_offsets, const int* y_offsets,
                         float* depth_in_color, int color_width,
                         int color_height, cudaStream_t stream) {
  const int zn = zbuffer_width * zbuffer_height;
  FillKernel<<<Blocks(zn), kThreads, 0, stream>>>(zbuffer, zn, INT_MAX);
  ZBufferKernel<<<Blocks(depth_width * depth_height), kThreads, 0, stream>>>(
      depth, depth_width, depth_height, params, zbuffer, zbuffer_width,
      zbuffer_height);
  UpsampleKernel<<<Blocks(color_width * color_height), kThreads, 0, stream>>>(
      zbuffer, zbuffer_width, x_offsets, y_offsets, depth_in_color, color_width,
      color_height);
}

}  // namespace grtmesh::preprocess_kernels
