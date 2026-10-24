// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "common/device_buffer.h"
#include "grtmesh/mesh_topology.cuh"
#include "grtmesh/tex/atlas_tiling.h"
#include "grtmesh/tex/color_correction_internal.h"

namespace grtmesh::tex {
namespace {

__device__ __forceinline__ float SrgbToLinearDev(unsigned char v) {
  float c = v * (1.f / 255.f);
  return c <= 0.04045f ? c * (1.f / 12.92f)
                       : __powf((c + 0.055f) * (1.f / 1.055f), 2.4f);
}
__device__ __forceinline__ unsigned char LinearToSrgbDev(float c) {
  c = fminf(1.f, fmaxf(0.f, c));
  float s =
      c <= 0.0031308f ? c * 12.92f : 1.055f * __powf(c, 1.f / 2.4f) - 0.055f;
  int i = static_cast<int>((s * 255.f + 0.5f));
  return static_cast<unsigned char>((i < 0 ? 0 : (i > 255 ? 255 : i)));
}

// Bilinear sample of camera `cam`'s grid_w x grid_h field at source pixel
// (x,y).
__device__ void SampleField(const float* field, int cam, int image_width,
                            int image_height, int grid_width, int grid_height,
                            int nch, float x, float y, float* out) {
  float gx = x * grid_width / image_width - 0.5f,
        gy = y * grid_height / image_height - 0.5f;
  int i0 = static_cast<int>(floorf(gx)), j0 = static_cast<int>(floorf(gy));
  float fx = gx - i0, fy = gy - j0;
  int i1 = i0 + 1, j1 = j0 + 1;
  i0 = min(grid_width - 1, max(0, i0));
  i1 = min(grid_width - 1, max(0, i1));
  j0 = min(grid_height - 1, max(0, j0));
  j1 = min(grid_height - 1, max(0, j1));
  const float* g =
      field + static_cast<size_t>(cam) * grid_width * grid_height * nch;
  for (int c = 0; c < nch; ++c) {
    float v00 = g[(j0 * grid_width + i0) * nch + c];
    float v10 = g[(j0 * grid_width + i1) * nch + c];
    float v01 = g[(j1 * grid_width + i0) * nch + c];
    float v11 = g[(j1 * grid_width + i1) * nch + c];
    out[c] = (v00 * (1 - fx) + v10 * fx) * (1 - fy) +
             (v01 * (1 - fx) + v11 * fx) * fy;
  }
}

// One thread per atlas pixel.
__global__ void BakeFieldKernel(unsigned char* atlas, AtlasTiling tiling,
                                const float* field, int grid_width,
                                int grid_height, int nch, int rgb,
                                float max_log_gain) {
  int ax = blockIdx.x * blockDim.x + threadIdx.x;
  int ay = blockIdx.y * blockDim.y + threadIdx.y;
  const int aw = tiling.Width();
  const int ah = tiling.Height();
  const int image_width = tiling.image_width;
  const int image_height = tiling.image_height;
  if (ax >= aw || ay >= ah) return;

  int x, y;
  const int cam = tiling.CameraAt(ax, ay, &x, &y);
  if (cam < 0 || cam >= tiling.num_images) return;

  float gain[3];
  SampleField(field, cam, image_width, image_height, grid_width, grid_height,
              nch, static_cast<float>(x), static_cast<float>(y), gain);

  unsigned char* px = atlas + static_cast<size_t>(3) * (ay * aw + ax);
  float lin[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) lin[c] = SrgbToLinearDev(px[c]);
  if (rgb) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      float gg = fminf(max_log_gain, fmaxf(-max_log_gain, gain[c]));
      lin[c] *= __expf(gg);
    }
  } else {
    float gg = fminf(max_log_gain, fmaxf(-max_log_gain, gain[0]));
    float s = __expf(gg);
#pragma unroll
    for (int c = 0; c < 3; ++c) lin[c] *= s;
  }
#pragma unroll
  for (int c = 0; c < 3; ++c) px[c] = LinearToSrgbDev(lin[c]);
}

}  // namespace

// ===========================================================================
// GPU field compute: adjacency + seam detect + proxy residual + Jacobi solve.
// Mirrors src/grtmesh/tex/color_correction_cpu.cc (the CPU reference / parity
// ground truth).
// ===========================================================================
namespace {

struct SeamSample {
  int face_a, face_b, edge_a, camera_a, camera_b, cell_a, cell_b;
  float residual[3], weight;
};

__device__ __forceinline__ float Luminance(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}
__device__ __forceinline__ int LocalVertexIndex(const int* faces, int face,
                                                int v) {
#pragma unroll
  for (int k = 0; k < 3; ++k)
    if (faces[3 * face + k] == v) return k;
  return -1;
}
__device__ __forceinline__ int GridCell(float x, float y, int image_width,
                                        int image_height, int grid_width,
                                        int grid_height) {
  int i = static_cast<int>(floorf(x * grid_width / image_width)),
      j = static_cast<int>(floorf(y * grid_height / image_height));
  i = min(grid_width - 1, max(0, i));
  j = min(grid_height - 1, max(0, j));
  return j * grid_width + i;
}
// Bilinear sample of the per-camera interleaved-float proxy (matches CPU
// SampleProxyLinear): proxy coords px = x/ds - 0.5.
__device__ bool SampleProxy(const float* proxy, int pw, int ph, int ds, int cam,
                            float x, float y, float out[3]) {
  if (x < 0.f || y < 0.f) return false;
  float px = x / ds - 0.5f, py = y / ds - 0.5f;
  if (px < -0.5f) px = 0.f;
  if (py < -0.5f) py = 0.f;
  int x0 = static_cast<int>(floorf(px)), y0 = static_cast<int>(floorf(py));
  if (x0 < 0 || y0 < 0 || x0 > pw - 1 || y0 > ph - 1) return false;
  int x1 = min(x0 + 1, pw - 1), y1 = min(y0 + 1, ph - 1);
  float fx = px - x0, fy = py - y0;
  const float* base = proxy + static_cast<size_t>(cam) * pw * ph * 3;
  for (int c = 0; c < 3; ++c) {
    float v00 = base[(y0 * pw + x0) * 3 + c],
          v10 = base[(y0 * pw + x1) * 3 + c];
    float v01 = base[(y1 * pw + x0) * 3 + c],
          v11 = base[(y1 * pw + x1) * 3 + c];
    out[c] = (v00 * (1 - fx) + v10 * fx) * (1 - fy) +
             (v01 * (1 - fx) + v11 * fx) * fy;
  }
  return true;
}

__global__ void DetectSeamsKernel(const int* face_adjacency,
                                  const int* face_camera_ids, int num_faces,
                                  int num_cams, SeamSample* seams, uint32_t cap,
                                  uint32_t* count, const float3* normals,
                                  float min_cos) {
  int fid = blockIdx.x * blockDim.x + threadIdx.x;
  if (fid >= num_faces) return;
  for (int e = 0; e < 3; ++e) {
    int neighbor = face_adjacency[3 * fid + e];
    if (neighbor < 0 || neighbor >= num_faces || fid >= neighbor) continue;
    int cam_a = face_camera_ids[fid], cam_b = face_camera_ids[neighbor];
    if (cam_a < 0 || cam_b < 0 || cam_a == cam_b) continue;
    if (cam_a >= num_cams || cam_b >= num_cams) continue;
    // Skip sharp folds / veil triangles (occlusion edges): the two faces are
    // not a continuous surface, so matching their colours is meaningless.
    if (normals != nullptr) {
      float3 na = normals[fid], nb = normals[neighbor];
      float c = na.x * nb.x + na.y * nb.y + na.z * nb.z;
      if (c < min_cos) continue;
    }
    uint32_t idx = atomicAdd(count, 1u);
    if (idx < cap) {
      SeamSample s;
      s.face_a = fid;
      s.face_b = neighbor;
      s.edge_a = e;
      s.camera_a = cam_a;
      s.camera_b = cam_b;
      s.cell_a = 0;
      s.cell_b = 0;
      s.residual[0] = s.residual[1] = s.residual[2] = 0.f;
      s.weight = 0.f;
      seams[idx] = s;
    }
  }
}

__device__ __forceinline__ float MedianInplace(float* a, int n) {
  // insertion sort (n <= 32), robust to a few outlier samples.
  for (int i = 1; i < n; ++i) {
    float k = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > k) {
      a[j + 1] = a[j];
      --j;
    }
    a[j + 1] = k;
  }
  return (n & 1) ? a[n / 2] : 0.5f * (a[n / 2 - 1] + a[n / 2]);
}

__global__ void SampleResidualsKernel(SeamSample* seams, uint32_t count,
                                      const int* faces, const float2* proj,
                                      const float* proxy, int pw, int ph,
                                      int ds, int samples, float inset,
                                      int luma, int grid_width, int grid_height,
                                      int image_width, int image_height) {
  uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= count) return;
  SeamSample s = seams[sid];
  const int fa = s.face_a, fb = s.face_b, ea = s.edge_a;
  const int vi = faces[3 * fa + ea], vj = faces[3 * fa + (ea + 1) % 3];
  const int ai = ea, aj = (ea + 1) % 3, aopp = (ea + 2) % 3;
  const int bi = LocalVertexIndex(faces, fb, vi),
            bj = LocalVertexIndex(faces, fb, vj);
  if (bi < 0 || bj < 0) {
    s.weight = 0.f;
    seams[sid] = s;
    return;
  }
  const int bopp = 3 - bi - bj;
  const float2 pa_i = proj[3 * fa + ai], pa_j = proj[3 * fa + aj],
               pa_opp = proj[3 * fa + aopp];
  const float2 pb_i = proj[3 * fb + bi], pb_j = proj[3 * fb + bj],
               pb_opp = proj[3 * fb + bopp];
  // (B) skip inpainted / degenerate faces: their projected triangle is
  // collapsed (all verts ~coincide), so edge sampling reads a single bogus
  // spot and corrupts the residual. Require a non-trivial projected area.
  {
    float area_a = fabsf((pa_j.x - pa_i.x) * (pa_opp.y - pa_i.y) -
                         (pa_j.y - pa_i.y) * (pa_opp.x - pa_i.x));
    float area_b = fabsf((pb_j.x - pb_i.x) * (pb_opp.y - pb_i.y) -
                         (pb_j.y - pb_i.y) * (pb_opp.x - pb_i.x));
    if (area_a < 1.f || area_b < 1.f) {
      s.weight = 0.f;
      seams[sid] = s;
      return;
    }
  }
  // (1) Robust residual: reject dark/occluded samples and take the MEDIAN of
  // the per-sample log-ratios so a few outliers (inpaint, holes, dots) near the
  // seam cannot drag the estimate. d_min in linear luma (~sRGB 28).
  float v0[32], v1[32], v2[32];
  int valid = 0;
  for (int i = 0; i < samples; ++i) {
    float t = (i + 0.5f) / samples;
    float2 ea_p = make_float2(pa_i.x + (pa_j.x - pa_i.x) * t,
                              pa_i.y + (pa_j.y - pa_i.y) * t);
    float2 eb_p = make_float2(pb_i.x + (pb_j.x - pb_i.x) * t,
                              pb_i.y + (pb_j.y - pb_i.y) * t);
    float2 da = make_float2(pa_opp.x - ea_p.x, pa_opp.y - ea_p.y);
    float2 db = make_float2(pb_opp.x - eb_p.x, pb_opp.y - eb_p.y);
    float la = sqrtf(da.x * da.x + da.y * da.y),
          lb = sqrtf(db.x * db.x + db.y * db.y);
    if (la > 1e-4f) {
      da.x /= la;
      da.y /= la;
    }
    if (lb > 1e-4f) {
      db.x /= lb;
      db.y /= lb;
    }
    float2 pa = make_float2(ea_p.x + da.x * inset, ea_p.y + da.y * inset);
    float2 pb = make_float2(eb_p.x + db.x * inset, eb_p.y + db.y * inset);
    float color_a[3], color_b[3];
    if (!SampleProxy(proxy, pw, ph, ds, s.camera_a, pa.x, pa.y, color_a))
      continue;
    if (!SampleProxy(proxy, pw, ph, ds, s.camera_b, pb.x, pb.y, color_b))
      continue;
    float luminance_a = Luminance(color_a[0], color_a[1], color_a[2]),
          luminance_b = Luminance(color_b[0], color_b[1], color_b[2]);
    if (fminf(luminance_a, luminance_b) < 0.01f)
      continue;  // reject dark/occlusion samples
    if (luma) {
      v0[valid] = logf(luminance_b) - logf(luminance_a);
    } else {
      v0[valid] =
          logf(fmaxf(color_b[0], 1e-6f)) - logf(fmaxf(color_a[0], 1e-6f));
      v1[valid] =
          logf(fmaxf(color_b[1], 1e-6f)) - logf(fmaxf(color_a[1], 1e-6f));
      v2[valid] =
          logf(fmaxf(color_b[2], 1e-6f)) - logf(fmaxf(color_a[2], 1e-6f));
    }
    ++valid;
  }
  if (valid > 0) {
    s.residual[0] = MedianInplace(v0, valid);
    s.residual[1] = luma ? 0.f : MedianInplace(v1, valid);
    s.residual[2] = luma ? 0.f : MedianInplace(v2, valid);
    s.weight = static_cast<float>(valid) / samples;
    // nearest field cell at the midpoint inset position (matches CPU).
    float t = 0.5f;
    float2 ea_p = make_float2(pa_i.x + (pa_j.x - pa_i.x) * t,
                              pa_i.y + (pa_j.y - pa_i.y) * t);
    float2 eb_p = make_float2(pb_i.x + (pb_j.x - pb_i.x) * t,
                              pb_i.y + (pb_j.y - pb_i.y) * t);
    float2 da = make_float2(pa_opp.x - ea_p.x, pa_opp.y - ea_p.y);
    float2 db = make_float2(pb_opp.x - eb_p.x, pb_opp.y - eb_p.y);
    float la = sqrtf(da.x * da.x + da.y * da.y),
          lb = sqrtf(db.x * db.x + db.y * db.y);
    if (la > 1e-4f) {
      da.x /= la;
      da.y /= la;
    }
    if (lb > 1e-4f) {
      db.x /= lb;
      db.y /= lb;
    }
    float2 pa = make_float2(ea_p.x + da.x * inset, ea_p.y + da.y * inset);
    float2 pb = make_float2(eb_p.x + db.x * inset, eb_p.y + db.y * inset);
    s.cell_a = GridCell(pa.x, pa.y, image_width, image_height, grid_width,
                        grid_height);
    s.cell_b = GridCell(pb.x, pb.y, image_width, image_height, grid_width,
                        grid_height);
  } else {
    s.weight = 0.f;
  }
  seams[sid] = s;
}

__global__ void ClearAccumKernel(const float* g, float* num, float* den,
                                 int num_cams, int grid_width, int grid_height,
                                 int nch, float ls, float la) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cells = grid_width * grid_height;
  const int total = num_cams * cells * nch;
  if (idx >= total) return;
  int ch = idx % nch, tmp = idx / nch, cell = tmp % cells, cam = tmp / cells;
  int i = cell % grid_width, j = cell / grid_width;
  float n = 0.f, d = 0.f;
  if (cells > 1 && ls > 0.f) {
    const int di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
      int ni = i + di[k], nj = j + dj[k];
      if (ni < 0 || ni >= grid_width || nj < 0 || nj >= grid_height) continue;
      n += ls * g[(cam * cells + nj * grid_width + ni) * nch + ch];
      d += ls;
    }
  }
  d += la;
  num[idx] = n;
  den[idx] = d;
}

__global__ void SplatSeamsKernel(const SeamSample* seams, uint32_t count,
                                 const float* g, float* num, float* den,
                                 int cells, int nch, float dw) {
  uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= count) return;
  SeamSample s = seams[sid];
  if (s.weight <= 0.f) return;
  const float w = dw * s.weight;  // seam-data-term weight multiplier
  for (int ch = 0; ch < nch; ++ch) {
    const float d = s.residual[ch];
    int ix_a = (s.camera_a * cells + s.cell_a) * nch + ch;
    int ix_b = (s.camera_b * cells + s.cell_b) * nch + ch;
    float g_a = g[ix_a], g_b = g[ix_b];
    atomicAdd(&num[ix_a], w * (g_b + d));
    atomicAdd(&den[ix_a], w);
    atomicAdd(&num[ix_b], w * (g_a - d));
    atomicAdd(&den[ix_b], w);
  }
}

__global__ void UpdateFieldKernel(float* g, const float* num, const float* den,
                                  int total, float omega, float max_g) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float cand = num[idx] / fmaxf(den[idx], 1e-6f);
  float v = (1.f - omega) * g[idx] + omega * cand;
  if (max_g > 0.f) v = fminf(max_g, fmaxf(-max_g, v));
  g[idx] = v;
}

// Accumulates seam residual magnitude before (|d|) and after (|d-(gA-gB)|)
// correction, channel 0, into out[0]/out[1]. For CPU/GPU parity reporting.
__global__ void ResidualStatsKernel(const SeamSample* seams, uint32_t count,
                                    const float* g, int cells, int nch,
                                    float* out) {
  uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= count) return;
  SeamSample s = seams[sid];
  if (s.weight <= 0.f) return;
  float g_a = g[(s.camera_a * cells + s.cell_a) * nch];
  float g_b = g[(s.camera_b * cells + s.cell_b) * nch];
  atomicAdd(&out[0], fabsf(s.residual[0]));
  atomicAdd(&out[1], fabsf(s.residual[0] - (g_a - g_b)));
  atomicAdd(&out[2], 1.f);
}

__global__ void GaugeKernel(float* g, int num_cams, int cells, int nch,
                            int ref) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= nch) return;
  if (ref >= 0 && ref < num_cams) {
    for (int c = 0; c < cells; ++c) g[(ref * cells + c) * nch + ch] = 0.f;
  } else {
    float mean = 0.f;
    for (int k = 0; k < num_cams; ++k)
      for (int c = 0; c < cells; ++c) mean += g[(k * cells + c) * nch + ch];
    mean /= (num_cams * cells);
    for (int k = 0; k < num_cams; ++k)
      for (int c = 0; c < cells; ++c) g[(k * cells + c) * nch + ch] -= mean;
  }
}

// Build the per-camera low-res linear-RGB proxy directly from the device atlas
// (one thread per proxy cell). Box filter, sRGB->linear; matches the CPU proxy.
__global__ void BuildProxyFromAtlasKernel(const unsigned char* atlas,
                                          AtlasTiling tiling, int ds, int pw,
                                          int ph, float* proxy) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cellsp = pw * ph;
  const int num = tiling.num_images;
  const int aw = tiling.Width();
  const int image_width = tiling.image_width;
  const int image_height = tiling.image_height;
  if (idx >= num * cellsp) return;
  int cell = idx % cellsp, cam = idx / cellsp;
  int pi = cell % pw, pj = cell / pw;
  const int2 origin = tiling.Origin(cam);
  const int ox = origin.x, oy = origin.y;
  double acc[3] = {0, 0, 0};
  int cnt = 0;
  for (int yy = pj * ds; yy < min((pj + 1) * ds, image_height); ++yy)
    for (int xx = pi * ds; xx < min((pi + 1) * ds, image_width); ++xx) {
      const unsigned char* p =
          atlas + static_cast<size_t>(3) * ((oy + yy) * aw + (ox + xx));
      acc[0] += SrgbToLinearDev(p[0]);
      acc[1] += SrgbToLinearDev(p[1]);
      acc[2] += SrgbToLinearDev(p[2]);
      ++cnt;
    }
  float inv = cnt > 0 ? 1.f / cnt : 0.f;
  float* dst = proxy + static_cast<size_t>((cam * cellsp + pj * pw + pi)) * 3;
  dst[0] = static_cast<float>(acc[0]) * inv;
  dst[1] = static_cast<float>(acc[1]) * inv;
  dst[2] = static_cast<float>(acc[2]) * inv;
}

// Temporal smoothing: EMA blend of the current per-frame field with the
// previous frame's field. cur = a*cur + (1-a)*prev.
__global__ void EmaBlendKernel(float* cur, const float* prev, int n, float a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  cur[i] = a * cur[i] + (1.f - a) * prev[i];
}

}  // namespace

struct ColorCorrectionContext::Impl {
  // Bake-only path (field solved on the host).
  grtmesh::DeviceBuffer<unsigned char> bake_atlas;
  grtmesh::DeviceBuffer<float> bake_field;
  cudaEvent_t bake_start = nullptr, bake_stop = nullptr;

  // Transfer-merged GPU path: persisted faces + normals (copied before the
  // volume is cleared), the face adjacency, seams, the per-camera proxy, the
  // field and its Jacobi accumulators, the EMA state, and the device atlas
  // (uploaded once, used for the proxy build AND the bake).
  grtmesh::DeviceBuffer<int> faces;
  grtmesh::DeviceBuffer<float3> normals;
  grtmesh::DeviceBuffer<uint64_t> edge_key;
  grtmesh::DeviceBuffer<int> edge_face, face_adjacency;
  grtmesh::DeviceBuffer<SeamSample> seams;
  grtmesh::DeviceBuffer<uint32_t> seam_count;
  grtmesh::DeviceBuffer<float> proxy, field, num, den, resid;
  grtmesh::DeviceBuffer<float> field_prev;  // temporal EMA state across frames
  grtmesh::DeviceBuffer<unsigned char> atlas;
  int prepared_num_faces = 0;
  bool has_prev = false;
  int prev_cams = 0, prev_w = 0, prev_h = 0, prev_channels = 0;
  int dump_index = 0;
  cudaEvent_t ev_start = nullptr, ev_stop = nullptr;

  ~Impl() {
    if (bake_start) cudaEventDestroy(bake_start);
    if (bake_stop) cudaEventDestroy(bake_stop);
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_stop) cudaEventDestroy(ev_stop);
  }
};

ColorCorrectionContext::ColorCorrectionContext()
    : impl_(std::make_unique<Impl>()) {}
ColorCorrectionContext::~ColorCorrectionContext() = default;
void ColorCorrectionContext::Reset() { impl_ = std::make_unique<Impl>(); }

float ColorCorrectionContext::Bake(grtmesh::ColorImage& atlas,
                                   const std::vector<float>& field, int img_w,
                                   int img_h, int num_cams, int grid_w,
                                   int grid_h, int nch, bool rgb,
                                   float max_log_gain, AtlasLayout layout) {
  if (atlas.empty() || field.empty()) return -1.f;
  const AtlasTiling tiling = MakeAtlasTiling(layout, num_cams, img_w, img_h);
  const int aw = tiling.Width();
  const int ah = tiling.Height();
  if (static_cast<int>(atlas.width) != aw ||
      static_cast<int>(atlas.height) != ah)
    return -1.f;
  const size_t atlas_bytes = static_cast<size_t>(aw) * ah * 3;
  const size_t field_count = field.size();

  Impl& b = *impl_;
  if (b.bake_start == nullptr) {
    cudaEventCreate(&b.bake_start);
    cudaEventCreate(&b.bake_stop);
  }
  if (!b.bake_atlas.Reserve(atlas_bytes)) return -1.f;
  if (!b.bake_field.Reserve(field_count)) return -1.f;

  cudaEventRecord(b.bake_start);
  cudaMemcpy(b.bake_atlas.data(), atlas.data(), atlas_bytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(b.bake_field.data(), field.data(), field_count * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((aw + block.x - 1) / block.x, (ah + block.y - 1) / block.y);
  BakeFieldKernel<<<grid, block>>>(b.bake_atlas.data(), tiling,
                                   b.bake_field.data(), grid_w, grid_h, nch,
                                   rgb ? 1 : 0, max_log_gain);

  cudaMemcpy(atlas.data(), b.bake_atlas.data(), atlas_bytes,
             cudaMemcpyDeviceToHost);
  cudaEventRecord(b.bake_stop);
  cudaEventSynchronize(b.bake_stop);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, b.bake_start, b.bake_stop);
  if (cudaGetLastError() != cudaSuccess) return -1.f;
  return ms;
}

// --- Transfer-merged path ---------------------------------------------------

bool ColorCorrectionContext::PrepareGeometry(
    const int* d_faces, int num_faces, const float3* d_face_normals,
    const int* d_prebuilt_face_adjacency) {
  Impl& b = *impl_;
  b.prepared_num_faces = 0;
  if (num_faces < 1) return false;
  const size_t fc = static_cast<size_t>(num_faces);
  if (!b.edge_key.Reserve(3 * fc) || !b.edge_face.Reserve(3 * fc) ||
      !b.face_adjacency.Reserve(3 * fc) || !b.seams.Reserve(fc) ||
      !b.faces.Reserve(3 * fc)) {
    return false;
  }
  cudaMemcpy(b.faces.data(), d_faces, sizeof(int) * 3 * fc,
             cudaMemcpyDeviceToDevice);
  // Persist per-face normals so the (later, post-Clear) seam detection can skip
  // sharp folds / veil triangles at occlusion edges.
  if (d_face_normals != nullptr) {
    if (!b.normals.Reserve(fc)) return false;
    cudaMemcpy(b.normals.data(), d_face_normals, sizeof(float3) * fc,
               cudaMemcpyDeviceToDevice);
  }
  if (d_prebuilt_face_adjacency != nullptr) {
    // Reuse the propagation stage's identical fixed-size adjacency (same
    // mesh): a device copy instead of rebuilding the edge-key radix sort +
    // linking.
    cudaMemcpy(b.face_adjacency.data(), d_prebuilt_face_adjacency,
               sizeof(int) * 3 * fc, cudaMemcpyDeviceToDevice);
  } else {
    grtmesh::BuildFaceAdjacency(d_faces, num_faces, b.edge_key.data(),
                                b.edge_face.data(), b.face_adjacency.data());
  }
  b.prepared_num_faces = num_faces;
  return cudaGetLastError() == cudaSuccess;
}

float ColorCorrectionContext::SolveAndBake(
    grtmesh::ColorImage& atlas, const int* d_face_camera_ids,
    const float2* d_face_image_coords, int num_cams,
    const ColorCorrectionConfig& cfg, int img_w, int img_h, AtlasLayout layout,
    ColorCorrectionStats& stats, const unsigned char* d_atlas_source) {
  Impl& b = *impl_;
  const int num_faces = b.prepared_num_faces;
  if (num_faces < 1 || num_cams <= 1 || !b.faces || atlas.empty()) return -1.f;
  const bool rgb = cfg.space == ColorCorrectionSpace::kLinearRgb;
  const int nch = rgb ? 3 : 1;
  const int grid_width = cfg.mode == ColorCorrectionMode::kGlobalOnly
                             ? 1
                             : std::max(1, cfg.field_grid_w);
  const int grid_height = cfg.mode == ColorCorrectionMode::kGlobalOnly
                              ? 1
                              : std::max(1, cfg.field_grid_h);
  const int cells = grid_width * grid_height;
  const int ds = std::max(1, cfg.proxy_downscale);
  const int samples = std::max(1, std::min(32, cfg.seam_samples));
  const float inset = std::max(0.f, cfg.sample_inset_proxy_pixels) *
                      std::max(1.f, static_cast<float>(ds));
  const int pw = (img_w + ds - 1) / ds;
  const int ph = (img_h + ds - 1) / ds;
  const AtlasTiling tiling = MakeAtlasTiling(layout, num_cams, img_w, img_h);
  const int aw = tiling.Width();
  const int ah = tiling.Height();
  if (static_cast<int>(atlas.width) != aw ||
      static_cast<int>(atlas.height) != ah)
    return -1.f;
  const size_t atlas_bytes = static_cast<size_t>(aw) * ah * 3;

  if (!b.ev_start) {
    cudaEventCreate(&b.ev_start);
    cudaEventCreate(&b.ev_stop);
  }
  if (!b.seam_count.Reserve(1)) return -1.f;
  if (!b.resid.Reserve(3)) return -1.f;
  const size_t proxy_sz = static_cast<size_t>(pw) * ph * num_cams * 3;
  if (!b.proxy.Reserve(proxy_sz)) return -1.f;
  const size_t fld = static_cast<size_t>(num_cams) * cells * nch;
  if (!b.field.Reserve(fld) || !b.num.Reserve(fld) || !b.den.Reserve(fld))
    return -1.f;
  if (!b.atlas.Reserve(atlas_bytes)) return -1.f;

  // Optional transfer/compute split profiling.
  const bool profile_transfers = cfg.profile_transfers;
  cudaEvent_t ev_uploaded = nullptr, ev_downloading = nullptr;
  cudaEventCreate(&ev_uploaded);
  cudaEventCreate(&ev_downloading);

  // The atlas is expected to be in pinned host memory (see Mesher), so this
  // round trip uses the fast path.
  cudaEventRecord(b.ev_start);
  if (d_atlas_source != nullptr) {
    cudaMemcpy(b.atlas.data(), d_atlas_source, atlas_bytes,
               cudaMemcpyDeviceToDevice);
  } else {
    cudaMemcpy(b.atlas.data(), atlas.data(), atlas_bytes,
               cudaMemcpyHostToDevice);
  }
  cudaEventRecord(ev_uploaded);  // after the atlas upload
  // Proxy built ON the GPU from the just-uploaded atlas (no host proxy build).
  BuildProxyFromAtlasKernel<<<(num_cams * pw * ph + 255) / 256, 256>>>(
      b.atlas.data(), tiling, ds, pw, ph, b.proxy.data());

  cudaMemset(b.seam_count.data(), 0, sizeof(uint32_t));
  DetectSeamsKernel<<<(num_faces + 255) / 256, 256>>>(
      b.face_adjacency.data(), d_face_camera_ids, num_faces, num_cams,
      b.seams.data(), static_cast<uint32_t>(b.seams.capacity()),
      b.seam_count.data(), b.normals.data(), cfg.seam_max_dihedral_cos);
  uint32_t seam_count = 0;
  cudaMemcpy(&seam_count, b.seam_count.data(), sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  seam_count = std::min(seam_count, static_cast<uint32_t>(b.seams.capacity()));
  if (seam_count > 0)
    SampleResidualsKernel<<<(seam_count + 255) / 256, 256>>>(
        b.seams.data(), seam_count, b.faces.data(), d_face_image_coords,
        b.proxy.data(), pw, ph, ds, samples, inset, rgb ? 0 : 1, grid_width,
        grid_height, img_w, img_h);

  cudaMemset(b.field.data(), 0, sizeof(float) * fld);
  const int total = static_cast<int>(fld);
  const int iters = std::max(1, cfg.field_iterations);
  const float ls = std::max(0.f, cfg.lambda_smooth);
  const float la = std::max(0.f, cfg.lambda_anchor);
  const float omega = cfg.field_omega <= 0.f ? 0.7f : cfg.field_omega;
  const float max_g = std::max(0.f, cfg.max_log_gain);
  for (int it = 0; it < iters; ++it) {
    ClearAccumKernel<<<(total + 255) / 256, 256>>>(
        b.field.data(), b.num.data(), b.den.data(), num_cams, grid_width,
        grid_height, nch, ls, la);
    if (seam_count > 0)
      SplatSeamsKernel<<<(seam_count + 255) / 256, 256>>>(
          b.seams.data(), seam_count, b.field.data(), b.num.data(),
          b.den.data(), cells, nch, std::max(0.f, cfg.seam_data_weight));
    UpdateFieldKernel<<<(total + 255) / 256, 256>>>(
        b.field.data(), b.num.data(), b.den.data(), total, omega, max_g);
    GaugeKernel<<<1, nch>>>(b.field.data(), num_cams, cells, nch,
                            cfg.reference_camera);
  }

  // Temporal smoothing (EMA across frames) to remove flicker from
  // independently-solved per-frame fields. Camera gain is near-constant over
  // time, so a small current-frame weight is appropriate.
  {
    if (b.prev_cams != num_cams || b.prev_w != grid_width ||
        b.prev_h != grid_height || b.prev_channels != nch) {
      b.has_prev = false;
      b.prev_cams = num_cams;
      b.prev_w = grid_width;
      b.prev_h = grid_height;
      b.prev_channels = nch;
    }
    const float ema_a = cfg.temporal_alpha;
    if (ema_a > 0.f && ema_a < 1.f) {
      if (fld > b.field_prev.capacity()) {
        if (!b.field_prev.Reserve(fld)) return -1.f;
        b.has_prev = false;
      }
      if (b.has_prev) {
        EmaBlendKernel<<<(total + 255) / 256, 256>>>(
            b.field.data(), b.field_prev.data(), total, ema_a);
        GaugeKernel<<<1, nch>>>(b.field.data(), num_cams, cells, nch,
                                cfg.reference_camera);
      }
      cudaMemcpy(b.field_prev.data(), b.field.data(), sizeof(float) * fld,
                 cudaMemcpyDeviceToDevice);
      b.has_prev = true;
    }
  }

  // Optional per-frame gain-field dump for visualization. Layout matches the
  // field: [cam][gy][gx][ch], log gains before the |g|<=max_log_gain clamp
  // applied at bake time.
  if (!cfg.gain_dump_dir.empty()) {
    std::vector<float> hf(fld);
    cudaMemcpy(hf.data(), b.field.data(), sizeof(float) * fld,
               cudaMemcpyDeviceToHost);
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/gain_%06d.txt",
                  cfg.gain_dump_dir.c_str(), b.dump_index++);
    if (FILE* fp = std::fopen(path, "w")) {
      std::fprintf(fp, "%d %d %d %d %g\n", num_cams, grid_width, grid_height,
                   nch, static_cast<double>(max_g));
      for (size_t i = 0; i < fld; ++i)
        std::fprintf(fp, "%g\n", static_cast<double>(hf[i]));
      std::fclose(fp);
    }
  }

  cudaMemset(b.resid.data(), 0, sizeof(float) * 3);
  if (seam_count > 0)
    ResidualStatsKernel<<<(seam_count + 255) / 256, 256>>>(
        b.seams.data(), seam_count, b.field.data(), cells, nch, b.resid.data());
  float resid[3] = {0, 0, 0};
  cudaMemcpy(resid, b.resid.data(), sizeof(float) * 3, cudaMemcpyDeviceToHost);
  if (resid[2] > 0.f) {
    stats.mean_abs_seam_residual_before = resid[0] / resid[2];
    stats.mean_abs_seam_residual_after = resid[1] / resid[2];
  }

  // Bake the field into the device atlas in place, then download once.
  dim3 block(16, 16);
  dim3 grid((aw + block.x - 1) / block.x, (ah + block.y - 1) / block.y);
  BakeFieldKernel<<<grid, block>>>(b.atlas.data(), tiling, b.field.data(),
                                   grid_width, grid_height, nch, rgb ? 1 : 0,
                                   max_g);
  cudaEventRecord(ev_downloading);  // before D2H
  cudaMemcpy(atlas.data(), b.atlas.data(), atlas_bytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(b.ev_stop);
  cudaEventSynchronize(b.ev_stop);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, b.ev_start, b.ev_stop);
  {
    float up = 0.f, comp = 0.f, dn = 0.f;
    cudaEventElapsedTime(&up, b.ev_start, ev_uploaded);  // atlas upload
    cudaEventElapsedTime(&comp, ev_uploaded,
                         ev_downloading);                  // all CC kernels
    cudaEventElapsedTime(&dn, ev_downloading, b.ev_stop);  // D2H download
    stats.atlas_upload_ms = up;
    stats.atlas_download_ms = dn;
    if (profile_transfers) {
      fprintf(stderr,
              "CCSPLIT bytes=%zu up=%.4f comp=%.4f dn=%.4f total=%.4f\n",
              atlas_bytes, up, comp, dn, ms);
    }
    cudaEventDestroy(ev_uploaded);
    cudaEventDestroy(ev_downloading);
  }
  stats.seam_edge_count = seam_count;
  if (cudaGetLastError() != cudaSuccess) return -1.f;
  return ms;
}

}  // namespace grtmesh::tex
