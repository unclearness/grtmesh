// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "grtmesh/tex/color_correction_internal.h"

namespace grtmesh::tex {
namespace {

constexpr float kEps = 1e-6f;

// ---- sRGB <-> linear -------------------------------------------------------
constexpr int kLinToSrgbN = 4096;
struct SrgbLut {
  float to_linear[256];
  uint8_t to_srgb[kLinToSrgbN];  // linear[0,1] (quantized) -> sRGB byte
  SrgbLut() {
    for (int i = 0; i < 256; ++i) {
      float c = i / 255.f;
      to_linear[i] =
          c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    }
    for (int i = 0; i < kLinToSrgbN; ++i) {
      float c = static_cast<float>(i) / (kLinToSrgbN - 1);
      float s = c <= 0.0031308f ? c * 12.92f
                                : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
      int v = static_cast<int>(s * 255.f + 0.5f);
      to_srgb[i] = static_cast<uint8_t>(std::min(255, std::max(0, v)));
    }
  }
};
const SrgbLut kGSrgbLut;

inline float SrgbToLinear(uint8_t v) { return kGSrgbLut.to_linear[v]; }

inline uint8_t LinearToSrgb(float c) {
  c = std::min(1.f, std::max(0.f, c));
  int i = static_cast<int>(c * (kLinToSrgbN - 1) + 0.5f);
  return kGSrgbLut.to_srgb[i];
}

inline float Luminance(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Median by insertion sort (n <= 32); robust to a few outlier samples.
inline float MedianInplace(float* a, int n) {
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

// Bilinear sample of a packed RGB8 image, returned as linear RGB. Returns false
// when the sample falls outside the valid interior.
inline bool SampleLinearBilinear(const grtmesh::ColorImage& img, float x,
                                 float y, float out[3]) {
  const int w = static_cast<int>(img.width);
  const int h = static_cast<int>(img.height);
  if (x < 0.f || y < 0.f || x > w - 1.f || y > h - 1.f) return false;
  int x0 = static_cast<int>(std::floor(x));
  int y0 = static_cast<int>(std::floor(y));
  int x1 = std::min(x0 + 1, w - 1);
  int y1 = std::min(y0 + 1, h - 1);
  float fx = x - x0, fy = y - y0;
  const uint8_t* p = img.data();
  for (int c = 0; c < 3; ++c) {
    float v00 = SrgbToLinear(p[3 * (y0 * w + x0) + c]);
    float v10 = SrgbToLinear(p[3 * (y0 * w + x1) + c]);
    float v01 = SrgbToLinear(p[3 * (y1 * w + x0) + c]);
    float v11 = SrgbToLinear(p[3 * (y1 * w + x1) + c]);
    float v0 = v00 * (1 - fx) + v10 * fx;
    float v1 = v01 * (1 - fx) + v11 * fx;
    out[c] = v0 * (1 - fy) + v1 * fy;
  }
  return true;
}

// Low-res radiometric proxy: box-downscaled linear-RGB copy of a source image.
// Sampling residuals from the proxy (not full-res) isolates the low-frequency
// radiometric difference from high-frequency texture / sub-pixel misalignment,
// which would otherwise inflate the seam residual and cause over-correction.
struct Proxy {
  int w, h, ds;
  std::vector<float> data;  // w*h*3 linear RGB
};
inline Proxy BuildProxy(const grtmesh::ColorImage& img, int ds) {
  const int w = static_cast<int>(img.width), h = static_cast<int>(img.height);
  ds = std::max(1, ds);
  Proxy p;
  p.ds = ds;
  p.w = (w + ds - 1) / ds;
  p.h = (h + ds - 1) / ds;
  p.data.assign(static_cast<size_t>(p.w) * p.h * 3, 0.f);
  const uint8_t* src = img.data();
  for (int j = 0; j < p.h; ++j)
    for (int i = 0; i < p.w; ++i) {
      double acc[3] = {0, 0, 0};
      int cnt = 0;
      for (int yy = j * ds; yy < std::min((j + 1) * ds, h); ++yy)
        for (int xx = i * ds; xx < std::min((i + 1) * ds, w); ++xx) {
          for (int c = 0; c < 3; ++c)
            acc[c] += SrgbToLinear(src[3 * (yy * w + xx) + c]);
          cnt++;
        }
      float inv = cnt > 0 ? 1.f / cnt : 0.f;
      for (int c = 0; c < 3; ++c)
        p.data[3 * (j * p.w + i) + c] = static_cast<float>(acc[c]) * inv;
    }
  return p;
}
// Bilinear sample of the proxy at full-resolution source coords (x,y).
inline bool SampleProxyLinear(const Proxy& p, float x, float y, float out[3]) {
  if (x < 0.f || y < 0.f) return false;
  float px = x / p.ds - 0.5f, py = y / p.ds - 0.5f;
  if (px < -0.5f) px = 0.f;
  if (py < -0.5f) py = 0.f;
  int x0 = static_cast<int>(std::floor(px)),
      y0 = static_cast<int>(std::floor(py));
  int x1 = std::min(x0 + 1, p.w - 1), y1 = std::min(y0 + 1, p.h - 1);
  if (x0 < 0 || y0 < 0 || x0 > p.w - 1 || y0 > p.h - 1) return false;
  float fx = px - x0, fy = py - y0;
  for (int c = 0; c < 3; ++c) {
    float v00 = p.data[3 * (y0 * p.w + x0) + c];
    float v10 = p.data[3 * (y0 * p.w + x1) + c];
    float v01 = p.data[3 * (y1 * p.w + x0) + c];
    float v11 = p.data[3 * (y1 * p.w + x1) + c];
    out[c] = (v00 * (1 - fx) + v10 * fx) * (1 - fy) +
             (v01 * (1 - fx) + v11 * fx) * fy;
  }
  return true;
}

// ---- low-res field grid helpers -------------------------------------------
// 4-tap bilinear weights of source pixel (x,y) onto the GwxGh cell grid that
// spans a WxH image. Returns the cell indices (j*Gw+i) and weights.
struct GridTaps {
  int cell[4];
  float w[4];
  int n;
};
// Single nearest cell (weight 1). Used for the seam DATA term: bilinear 4-tap
// splatting couples a seam's co-cells positively within one grid and destroys
// diagonal dominance (Jacobi then diverges). Nearest-cell keeps the normal
// system diagonally dominant; the Laplacian still makes the field smooth.
inline GridTaps GridNearestTap(float x, float y, int w, int h, int gw, int gh) {
  int i = static_cast<int>(std::floor(x * gw / w));
  int j = static_cast<int>(std::floor(y * gh / h));
  i = std::min(gw - 1, std::max(0, i));
  j = std::min(gh - 1, std::max(0, j));
  GridTaps t;
  t.cell[0] = j * gw + i;
  t.w[0] = 1.f;
  t.n = 1;
  return t;
}
inline GridTaps GridBilinearTaps(float x, float y, int w, int h, int gw,
                                 int gh) {
  float gx = x * gw / w - 0.5f;
  float gy = y * gh / h - 0.5f;
  int i0 = static_cast<int>(std::floor(gx));
  int j0 = static_cast<int>(std::floor(gy));
  float fx = gx - i0, fy = gy - j0;
  int i1 = i0 + 1, j1 = j0 + 1;
  i0 = std::min(gw - 1, std::max(0, i0));
  i1 = std::min(gw - 1, std::max(0, i1));
  j0 = std::min(gh - 1, std::max(0, j0));
  j1 = std::min(gh - 1, std::max(0, j1));
  GridTaps t;
  t.cell[0] = j0 * gw + i0;
  t.w[0] = (1 - fx) * (1 - fy);
  t.cell[1] = j0 * gw + i1;
  t.w[1] = fx * (1 - fy);
  t.cell[2] = j1 * gw + i0;
  t.w[2] = (1 - fx) * fy;
  t.cell[3] = j1 * gw + i1;
  t.w[3] = fx * fy;
  t.n = 4;
  return t;
}

// Bilinear sample of the field grid (per channel) at source pixel (x,y).
inline void GridSample(const float* grid, int w, int h, int gw, int gh, int nch,
                       float x, float y, float out[3]) {
  GridTaps t = GridBilinearTaps(x, y, w, h, gw, gh);
  for (int c = 0; c < nch; ++c) {
    float v = 0.f;
    for (int k = 0; k < t.n; ++k) v += t.w[k] * grid[t.cell[k] * nch + c];
    out[c] = v;
  }
}

// ---- atlas region of a camera band ----------------------------------------
struct Region {
  int ox, oy;  // origin in the atlas
};
inline Region CameraAtlasRegion(int k, int w, int h, int num,
                                AtlasLayout layout) {
  Region r{0, 0};
  if (layout == AtlasLayout::kVertical) {
    r.ox = 0;
    r.oy = k * h;
  } else if (layout == AtlasLayout::kHorizontal) {
    r.ox = k * w;
    r.oy = 0;
  } else {  // kTile
    int x_tile =
        static_cast<int>(std::ceil(std::sqrt(static_cast<float>(num))));
    if (x_tile < 1) x_tile = 1;
    r.ox = (k % x_tile) * w;
    r.oy = (k / x_tile) * h;
  }
  return r;
}

// ---- seam record (cells/weights precomputed for the solver) ----------------
struct Seam {
  int camera_a, camera_b;
  GridTaps taps_a;  // bilinear taps on camera ca grid (side A sample)
  GridTaps taps_b;  // bilinear taps on camera cb grid (side B sample)
  float d[3];       // log residual per solved channel: log(B) - log(A)
  float res_abs;    // |luminance log residual| for stats
  float weight;
};

inline uint64_t EdgeKey(uint32_t a, uint32_t b) {
  uint32_t lo = std::min(a, b), hi = std::max(a, b);
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

}  // namespace

void ApplyColorCorrectionFieldCpu(
    ColorCorrectionContext& context, grtmesh::Mesh& mesh,
    const std::vector<grtmesh::RgbdFramePtr>& frames,
    const ColorCorrectionConfig& config, AtlasLayout layout,
    ColorCorrectionStats& stats) {
  if (config.mode == ColorCorrectionMode::kOff) return;
  const int num = static_cast<int>(frames.size());
  if (num <= 1) return;  // no seams possible
  if (mesh.faces.empty() || mesh.atlas.empty()) return;

  const int w = static_cast<int>(frames[0]->color.width);
  const int h = static_cast<int>(frames[0]->color.height);
  const bool rgb = config.space == ColorCorrectionSpace::kLinearRgb;
  const int nch = rgb ? 3 : 1;
  // kGlobalOnly collapses to a single gain per camera (1x1 field).
  const int gw = config.mode == ColorCorrectionMode::kGlobalOnly
                     ? 1
                     : std::max(1, config.field_grid_w);
  const int gh = config.mode == ColorCorrectionMode::kGlobalOnly
                     ? 1
                     : std::max(1, config.field_grid_h);
  const int cells = gw * gh;

  const auto& faces = mesh.faces;
  const auto& tex_ids = mesh.face_camera_ids;
  const auto& proj = mesh.face_image_coords;
  const int num_faces = static_cast<int>(faces.size());

  stats.face_count = static_cast<uint32_t>(num_faces);

  using Clock = std::chrono::high_resolution_clock;
  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
  };
  auto t0 = Clock::now();

  // --- 0. low-res radiometric proxies (one per camera) ---------------------
  std::vector<Proxy> proxies(num);
  for (int k = 0; k < num; ++k)
    proxies[k] =
        BuildProxy(frames[k]->color, std::max(1, config.proxy_downscale));

  // --- 1. adjacency: edge -> (face, local_edge) ----------------------------
  struct EdgeRef {
    int face;
    uint8_t edge;
    uint8_t count;
  };
  std::unordered_map<uint64_t, EdgeRef> edge_map;
  edge_map.reserve(num_faces * 3);
  uint32_t adjacency_edges = 0, non_manifold = 0, boundary = 0;

  std::vector<Seam> seams;
  seams.reserve(4096);
  const int samples = std::max(1, std::min(32, config.seam_samples));
  const float inset = std::max(0.f, config.sample_inset_proxy_pixels) *
                      std::max(1.f, static_cast<float>(config.proxy_downscale));

  for (int f = 0; f < num_faces; ++f) {
    for (int e = 0; e < 3; ++e) {
      uint32_t va = static_cast<uint32_t>(faces[f][e]);
      uint32_t vb = static_cast<uint32_t>(faces[f][(e + 1) % 3]);
      uint64_t key = EdgeKey(va, vb);
      auto it = edge_map.find(key);
      if (it == edge_map.end()) {
        edge_map.emplace(key, EdgeRef{f, static_cast<uint8_t>(e), 1});
        continue;
      }
      EdgeRef& ref = it->second;
      ref.count++;
      if (ref.count > 2) {
        non_manifold++;
        continue;
      }
      adjacency_edges++;
      const int fa = ref.face, fb = f;
      const int ea = ref.edge, eb = e;
      const int32_t cam_a = tex_ids[fa], cam_b = tex_ids[fb];
      if (cam_a < 0 || cam_b < 0 || cam_a == cam_b) continue;
      if (cam_a >= num || cam_b >= num) continue;
      // Skip sharp folds / veil triangles (occlusion edges): the two faces are
      // not a continuous surface, so matching their colours is meaningless.
      if (!mesh.face_normals.empty() &&
          mesh.face_normals[fa].dot(mesh.face_normals[fb]) <
              config.seam_max_dihedral_cos) {
        continue;
      }

      // Shared global vertices (from face fa's edge ea).
      uint32_t vi = static_cast<uint32_t>(faces[fa][ea]);
      uint32_t vj = static_cast<uint32_t>(faces[fa][(ea + 1) % 3]);
      // Local indices of vi, vj inside each face (to fetch projected coords).
      auto local_of = [&](int face, uint32_t v) -> int {
        for (int k = 0; k < 3; ++k)
          if (static_cast<uint32_t>(faces[face][k]) == v) return k;
        return -1;
      };
      int ai = local_of(fa, vi), aj = local_of(fa, vj);
      int bi = local_of(fb, vi), bj = local_of(fb, vj);
      if (ai < 0 || aj < 0 || bi < 0 || bj < 0) continue;
      int aopp = 3 - ai - aj;  // opposite vertex local index
      int bopp = 3 - bi - bj;

      const Eigen::Vector2f pa_i = proj[fa][ai], pa_j = proj[fa][aj],
                            pa_opp = proj[fa][aopp];
      const Eigen::Vector2f pb_i = proj[fb][bi], pb_j = proj[fb][bj],
                            pb_opp = proj[fb][bopp];
      // Skip inpainted / degenerate faces: their projected triangle is
      // collapsed (all verts ~coincide), so edge sampling reads a single bogus
      // spot and corrupts the residual. Require a non-trivial projected area.
      {
        const float area_a =
            std::abs((pa_j.x() - pa_i.x()) * (pa_opp.y() - pa_i.y()) -
                     (pa_j.y() - pa_i.y()) * (pa_opp.x() - pa_i.x()));
        const float area_b =
            std::abs((pb_j.x() - pb_i.x()) * (pb_opp.y() - pb_i.y()) -
                     (pb_j.y() - pb_i.y()) * (pb_opp.x() - pb_i.x()));
        if (area_a < 1.f || area_b < 1.f) continue;
      }

      // Sample residuals along the shared edge, inset toward each face center.
      // Dark / occluded samples are rejected and the per-sample log ratios are
      // reduced by their median, so a few outliers cannot drag the estimate.
      float v0[32], v1[32], v2[32];
      int valid = 0;
      for (int s = 0; s < samples; ++s) {
        float t = (s + 0.5f) / samples;
        Eigen::Vector2f ea_p = (1 - t) * pa_i + t * pa_j;
        Eigen::Vector2f eb_p = (1 - t) * pb_i + t * pb_j;
        Eigen::Vector2f da = (pa_opp - ea_p);
        Eigen::Vector2f db = (pb_opp - eb_p);
        if (da.norm() > 1e-4f) da.normalize();
        if (db.norm() > 1e-4f) db.normalize();
        Eigen::Vector2f pa = ea_p + da * inset;
        Eigen::Vector2f pb = eb_p + db * inset;
        float color_a[3], color_b[3];
        if (!SampleProxyLinear(proxies[cam_a], pa.x(), pa.y(), color_a))
          continue;
        if (!SampleProxyLinear(proxies[cam_b], pb.x(), pb.y(), color_b))
          continue;
        const float luma_a = Luminance(color_a[0], color_a[1], color_a[2]);
        const float luma_b = Luminance(color_b[0], color_b[1], color_b[2]);
        if (std::min(luma_a, luma_b) < 0.01f) continue;
        if (rgb) {
          v0[valid] = std::log(std::max(color_b[0], kEps)) -
                      std::log(std::max(color_a[0], kEps));
          v1[valid] = std::log(std::max(color_b[1], kEps)) -
                      std::log(std::max(color_a[1], kEps));
          v2[valid] = std::log(std::max(color_b[2], kEps)) -
                      std::log(std::max(color_a[2], kEps));
        } else {
          v0[valid] = std::log(luma_b) - std::log(luma_a);
        }
        valid++;
      }
      if (valid == 0) {
        stats.invalid_seam_sample_count++;
        continue;
      }

      Seam seam;
      seam.camera_a = cam_a;
      seam.camera_b = cam_b;
      // Representative sample = edge midpoint inset (cell taps for the solver).
      {
        float t = 0.5f;
        Eigen::Vector2f ea_p = (1 - t) * pa_i + t * pa_j;
        Eigen::Vector2f eb_p = (1 - t) * pb_i + t * pb_j;
        Eigen::Vector2f da = (pa_opp - ea_p);
        Eigen::Vector2f db = (pb_opp - eb_p);
        if (da.norm() > 1e-4f) da.normalize();
        if (db.norm() > 1e-4f) db.normalize();
        Eigen::Vector2f pa = ea_p + da * inset;
        Eigen::Vector2f pb = eb_p + db * inset;
        seam.taps_a = GridNearestTap(pa.x(), pa.y(), w, h, gw, gh);
        seam.taps_b = GridNearestTap(pb.x(), pb.y(), w, h, gw, gh);
      }
      seam.d[0] = MedianInplace(v0, valid);
      seam.d[1] = rgb ? MedianInplace(v1, valid) : 0.f;
      seam.d[2] = rgb ? MedianInplace(v2, valid) : 0.f;
      seam.res_abs = std::abs(seam.d[0]);
      seam.weight = static_cast<float>(valid) / samples;
      seams.push_back(seam);
    }
  }

  // Account boundary edges (count==1) for stats.
  for (const auto& kv : edge_map)
    if (kv.second.count == 1) boundary++;
  stats.adjacency_edge_count = adjacency_edges;
  stats.seam_edge_count = static_cast<uint32_t>(seams.size());
  stats.non_manifold_edge_count = non_manifold;
  stats.boundary_edge_count = boundary;

  if (seams.empty()) return;

  double sum_before = 0;
  for (const auto& s : seams) sum_before += s.res_abs;
  stats.mean_abs_seam_residual_before =
      static_cast<float>(sum_before / seams.size());

  auto t_detect = Clock::now();

  // --- 2. solve per-camera grids (matrix-free weighted Jacobi) -------------
  const int total = num * cells * nch;
  std::vector<float> g(total, 0.f), acc_num(total, 0.f), acc_den(total, 0.f);
  auto idx = [&](int cam, int cell, int ch) {
    return (cam * cells + cell) * nch + ch;
  };
  const float ls = std::max(0.f, config.lambda_smooth);
  const float la = std::max(0.f, config.lambda_anchor);
  const float omega = config.field_omega <= 0.f ? 0.7f : config.field_omega;
  const float max_g = std::max(0.f, config.max_log_gain);
  const float dw = std::max(0.f, config.seam_data_weight);
  const int iters = std::max(1, config.field_iterations);

  for (int it = 0; it < iters; ++it) {
    std::fill(acc_num.begin(), acc_num.end(), 0.f);
    std::fill(acc_den.begin(), acc_den.end(), 0.f);

    // Laplacian (per camera, 4-neighbour) + identity anchor.
    if (cells > 1 && ls > 0.f) {
      for (int k = 0; k < num; ++k)
        for (int j = 0; j < gh; ++j)
          for (int i = 0; i < gw; ++i) {
            int c = j * gw + i;
            for (int ch = 0; ch < nch; ++ch) {
              float num_v = 0.f, den_v = 0.f;
              const int di[4] = {-1, 1, 0, 0};
              const int dj[4] = {0, 0, -1, 1};
              for (int n = 0; n < 4; ++n) {
                int ni = i + di[n], nj = j + dj[n];
                if (ni < 0 || ni >= gw || nj < 0 || nj >= gh) continue;
                num_v += ls * g[idx(k, nj * gw + ni, ch)];
                den_v += ls;
              }
              acc_num[idx(k, c, ch)] += num_v;
              acc_den[idx(k, c, ch)] += den_v;
            }
          }
    }
    for (int t = 0; t < total; ++t) acc_den[t] += la;  // anchor toward 0

    // Seam data term.
    for (const auto& s : seams) {
      for (int ch = 0; ch < nch; ++ch) {
        float pred_a = 0.f, pred_b = 0.f;
        for (int n = 0; n < s.taps_a.n; ++n)
          pred_a += s.taps_a.w[n] * g[idx(s.camera_a, s.taps_a.cell[n], ch)];
        for (int m = 0; m < s.taps_b.n; ++m)
          pred_b += s.taps_b.w[m] * g[idx(s.camera_b, s.taps_b.cell[m], ch)];
        // Constraint: g_a - g_b = d  (d = logY_B - logY_A), so corrected
        // luminances match: (logY_A + g_a) == (logY_B + g_b).
        const float d = s.d[ch];
        for (int n = 0; n < s.taps_a.n; ++n) {
          float wn = s.taps_a.w[n];
          if (wn <= 0.f) continue;
          int ix = idx(s.camera_a, s.taps_a.cell[n], ch);
          // target: predA -> predB + d
          float rest = pred_b + d - (pred_a - wn * g[ix]);
          acc_num[ix] += dw * s.weight * wn * rest;
          acc_den[ix] += dw * s.weight * wn * wn;
        }
        for (int m = 0; m < s.taps_b.n; ++m) {
          float wm = s.taps_b.w[m];
          if (wm <= 0.f) continue;
          int ix = idx(s.camera_b, s.taps_b.cell[m], ch);
          // target: predB -> predA - d
          float rest = pred_a - d - (pred_b - wm * g[ix]);
          acc_num[ix] += dw * s.weight * wm * rest;
          acc_den[ix] += dw * s.weight * wm * wm;
        }
      }
    }

    // Update.
    for (int t = 0; t < total; ++t) {
      float cand = acc_num[t] / std::max(acc_den[t], kEps);
      float v = (1.f - omega) * g[t] + omega * cand;
      if (max_g > 0.f) v = std::min(max_g, std::max(-max_g, v));
      g[t] = v;
    }

    // Gauge fix: pin the brightness null-space so the solution is well-defined.
    //  reference_camera < 0 : zero-mean over all cameras (symmetric; no single
    //                         camera is preserved).
    //  reference_camera >=0 : HARD-pin that camera to identity (g == 0 on all
    //                         its cells); every correction accrues to the other
    //                         cameras, so the reference color is fully fixed.
    if (config.reference_camera >= 0 && config.reference_camera < num) {
      const int ref = config.reference_camera;
      for (int ch = 0; ch < nch; ++ch)
        for (int c = 0; c < cells; ++c) g[idx(ref, c, ch)] = 0.f;
    } else {
      for (int ch = 0; ch < nch; ++ch) {
        float mean = 0.f;
        for (int k = 0; k < num; ++k)
          for (int c = 0; c < cells; ++c) mean += g[idx(k, c, ch)];
        mean /= (num * cells);
        for (int k = 0; k < num; ++k)
          for (int c = 0; c < cells; ++c) g[idx(k, c, ch)] -= mean;
      }
    }
  }

  // --- residual after (for stats) ------------------------------------------
  {
    // Remaining residual in solved channel 0 = |d - (predA - predB)|
    // (the seam matches when g_a - g_b == d).
    double sum_after = 0;
    for (const auto& s : seams) {
      float pred_a = 0.f, pred_b = 0.f;
      for (int n = 0; n < s.taps_a.n; ++n)
        pred_a += s.taps_a.w[n] * g[idx(s.camera_a, s.taps_a.cell[n], 0)];
      for (int m = 0; m < s.taps_b.n; ++m)
        pred_b += s.taps_b.w[m] * g[idx(s.camera_b, s.taps_b.cell[m], 0)];
      sum_after += std::abs(s.d[0] - (pred_a - pred_b));
    }
    stats.mean_abs_seam_residual_after =
        static_cast<float>(sum_after / seams.size());
  }

  auto t_solve = Clock::now();

  // --- 3. bake the field into each camera's atlas band ---------------------
  grtmesh::ColorImage& atlas = mesh.atlas;
  const int aw = static_cast<int>(atlas.width);
  if (!config.cpu_bake) {
    float gpu_ms =
        context.Bake(atlas, g, w, h, num, gw, gh, nch, rgb, max_g, layout);
    if (gpu_ms >= 0.f) {
      auto t_bake_gpu = Clock::now();
      stats.seam_extract_ms = ms(t0, t_detect);
      stats.field_solve_ms = ms(t_detect, t_solve);
      stats.bake_ms = ms(t_solve, t_bake_gpu);
      return;  // GPU bake done
    }
    // A failed GPU download may have modified only part of the atlas.
    // Always apply the CPU field to the original source colors exactly once.
    std::vector<grtmesh::ColorImage> images(num);
    for (int k = 0; k < num; ++k) images[k] = frames[k]->color;
    PackAtlas(images, atlas, layout);
  }
  for (int k = 0; k < num; ++k) {
    const float* grid = &g[idx(k, 0, 0)];
    Region reg = CameraAtlasRegion(k, w, h, num, layout);
    for (int y = 0; y < h; ++y) {
      int ay = reg.oy + y;
      for (int x = 0; x < w; ++x) {
        int ax = reg.ox + x;
        uint8_t* px = &atlas.data()[3 * (ay * aw + ax)];
        float gain[3];
        GridSample(grid, w, h, gw, gh, nch, static_cast<float>(x),
                   static_cast<float>(y), gain);
        float lin[3];
        for (int c = 0; c < 3; ++c) lin[c] = SrgbToLinear(px[c]);
        if (rgb) {
          for (int c = 0; c < 3; ++c) lin[c] *= std::exp(gain[c]);
        } else {
          float s = std::exp(gain[0]);
          for (int c = 0; c < 3; ++c) lin[c] *= s;
        }
        for (int c = 0; c < 3; ++c) px[c] = LinearToSrgb(lin[c]);
      }
    }
  }
  auto t_bake = Clock::now();

  stats.seam_extract_ms = ms(t0, t_detect);  // proxy + adjacency + residual
  stats.field_solve_ms = ms(t_detect, t_solve);
  stats.bake_ms = ms(t_solve, t_bake);
}

}  // namespace grtmesh::tex
