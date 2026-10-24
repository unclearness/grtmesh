// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// cv::initUndistortRectifyMap(K, distortion, I, K, size, CV_16SC2) of OpenCV
// 4.x (modules/calib3d/src/undistort.simd.hpp), operation for operation, so
// the maps are bit-identical to the ones the published CPU preprocessing used.
// OpenCV dispatches that function to AVX2 on current x86-64 CPUs: a
// double-precision loop over blocks of 2 x 4 pixels whose v_muladd is a fused
// multiply-add, followed by a scalar loop over the pixels left at the end of
// a row. Both are reproduced below; the rational-model distortion has no thin
// prism (s1..s4) or tilt terms, which vanish exactly in OpenCV as well.
// The scalar tail must not be contracted into FMAs (CMake passes
// -ffp-contract=off to GCC/Clang for this file; MSVC does not contract).

#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

#include "grtmesh/input_preprocessor.h"

namespace grtmesh {
namespace {

constexpr int kInterBits = 5;
constexpr int kInterTabSize = 1 << kInterBits;
constexpr int kBlock = 8;  // 2 AVX2 vectors of 4 doubles

// cv::invert of a 3x3 double matrix (its closed form for n <= 3).
void Invert3x3(const double m[9], double t[9]) {
  auto a = [m](int r, int c) { return m[3 * r + c]; };
  double d = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
             a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
             a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
  d = 1. / d;
  t[0] = (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * d;
  t[1] = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * d;
  t[2] = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * d;
  t[3] = (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * d;
  t[4] = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * d;
  t[5] = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * d;
  t[6] = (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * d;
  t[7] = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * d;
  t[8] = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * d;
}

// _mm256_cvtpd_epi32 / cvRound: round half to even; out of range -> INT_MIN.
int RoundToInt(double v) {
  const double r = std::nearbyint(v);
  if (!(r >= static_cast<double>(INT_MIN) &&
        r <= static_cast<double>(INT_MAX))) {
    return INT_MIN;
  }
  return static_cast<int>(r);
}

int16_t SaturateInt16(int v) {
  return static_cast<int16_t>(v < INT16_MIN   ? INT16_MIN
                              : v > INT16_MAX ? INT16_MAX
                                              : v);
}

struct Model {
  double fx, fy, u0, v0;
  double k1, k2, p1, p2, k3, k4, k5, k6;
};

}  // namespace

void BuildUndistortMaps(const CameraCalibration& calibration,
                        std::vector<int16_t>& map_xy,
                        std::vector<uint16_t>& map_frac) {
  const int width = static_cast<int>(calibration.width);
  const int height = static_cast<int>(calibration.height);
  map_xy.assign(static_cast<size_t>(width) * height * 2, 0);
  map_frac.assign(static_cast<size_t>(width) * height, 0);

  const CameraIntrinsics& k = calibration.intrinsics;
  const double camera[9] = {k.fx, 0., k.cx, 0., k.fy, k.cy, 0., 0., 1.};
  double ir[9];
  Invert3x3(camera, ir);  // inverse of the new camera matrix (= K) times I
  const auto& d = calibration.distortion;
  const Model m{camera[0], camera[4], camera[2], camera[5], d[0], d[1],
                d[2],      d[3],      d[4],      d[5],      d[6], d[7]};
  double s_x[kBlock], s_y[kBlock], s_w[kBlock];
  for (int i = 0; i < kBlock; ++i) {
    s_x[i] = ir[0] * i;
    s_y[i] = ir[3] * i;
    s_w[i] = ir[6] * i;
  }

  for (int i = 0; i < height; ++i) {
    int16_t* xy = map_xy.data() + static_cast<size_t>(i) * width * 2;
    uint16_t* frac = map_frac.data() + static_cast<size_t>(i) * width;
    double _x = i * ir[1] + ir[2], _y = i * ir[4] + ir[5],
           _w = i * ir[7] + ir[8];
    int j = 0;
    // Vector path: lane-wise, with fused multiply-adds where OpenCV has
    // v_muladd.
    for (; j <= width - kBlock; j += kBlock, _x += kBlock * ir[0],
                                _y += kBlock * ir[3], _w += kBlock * ir[6]) {
      for (int l = 0; l < kBlock; ++l) {
        const double w = 1.0 / (_w + s_w[l]);
        double x = (_x + s_x[l]) * w;
        double y = (_y + s_y[l]) * w;
        double xd = x * x;
        double yd = y * y;
        const double r2 = xd + yd;
        double kr =
            std::fma(std::fma(std::fma(m.k3, r2, m.k2), r2, m.k1), r2, 1.0);
        kr = kr /
             std::fma(std::fma(std::fma(m.k6, r2, m.k5), r2, m.k4), r2, 1.0);
        xd = std::fma(2.0, xd, r2);
        yd = std::fma(2.0, yd, r2);
        const double xy2 = (x * y) * 2.0;
        x = x * kr;
        y = y * kr;
        xd = std::fma(xd, m.p2, x);
        yd = std::fma(yd, m.p1, y);
        xd = std::fma(m.p1, xy2, xd);
        yd = std::fma(m.p2, xy2, yd);
        // Thin prism and tilt are zero: xd, yd and the unit projective scale
        // pass through unchanged.
        const double u = std::fma(m.fx, xd, m.u0) * kInterTabSize;
        const double v = std::fma(m.fy, yd, m.v0) * kInterTabSize;
        const int iu = RoundToInt(u);
        const int iv = RoundToInt(v);
        frac[j + l] =
            static_cast<uint16_t>((iu & (kInterTabSize - 1)) +
                                  (iv & (kInterTabSize - 1)) * kInterTabSize);
        xy[2 * (j + l)] = SaturateInt16(iu >> kInterBits);
        xy[2 * (j + l) + 1] = SaturateInt16(iv >> kInterBits);
      }
    }
    // Scalar tail.
    for (; j < width; j++, _x += ir[0], _y += ir[3], _w += ir[6]) {
      const double w = 1. / _w, x = _x * w, y = _y * w;
      const double x2 = x * x, y2 = y * y;
      const double r2 = x2 + y2, _2xy = 2 * x * y;
      const double kr = (1 + ((m.k3 * r2 + m.k2) * r2 + m.k1) * r2) /
                        (1 + ((m.k6 * r2 + m.k5) * r2 + m.k4) * r2);
      const double xd = (x * kr + m.p1 * _2xy + m.p2 * (r2 + 2 * x2));
      const double yd = (y * kr + m.p1 * (r2 + 2 * y2) + m.p2 * _2xy);
      const double u = m.fx * xd + m.u0;
      const double v = m.fy * yd + m.v0;
      const int iu = RoundToInt(u * static_cast<double>(kInterTabSize));
      const int iv = RoundToInt(v * static_cast<double>(kInterTabSize));
      xy[2 * j] = static_cast<int16_t>(iu >> kInterBits);
      xy[2 * j + 1] = static_cast<int16_t>(iv >> kInterBits);
      frac[j] =
          static_cast<uint16_t>((iv & (kInterTabSize - 1)) * kInterTabSize +
                                (iu & (kInterTabSize - 1)));
    }
  }
}

}  // namespace grtmesh
