// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_INCLUDE_GRTMESH_CAMERA_H_
#define GRTMESH_INCLUDE_GRTMESH_CAMERA_H_

#include <algorithm>
#include <array>
#include <cstdint>

namespace grtmesh {

// Pinhole intrinsics in pixels. The image size comes from the image itself;
// the images reaching Mesher are already undistorted (by InputPreprocessor or
// by the application).
struct CameraIntrinsics {
  float fx = -1.f;
  float fy = -1.f;
  float cx = -1.f;
  float cy = -1.f;
};

// A sensor camera before undistortion: pinhole intrinsics, image size and the
// lens distortion in OpenCV order (k1 k2 p1 p2 k3 k4 k5 k6; all zero = none).
// Undistortion keeps the intrinsics, so `intrinsics` also describes the
// undistorted image.
struct CameraCalibration {
  CameraIntrinsics intrinsics;
  uint32_t width = 0;
  uint32_t height = 0;
  std::array<float, 8> distortion = {};
};

inline bool operator==(const CameraCalibration& a, const CameraCalibration& b) {
  return a.intrinsics.fx == b.intrinsics.fx &&
         a.intrinsics.fy == b.intrinsics.fy &&
         a.intrinsics.cx == b.intrinsics.cx &&
         a.intrinsics.cy == b.intrinsics.cy && a.width == b.width &&
         a.height == b.height && a.distortion == b.distortion;
}

inline bool HasDistortion(const CameraCalibration& calibration) {
  return std::any_of(calibration.distortion.begin(),
                     calibration.distortion.end(),
                     [](float k) { return k != 0.f; });
}

// Degrees -> radians. Deliberately a single float multiply by this constant:
// std::cos(Radians(80.f)) feeds the view-selection threshold and must stay
// bit-identical to the published build.
inline constexpr float Radians(float deg) {
  return deg * static_cast<float>(0.01745329251994329576923690768489);
}

}  // namespace grtmesh

#endif  // GRTMESH_INCLUDE_GRTMESH_CAMERA_H_
