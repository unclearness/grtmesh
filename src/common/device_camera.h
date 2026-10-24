// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Camera record read by the CUDA kernels.

#ifndef GRTMESH_SRC_COMMON_DEVICE_CAMERA_H_
#define GRTMESH_SRC_COMMON_DEVICE_CAMERA_H_

namespace grtmesh {

// Pinhole intrinsics in pixels plus the world-to-camera rigid transform
// (row-major rotation and translation) and the camera's viewing direction in
// world space (the +Z axis of camera-to-world; OpenCV convention). One array
// of these per Mesher instance is passed to the kernels as an argument.
struct DeviceCamera {
  float fx, fy, cx, cy;
  float r[9];
  float t[3];
  float dir[3];
};

}  // namespace grtmesh

#endif  // GRTMESH_SRC_COMMON_DEVICE_CAMERA_H_
