// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_INCLUDE_GRTMESH_FRAME_H_
#define GRTMESH_INCLUDE_GRTMESH_FRAME_H_

#include <cstdint>
#include <memory>

#include "Eigen/Geometry"
#include "grtmesh/camera.h"
#include "grtmesh/image.h"

namespace grtmesh {

// One sensor's input for one frame. Depth is in any consistent metric unit
// (0 = invalid); the volume, the voxel size and the occlusion threshold use
// the same unit. depth_in_color is the depth reprojected into the color image
// (same size as `color`) and feeds the per-face visibility test.
struct RgbdFrame {
  ColorImage color;
  CameraIntrinsics color_intrinsics;
  Eigen::Affine3f color_to_world = Eigen::Affine3f::Identity();
  DepthImage depth_in_color;

  DepthImage depth;
  CameraIntrinsics depth_intrinsics;
  Eigen::Affine3f depth_to_world = Eigen::Affine3f::Identity();
};

using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

// The images of all cameras of one frame already on the device (e.g. written
// there by a GPU preprocessor), so Mesher::Process reads them in place instead
// of uploading the host images. Each array holds the cameras one after the
// other in the order of the frames: color width x height x 3 (BGR8), depth
// and depth_in_color as in RgbdFrame. The host images of the frames then only
// need their width and height. The memory must stay valid and unchanged until
// Process returns.
struct DeviceRgbdBatch {
  const uint8_t* color = nullptr;
  const float* depth = nullptr;
  const float* depth_in_color = nullptr;
};

}  // namespace grtmesh

#endif  // GRTMESH_INCLUDE_GRTMESH_FRAME_H_
