// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_APP_RGBD_SEQUENCE_DATASET_H_
#define GRTMESH_APP_RGBD_SEQUENCE_DATASET_H_

#include <filesystem>
#include <string>
#include <vector>

#include "app/util.h"

namespace grtmesh::app {

// A calibrated multi-view RGB-D sequence with depth registered to color (one
// pinhole camera per sensor), e.g. the synthetic sequence rendered by
// python/synthetic/. Layout:
//
//   <root>/cameras.json
//   <root>/frames/NNNNNN/<camera>_color.{png,jpg}   8-bit color
//   <root>/frames/NNNNNN/<camera>_depth.png         uint16 depth
//
// cameras.json:
//
//   {
//     "depth_scale": 1000.0,          // depth PNG values per meter
//     "bounds": {"min": [x, y, z],    // reconstruction volume (world,
//                "max": [x, y, z]},   // meters)
//     "cameras": [
//       {"name": "cam00", "width": 1280, "height": 720,
//        "fx": ..., "fy": ..., "cx": ..., "cy": ...,
//        "distortion": [k1, k2, p1, p2, k3],    // optional, default 0
//        "camera_to_world": [[...], [...], [...], [0, 0, 0, 1]]},
//       ...
//     ]
//   }
//
// Camera axes follow OpenCV (x right, y down, z forward); depth is the z
// coordinate in the camera frame (not the ray length), 0 = invalid.
class RgbdSequenceDataset final : public MultiViewDataset {
 public:
  struct Camera {
    std::string name;
    CameraCalibration calibration;
    Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
  };

  // Loads cameras.json and indexes frames/. `camera_names` selects and
  // orders the cameras (empty = all, in cameras.json order). Throws
  // std::runtime_error on missing files or unknown camera names.
  void Load(const std::filesystem::path& root,
            const std::vector<std::string>& camera_names = {});

  const std::vector<Camera>& cameras() const { return cameras_; }
  const Eigen::Vector3f& bounds_min() const { return bounds_min_; }
  const Eigen::Vector3f& bounds_max() const { return bounds_max_; }

 private:
  std::vector<Camera> cameras_;  // the selected cameras
  Eigen::Vector3f bounds_min_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f bounds_max_ = Eigen::Vector3f::Zero();
};

}  // namespace grtmesh::app

#endif  // GRTMESH_APP_RGBD_SEQUENCE_DATASET_H_
