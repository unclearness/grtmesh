// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_APP_PANOPTIC_DATASET_H_
#define GRTMESH_APP_PANOPTIC_DATASET_H_

#include <filesystem>
#include <utility>
#include <vector>

#include "app/util.h"

namespace grtmesh::app {

// CMU Panoptic (Kinoptic) Kinect data as exported by
// python/panoptic_prepare.py:
//
//   <frames_dir>/hd_NNNNNNNN/kKK_rgb.jpg     color (1920x1080)
//   <frames_dir>/hd_NNNNNNNN/kKK_depth.png   uint16 depth in mm (512x424)
//   <calib_dir>/kcalibration_<seq>.json      Kinect intrinsics, distortion
//                                            and poses
//
// Frames are addressed by their HD frame number NNNNNNNN.
class PanopticKinopticDataset final : public MultiViewDataset {
 public:
  struct Options {
    // Kinect ids to load, e.g. {2, 4, 6, 8, 10} for the held-out protocol
    // (reconstruct from the even Kinects, evaluate on the odd ones). Empty =
    // every Kinect in the calibration. The NUM_KINECTS environment variable
    // (first N Kinects {1..N}) overrides this list when set > 0.
    std::vector<int> kinect_ids;
  };

  explicit PanopticKinopticDataset(Options options)
      : options_(std::move(options)) {}

  // Throws std::runtime_error when the calibration or the frames are missing.
  void Load(const std::filesystem::path& frames_dir,
            const std::filesystem::path& calib_dir);

 private:
  Options options_;
};

}  // namespace grtmesh::app

#endif  // GRTMESH_APP_PANOPTIC_DATASET_H_
