// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_INCLUDE_GRTMESH_INPUT_PREPROCESSOR_H_
#define GRTMESH_INCLUDE_GRTMESH_INPUT_PREPROCESSOR_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "Eigen/Geometry"
#include "grtmesh/camera.h"
#include "grtmesh/frame.h"

namespace grtmesh {

// Size of the depth-to-color z-buffer relative to the color image (the
// published setting); it is upsampled to the color size with nearest
// neighbor.
inline constexpr float kDepthInColorScale = 0.35f;

// Fixed-point undistortion maps of `calibration`, identical to
// cv::initUndistortRectifyMap(K, distortion, I, K, size, CV_16SC2) of OpenCV
// 4.x on x86-64 with AVX2 (its double-precision vector path with fused
// multiply-add, which the published results used): per pixel the integer
// source pixel (map_xy, x then y) and the 5-bit sub-pixel index (map_frac,
// fy * 32 + fx). Row-major, width x height entries (2x for map_xy).
void BuildUndistortMaps(const CameraCalibration& calibration,
                        std::vector<int16_t>& map_xy,
                        std::vector<uint16_t>& map_frac);

// One sensor's frame as captured: the decoded color and depth images before
// undistortion, in host memory, row-major and contiguous.
struct SensorFrame {
  const uint8_t* color = nullptr;  // BGR8, color_calibration size
  const float* depth = nullptr;    // depth_calibration size, 0 = invalid
  CameraCalibration color_calibration;
  CameraCalibration depth_calibration;
  Eigen::Affine3f color_to_world = Eigen::Affine3f::Identity();
  Eigen::Affine3f depth_to_world = Eigen::Affine3f::Identity();
  Eigen::Affine3f depth_to_color = Eigen::Affine3f::Identity();
};

// Input preprocessing on the GPU: the sensor images of all cameras are
// uploaded in one batch, undistorted (bilinear color, nearest-neighbor
// depth), cut at the far threshold, and the depth is projected into the
// color image (z-buffer at kDepthInColorScale, nearest-neighbor upsampling).
// The results stay on the device (DeviceFrames, read in place by
// Mesher::Process as a DeviceRgbdBatch) or are downloaded into host images.
// The output matches the OpenCV CPU reference of the sample apps
// (app/util.h, UndistortPreprocessor) pixel for pixel, except where that
// path's parallel z-buffer races.
class InputPreprocessor {
 public:
  struct Options {
    float depth_far_threshold = 6000.f;  // depth units
    // OpenMP threads of the host copy into pinned staging memory (0 = the
    // OpenMP default).
    int staging_threads = 0;
  };

  // Time of the last Run in ms: host copy of the sensor images into pinned
  // staging memory, then (CUDA events, one stream) the host-to-device copy,
  // the kernels and, for the host-output Run, the device-to-host copy.
  struct Timing {
    double staging_ms = 0;
    double upload_ms = 0;
    double kernel_ms = 0;
    double download_ms = 0;
    double total_ms = 0;  // wall time of Run
  };

  // The preprocessed images of one frame on the device, all cameras back to
  // back (the DeviceRgbdBatch layout). Allocated by the first Run into it.
  class DeviceFrames {
   public:
    DeviceFrames();
    ~DeviceFrames();
    DeviceFrames(const DeviceFrames&) = delete;
    DeviceFrames& operator=(const DeviceFrames&) = delete;
    DeviceRgbdBatch batch() const;

   private:
    friend class InputPreprocessor;
    struct Buffers;
    std::unique_ptr<Buffers> buffers_;
  };

  Options options;

  InputPreprocessor();
  ~InputPreprocessor();
  InputPreprocessor(const InputPreprocessor&) = delete;
  InputPreprocessor& operator=(const InputPreprocessor&) = delete;

  // Builds the undistortion maps (on the host, once) for the cameras of
  // `cameras`; only their calibrations and depth_to_color are read. All
  // cameras must share the color and depth image sizes. Later frames must
  // have the same cameras in the same order.
  bool Init(const std::vector<SensorFrame>& cameras);

  // Preprocesses into `device`. `frames` receive the intrinsics, poses and
  // image sizes, not the pixels.
  bool Run(const std::vector<SensorFrame>& sensor,
           std::vector<RgbdFramePtr>& frames, DeviceFrames& device);

  // Preprocesses and downloads the images into pinned host images of
  // `frames`.
  bool Run(const std::vector<SensorFrame>& sensor,
           std::vector<RgbdFramePtr>& frames);

  const Timing& last_timing() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace grtmesh

#endif  // GRTMESH_INCLUDE_GRTMESH_INPUT_PREPROCESSOR_H_
