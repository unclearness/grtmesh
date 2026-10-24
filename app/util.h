// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_APP_UTIL_H_
#define GRTMESH_APP_UTIL_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "app/mesh_io.h"
#include "grtmesh/grtmesh.h"
#include "opencv2/core.hpp"

namespace grtmesh::app {

// Wall-clock stopwatch in milliseconds.
class Stopwatch {
 public:
  Stopwatch() : start_(std::chrono::steady_clock::now()) {}
  void Restart() { start_ = std::chrono::steady_clock::now(); }
  double ElapsedMs() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

// Zero-padded decimal string: Zfill(7, 6) == "000007".
std::string Zfill(uint64_t value, int width = 5);

// "2,4,6,8,10" -> {2, 4, 6, 8, 10}. Empty tokens are ignored.
std::vector<int> ParseIntList(const std::string& s);

// "a,b,c" -> {"a", "b", "c"}. Empty tokens are ignored.
std::vector<std::string> ParseStringList(const std::string& s);

// Thread count of the OpenMP loop in UndistortPreprocessor::Run.
extern int preprocess_thread_count;

// Tightly packed copies between cv::Mat and grtmesh::Image (BGR8 / float32).
void CopyFromMat(const cv::Mat3b& mat, grtmesh::ColorImage& color);
void CopyFromMat(const cv::Mat1f& mat, grtmesh::DepthImage& img);
void CopyToMat(cv::Mat3b& mat, const grtmesh::ColorImage& color);

using grtmesh::CameraCalibration;
using grtmesh::HasDistortion;
using grtmesh::kDepthInColorScale;

// Fixed-point undistortion maps of `calibration` built by OpenCV
// (cv::initUndistortRectifyMap with CV_16SC2: map1 integer pixel, map2
// sub-pixel index; the intrinsics are kept). The CPU reference path uses
// them; grtmesh::BuildUndistortMaps reproduces them without OpenCV.
void BuildOpenCvUndistortMaps(const CameraCalibration& calibration,
                              cv::Mat& map1, cv::Mat& map2);

struct CameraCalibrationHash {
  size_t operator()(const CameraCalibration& calibration) const;
};

// One sensor's frame as stored on disk: image paths, calibration and, after
// MultiViewDataset::Get, the pixels.
struct SourceFrame {
  std::filesystem::path color_path;
  std::filesystem::path depth_path;
  cv::Mat3b color;
  cv::Mat1f depth;  // meters, 0 = invalid
  CameraCalibration color_calibration;
  CameraCalibration depth_calibration;
  Eigen::Affine3f color_to_world = Eigen::Affine3f::Identity();
  Eigen::Affine3f depth_to_world = Eigen::Affine3f::Identity();
  Eigen::Affine3f depth_to_color = Eigen::Affine3f::Identity();
};

using SourceFramePtr = std::shared_ptr<SourceFrame>;

// A multi-view RGB-D sequence on disk: color images (8-bit) and uint16 depth
// PNGs, addressed by frame number. Pixels are read on every Get() and not
// cached, so sequences of any length fit in memory.
class MultiViewDataset {
 public:
  virtual ~MultiViewDataset() = default;

  // Frame numbers present on disk, ascending.
  const std::vector<uint32_t>& frame_numbers() const { return frame_numbers_; }
  bool Has(uint32_t frame) const;

  // The views of `frame` with their pixels loaded. Throws if `frame` is not
  // in frame_numbers().
  std::vector<SourceFramePtr> Get(uint32_t frame) const;

 protected:
  // Registers the views (paths + calibration, no pixels) of `frame`; frames
  // must be added in ascending order.
  void AddFrame(uint32_t frame, std::vector<SourceFramePtr> views);

  // Depth PNG values per meter.
  float depth_scale_ = 1000.f;

 private:
  std::vector<uint32_t> frame_numbers_;
  std::vector<std::vector<SourceFramePtr>> views_;
};

// The pixels and calibration of `source` as the library's sensor input
// (grtmesh::InputPreprocessor); the pointers refer to the cv::Mat pixels.
grtmesh::SensorFrame ToSensorFrame(const SourceFrame& source);
std::vector<grtmesh::SensorFrame> ToSensorFrames(
    const std::vector<SourceFramePtr>& source);

// CPU reference of grtmesh::InputPreprocessor (OpenCV, same output):
// undistorts color and depth, drops far depth and projects the depth into the
// color image (RgbdFrame::depth_in_color).
class UndistortPreprocessor {
 public:
  struct Options {
    float depth_far_threshold = 6000.f;  // depth units
  };
  Options options;

  bool Init(const std::vector<SourceFramePtr>& init_frames);
  bool Run(const std::vector<SourceFramePtr>& source,
           std::vector<grtmesh::RgbdFramePtr>& frames);

 private:
  struct UndistortMaps {
    cv::Mat map1;
    cv::Mat map2;
  };
  std::unordered_map<CameraCalibration, UndistortMaps, CameraCalibrationHash>
      maps_;
};

// Applies the color-correction environment variables to `config`:
// CC_TEMPORAL (EMA alpha), CC_DUMP_GAIN_DIR (gain-field dump directory) and
// CC_TRANSFER_PROFILE (print the H2D / compute / D2H split).
void ApplyColorCorrectionEnvironment(
    grtmesh::tex::ColorCorrectionConfig& config);

// Usage text of the --color-correction* flags.
extern const char kColorCorrectionUsage[];

// Parses argv[i] if it is one of the --color-correction* flags (consuming its
// values and advancing `i`). Returns false for any other argument.
bool ParseColorCorrectionFlag(int argc, char* argv[], int& i,
                              grtmesh::tex::ColorCorrectionConfig& config);

}  // namespace grtmesh::app

#endif  // GRTMESH_APP_UTIL_H_
