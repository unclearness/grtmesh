// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "app/util.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "grtmesh/grtmesh.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using grtmesh::CameraIntrinsics;
using grtmesh::ColorImage;
using grtmesh::DepthImage;
using grtmesh::RgbdFrame;
using grtmesh::RgbdFramePtr;

namespace fs = std::filesystem;

namespace {

cv::Mat1d CameraMatrixOf(const grtmesh::app::CameraCalibration& calibration) {
  cv::Mat1d camera_matrix = cv::Mat1d::eye(3, 3);
  camera_matrix.at<double>(0, 0) = calibration.intrinsics.fx;
  camera_matrix.at<double>(1, 1) = calibration.intrinsics.fy;
  camera_matrix.at<double>(0, 2) = calibration.intrinsics.cx;
  camera_matrix.at<double>(1, 2) = calibration.intrinsics.cy;
  return camera_matrix;
}

cv::Mat1d DistortionOf(const grtmesh::app::CameraCalibration& calibration) {
  cv::Mat1d distortion = cv::Mat1d::zeros(8, 1);
  for (int i = 0; i < 8; ++i) {
    distortion.at<double>(i, 0) = calibration.distortion[i];
  }
  return distortion;
}

cv::Mat3b ReadColorBgr(const fs::path& p) {
  cv::Mat img = cv::imread(p.string(), cv::IMREAD_COLOR);
  if (img.empty()) {
    throw std::runtime_error("Failed to read color image: " + p.string());
  }
  if (img.type() != CV_8UC3) {
    img.convertTo(img, CV_8UC3);
  }
  return img;
}

// uint16 depth PNG -> float depth in meters (value / depth_scale, 0 stays 0).
cv::Mat1f ReadDepthMeters(const fs::path& p, float depth_scale) {
  cv::Mat d16 = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
  if (d16.empty()) {
    throw std::runtime_error("Failed to read depth image: " + p.string());
  }
  if (d16.type() != CV_16UC1) {
    throw std::runtime_error("Depth must be CV_16UC1 (16-bit PNG). Got type=" +
                             std::to_string(d16.type()));
  }
  cv::Mat1f df(d16.rows, d16.cols);
  const float inv = (depth_scale > 0.0f) ? (1.0f / depth_scale) : 1.0f;
  for (int y = 0; y < d16.rows; ++y) {
    const auto* src = d16.ptr<uint16_t>(y);
    auto* dst = df.ptr<float>(y);
    for (int x = 0; x < d16.cols; ++x) {
      const uint16_t v = src[x];
      dst[x] = (v == 0) ? 0.0f : (static_cast<float>(v) * inv);
    }
  }
  return df;
}

// Depth reprojected into the color image: a z-buffer at `scale` of the color
// size, then nearest-neighbor upsampling to the color size.
cv::Mat1f ProjectDepthToColor(const cv::Mat1f& depth,
                              const CameraIntrinsics& depth_intrinsics,
                              const CameraIntrinsics& color_intrinsics,
                              const Eigen::Affine3f& depth_to_color,
                              const cv::Size& color_size, float scale) {
  const float r = scale;
  const int h = static_cast<int>(color_size.height * r);
  const int w = static_cast<int>(color_size.width * r);
  cv::Mat1f map_depth_f = cv::Mat1f::zeros(h, w);

  depth.forEach([&](const float& val, const int* yx) {
    if (val < std::numeric_limits<float>::min()) {
      return;
    }

    float d_x = (yx[1] - depth_intrinsics.cx) * val / depth_intrinsics.fx;
    float d_y = (yx[0] - depth_intrinsics.cy) * val / depth_intrinsics.fy;

    Eigen::Vector3f d_p(d_x, d_y, val);

    Eigen::Vector3f c_p = depth_to_color * d_p;

    float c_z = c_p[2];
    float c_x = std::round(c_p[0] / val * color_intrinsics.fx * r +
                           color_intrinsics.cx * r);
    float c_y = std::round(c_p[1] / val * color_intrinsics.fy * r +
                           color_intrinsics.cy * r);

    if (c_y < 0 || h - 1 < c_y || c_x < 0 || w - 1 < c_x) {
      return;
    }

    float& md =
        map_depth_f.at<float>(static_cast<int>(c_y), static_cast<int>(c_x));

    if (md < std::numeric_limits<float>::min() || c_z < md) {
      md = c_z;
    }
  });

  if (r != 1.f) {
    cv::resize(map_depth_f.clone(), map_depth_f, color_size, 0.0, 0.0,
               cv::InterpolationFlags::INTER_NEAREST);
  }
  return map_depth_f;
}

}  // namespace

namespace grtmesh::app {

int preprocess_thread_count = 2;

void BuildOpenCvUndistortMaps(const CameraCalibration& calibration,
                              cv::Mat& map1, cv::Mat& map2) {
  const cv::Mat1d camera_matrix = CameraMatrixOf(calibration);
  cv::initUndistortRectifyMap(camera_matrix, DistortionOf(calibration),
                              cv::Mat1d::eye(3, 3), camera_matrix,
                              cv::Size(static_cast<int>(calibration.width),
                                       static_cast<int>(calibration.height)),
                              CV_16SC2, map1, map2);
}

std::string Zfill(uint64_t value, int width) {
  std::ostringstream sout;
  sout << std::setfill('0') << std::setw(width) << value;
  return sout.str();
}

std::vector<std::string> ParseStringList(const std::string& s) {
  std::vector<std::string> values;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      values.push_back(token);
    }
  }
  return values;
}

std::vector<int> ParseIntList(const std::string& s) {
  std::vector<int> values;
  for (const std::string& token : ParseStringList(s)) {
    values.push_back(std::stoi(token));
  }
  return values;
}

void CopyFromMat(const cv::Mat3b& mat, ColorImage& color) {
  if (static_cast<uint32_t>(mat.cols) != color.width ||
      static_cast<uint32_t>(mat.rows) != color.height || color.empty()) {
    color = ColorImage::Create(mat.cols, mat.rows);
  }
  std::memcpy(color.data(), mat.data, 3 * mat.cols * mat.rows);
}

void CopyFromMat(const cv::Mat1f& mat, DepthImage& img) {
  if (static_cast<uint32_t>(mat.cols) != img.width ||
      static_cast<uint32_t>(mat.rows) != img.height || img.empty()) {
    img = DepthImage::Create(mat.cols, mat.rows);
  }
  std::memcpy(img.data(), mat.data, sizeof(float) * mat.cols * mat.rows);
}

void CopyToMat(cv::Mat3b& mat, const ColorImage& color) {
  mat = cv::Mat3b::zeros(color.height, color.width);
  std::memcpy(mat.data, color.data(), 3 * mat.cols * mat.rows);
}

size_t CameraCalibrationHash::operator()(
    const CameraCalibration& calibration) const {
  size_t seed = 0;
  auto combine = [&seed](auto value) {
    seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
  };
  combine(calibration.intrinsics.fx);
  combine(calibration.intrinsics.fy);
  combine(calibration.intrinsics.cx);
  combine(calibration.intrinsics.cy);
  combine(calibration.width);
  combine(calibration.height);
  for (float k : calibration.distortion) {
    combine(k);
  }
  return seed;
}

// -----------------------------
// MultiViewDataset
// -----------------------------
bool MultiViewDataset::Has(uint32_t frame) const {
  return std::binary_search(frame_numbers_.begin(), frame_numbers_.end(),
                            frame);
}

void MultiViewDataset::AddFrame(uint32_t frame,
                                std::vector<SourceFramePtr> views) {
  if (!frame_numbers_.empty() && frame <= frame_numbers_.back()) {
    throw std::logic_error("MultiViewDataset::AddFrame: frames out of order");
  }
  frame_numbers_.push_back(frame);
  views_.push_back(std::move(views));
}

grtmesh::SensorFrame ToSensorFrame(const SourceFrame& source) {
  // The library takes the image size from the calibration; a mismatching or
  // strided image is passed as missing (Run then fails).
  auto fits = [](const cv::Mat& image, const CameraCalibration& calibration) {
    return image.isContinuous() &&
           image.cols == static_cast<int>(calibration.width) &&
           image.rows == static_cast<int>(calibration.height);
  };
  grtmesh::SensorFrame sensor;
  sensor.color = fits(source.color, source.color_calibration)
                     ? source.color.data
                     : nullptr;
  sensor.depth = fits(source.depth, source.depth_calibration)
                     ? reinterpret_cast<const float*>(source.depth.data)
                     : nullptr;
  sensor.color_calibration = source.color_calibration;
  sensor.depth_calibration = source.depth_calibration;
  sensor.color_to_world = source.color_to_world;
  sensor.depth_to_world = source.depth_to_world;
  sensor.depth_to_color = source.depth_to_color;
  return sensor;
}

std::vector<grtmesh::SensorFrame> ToSensorFrames(
    const std::vector<SourceFramePtr>& source) {
  std::vector<grtmesh::SensorFrame> sensor;
  sensor.reserve(source.size());
  for (const SourceFramePtr& s : source) sensor.push_back(ToSensorFrame(*s));
  return sensor;
}

std::vector<SourceFramePtr> MultiViewDataset::Get(uint32_t frame) const {
  const auto it =
      std::lower_bound(frame_numbers_.begin(), frame_numbers_.end(), frame);
  if (it == frame_numbers_.end() || *it != frame) {
    throw std::out_of_range("Frame " + std::to_string(frame) +
                            " is not in the dataset");
  }
  std::vector<SourceFramePtr> frames;
  for (const SourceFramePtr& view : views_[it - frame_numbers_.begin()]) {
    auto loaded = std::make_shared<SourceFrame>(*view);
    loaded->color = ReadColorBgr(view->color_path);
    loaded->depth = ReadDepthMeters(view->depth_path, depth_scale_);
    frames.push_back(std::move(loaded));
  }
  return frames;
}

// -----------------------------
// UndistortPreprocessor
// -----------------------------
bool UndistortPreprocessor::Init(
    const std::vector<SourceFramePtr>& init_frames) {
  maps_.clear();

  for (const auto& frame : init_frames) {
    for (const CameraCalibration& calibration :
         {frame->depth_calibration, frame->color_calibration}) {
      if (maps_.count(calibration) != 0 || !HasDistortion(calibration)) {
        continue;
      }
      UndistortMaps maps;
      BuildOpenCvUndistortMaps(calibration, maps.map1, maps.map2);
      maps_[calibration] = maps;
    }
  }

  return true;
}

bool UndistortPreprocessor::Run(const std::vector<SourceFramePtr>& source,
                                std::vector<RgbdFramePtr>& frames) {
  if (frames.size() != source.size()) {
    frames.resize(source.size());
    for (auto& frame : frames) {
      frame = std::make_shared<RgbdFrame>();
    }
  }

#pragma omp parallel for num_threads(preprocess_thread_count)
  for (int64_t f_idx = 0; f_idx < static_cast<int64_t>(source.size());
       f_idx++) {
    const SourceFrame& cv_frame = *source[f_idx];
    RgbdFrame& frame = *frames[f_idx];

    // Undistortion keeps the intrinsics; the distortion is gone.
    frame.color_intrinsics = cv_frame.color_calibration.intrinsics;
    frame.color_to_world = cv_frame.color_to_world;
    frame.depth_intrinsics = cv_frame.depth_calibration.intrinsics;
    frame.depth_to_world = cv_frame.depth_to_world;

    // Cameras without lens distortion are used as they are.
    cv::Mat3b c_undistorted;
    if (HasDistortion(cv_frame.color_calibration)) {
      const UndistortMaps& color_maps = maps_.at(cv_frame.color_calibration);
      cv::remap(cv_frame.color, c_undistorted, color_maps.map1, color_maps.map2,
                cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
    } else {
      c_undistorted = cv_frame.color;
    }

    cv::Mat1f d_undistorted;
    if (HasDistortion(cv_frame.depth_calibration)) {
      const UndistortMaps& depth_maps = maps_.at(cv_frame.depth_calibration);
      cv::remap(cv_frame.depth, d_undistorted, depth_maps.map1, depth_maps.map2,
                cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    } else {
      d_undistorted = cv_frame.depth.clone();
    }
    d_undistorted.forEach([&](float& val, const int* yx) {
      (void)yx;
      if (val > options.depth_far_threshold) {
        val = 0;
      }
    });

    CopyFromMat(c_undistorted, frame.color);
    CopyFromMat(d_undistorted, frame.depth);

    // Depth registered to the color camera (same calibration and pose) is
    // already the depth in color; otherwise reproject it.
    if (cv_frame.depth_calibration == cv_frame.color_calibration &&
        cv_frame.depth_to_color.matrix() == Eigen::Matrix4f::Identity()) {
      CopyFromMat(d_undistorted, frame.depth_in_color);
      continue;
    }
    const cv::Mat1f depth_in_color = ProjectDepthToColor(
        d_undistorted, frame.depth_intrinsics, frame.color_intrinsics,
        cv_frame.depth_to_color,
        cv::Size(static_cast<int>(cv_frame.color_calibration.width),
                 static_cast<int>(cv_frame.color_calibration.height)),
        kDepthInColorScale);
    CopyFromMat(depth_in_color, frame.depth_in_color);
  }
  return true;
}

// -----------------------------
// Color correction options
// -----------------------------
void ApplyColorCorrectionEnvironment(
    grtmesh::tex::ColorCorrectionConfig& config) {
  if (const char* alpha = std::getenv("CC_TEMPORAL")) {
    config.temporal_alpha = static_cast<float>(std::atof(alpha));
  }
  if (const char* dir = std::getenv("CC_DUMP_GAIN_DIR")) {
    config.gain_dump_dir = dir;
  }
  if (std::getenv("CC_TRANSFER_PROFILE") != nullptr) {
    config.profile_transfers = true;
  }
}

const char kColorCorrectionUsage[] =
    "  --color-correction field|global|off\n"
    "  --color-correction-space lum|rgb\n"
    "  --color-correction-grid W H\n"
    "  --color-correction-iters N\n"
    "  --color-correction-lambda-smooth F\n"
    "  --color-correction-lambda-anchor F\n"
    "  --color-correction-max-log-gain F\n"
    "  --color-correction-data-weight F\n"
    "  --color-correction-inset F\n"
    "  --color-correction-reference-camera I\n"
    "  --color-correction-proxy-downscale I\n"
    "  --color-correction-temporal A  EMA of the gain field across\n"
    "                              frames, A in (0,1); smaller =\n"
    "                              stronger (default 0.15; outside\n"
    "                              (0,1) = off)\n"
    "  --color-correction-gain-dump-dir DIR\n"
    "  --color-correction-profile  print the H2D/D2H/compute split\n"
    "  --color-correction-cpu      solve the gain field on the CPU\n"
    "                              reference path\n"
    "  --color-correction-cpu-bake bake the corrected atlas on the CPU\n";

bool ParseColorCorrectionFlag(int argc, char* argv[], int& i,
                              grtmesh::tex::ColorCorrectionConfig& config) {
  using grtmesh::tex::ColorCorrectionMode;
  using grtmesh::tex::ColorCorrectionSpace;
  const std::string a = argv[i];
  auto has = [&](int n) { return i + n < argc; };
  auto next = [&]() { return std::string(argv[++i]); };
  if (a == "--color-correction" && has(1)) {
    const std::string v = next();
    config.mode = v == "off"      ? ColorCorrectionMode::kOff
                  : v == "global" ? ColorCorrectionMode::kGlobalOnly
                                  : ColorCorrectionMode::kLowFreqField;
  } else if (a == "--color-correction-space" && has(1)) {
    config.space = (next() == "rgb") ? ColorCorrectionSpace::kLinearRgb
                                     : ColorCorrectionSpace::kLuminanceOnly;
  } else if (a == "--color-correction-grid" && has(2)) {
    config.field_grid_w = std::stoi(next());
    config.field_grid_h = std::stoi(next());
  } else if (a == "--color-correction-iters" && has(1)) {
    config.field_iterations = std::stoi(next());
  } else if (a == "--color-correction-lambda-smooth" && has(1)) {
    config.lambda_smooth = std::stof(next());
  } else if (a == "--color-correction-lambda-anchor" && has(1)) {
    config.lambda_anchor = std::stof(next());
  } else if (a == "--color-correction-max-log-gain" && has(1)) {
    config.max_log_gain = std::stof(next());
  } else if (a == "--color-correction-data-weight" && has(1)) {
    config.seam_data_weight = std::stof(next());
  } else if (a == "--color-correction-inset" && has(1)) {
    config.sample_inset_proxy_pixels = std::stof(next());
  } else if (a == "--color-correction-reference-camera" && has(1)) {
    config.reference_camera = std::stoi(next());
  } else if (a == "--color-correction-proxy-downscale" && has(1)) {
    config.proxy_downscale = std::stoi(next());
  } else if (a == "--color-correction-temporal" && has(1)) {
    config.temporal_alpha = std::stof(next());
  } else if (a == "--color-correction-gain-dump-dir" && has(1)) {
    config.gain_dump_dir = next();
  } else if (a == "--color-correction-profile") {
    config.profile_transfers = true;
  } else if (a == "--color-correction-cpu") {
    config.cpu_solve = true;
  } else if (a == "--color-correction-cpu-bake") {
    config.cpu_bake = true;
  } else {
    return false;
  }
  return true;
}

}  // namespace grtmesh::app
