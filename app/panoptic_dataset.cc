// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "app/panoptic_dataset.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

namespace fs = std::filesystem;

namespace grtmesh::app {
namespace {

// Kinect ids are 1-based; sensors[k - 1] of the calibration is Kinect k.
constexpr int kMaxKinectId = 90;
constexpr float kDepthScale = 1000.f;  // depth PNG in mm

struct KinectCalibration {
  int id = 0;
  CameraCalibration color;
  CameraCalibration depth;
  Eigen::Affine3f color_to_world = Eigen::Affine3f::Identity();
  Eigen::Affine3f depth_to_world = Eigen::Affine3f::Identity();
};

Eigen::Affine3f ParseMatrix4(const nlohmann::json& j) {
  if (!(j.is_array() && j.size() == 4 && j[0].is_array() && j[0].size() == 4)) {
    throw std::runtime_error("kcalibration: expected a 4x4 matrix");
  }
  Eigen::Matrix4f matrix;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      matrix(r, c) = j[r][c].get<float>();
    }
  }
  Eigen::Affine3f transform;
  transform.matrix() = matrix;
  return transform;
}

// K (3x3), distortion (OpenCV order, the first five coefficients are used)
// and the image size.
CameraCalibration ParseCamera(const nlohmann::json& k,
                              const nlohmann::json& distortion,
                              const nlohmann::json& width,
                              const nlohmann::json& height) {
  CameraCalibration calibration;
  calibration.width = width.get<int>();
  calibration.height = height.get<int>();
  calibration.intrinsics.fx = k[0][0].get<float>();
  calibration.intrinsics.fy = k[1][1].get<float>();
  calibration.intrinsics.cx = k[0][2].get<float>();
  calibration.intrinsics.cy = k[1][2].get<float>();
  // Entries are scalars or 1-element arrays ([[k1], [k2], ...]).
  const size_t n = std::min<size_t>(distortion.size(), 5);
  for (size_t i = 0; i < n; ++i) {
    const nlohmann::json& v = distortion[i];
    calibration.distortion[i] =
        v.is_array() ? v[0].get<float>() : v.get<float>();
  }
  return calibration;
}

std::vector<KinectCalibration> LoadKinectCalibration(
    const fs::path& calib_dir) {
  fs::path path;
  for (const fs::directory_entry& e : fs::directory_iterator(calib_dir)) {
    if (e.is_regular_file() &&
        e.path().filename().string().rfind("kcalibration_", 0) == 0) {
      path = e.path();
      break;
    }
  }
  if (path.empty()) {
    throw std::runtime_error("kcalibration_*.json not found in " +
                             calib_dir.string());
  }
  std::cout << "[INFO] using kcalibration: " << path.string() << std::endl;
  std::ifstream ifs(path);
  const nlohmann::json root = nlohmann::json::parse(ifs);
  if (!root.contains("sensors") || !root["sensors"].is_array()) {
    throw std::runtime_error("kcalibration json has no 'sensors' array");
  }
  const nlohmann::json& sensors = root["sensors"];

  std::vector<KinectCalibration> kinects;
  const int count =
      std::min<int>(static_cast<int>(sensors.size()), kMaxKinectId);
  for (int k = 1; k <= count; ++k) {
    const nlohmann::json& s = sensors[k - 1];
    KinectCalibration calib;
    calib.id = k;
    calib.color = ParseCamera(s["K_color"], s["distCoeffs_color"],
                              s["color_width"], s["color_height"]);
    calib.depth = ParseCamera(s["K_depth"], s["distCoeffs_depth"],
                              s["depth_width"], s["depth_height"]);
    // M_world2sensor: world -> Kinect; M_color / M_depth: Kinect -> camera.
    const Eigen::Affine3f world_to_sensor = ParseMatrix4(s["M_world2sensor"]);
    calib.color_to_world =
        (ParseMatrix4(s["M_color"]) * world_to_sensor).inverse();
    calib.depth_to_world =
        (ParseMatrix4(s["M_depth"]) * world_to_sensor).inverse();
    kinects.push_back(calib);
  }
  if (kinects.empty()) {
    throw std::runtime_error("No Kinect calibrations loaded.");
  }
  std::cout << "[INFO] Loaded Kinect calibrations: " << kinects.size()
            << std::endl;
  return kinects;
}

// Kinect selection. The NUM_KINECTS environment variable overrides the
// configured set:
//  1. env NUM_KINECTS=N: the first N Kinects {1..N} (sensor-count sweep
//     without changing the command line),
//  2. `requested` (e.g. {2,4,6,8,10}: the held-out protocol reconstructs from
//     the even Kinects and evaluates on the odd ones),
//  3. every Kinect in the calibration.
std::set<int> SelectKinects(const std::vector<int>& requested,
                            const std::vector<KinectCalibration>& kinects) {
  std::set<int> ids;
  std::string source = "all Kinects in the calibration";
  if (const char* nk = std::getenv("NUM_KINECTS")) {
    const int n = std::atoi(nk);
    for (int i = 1; i <= n; ++i) ids.insert(i);
    if (n > 0) source = "env NUM_KINECTS";
  }
  if (ids.empty() && !requested.empty()) {
    ids.insert(requested.begin(), requested.end());
    source = "--kinect-ids";
  }
  if (ids.empty()) {
    for (const KinectCalibration& k : kinects) ids.insert(k.id);
  }
  std::cout << "[INFO] Kinect ids (" << source << "):";
  for (int k : ids) std::cout << " " << k;
  std::cout << std::endl;
  return ids;
}

}  // namespace

void PanopticKinopticDataset::Load(const fs::path& frames_dir,
                                   const fs::path& calib_dir) {
  depth_scale_ = kDepthScale;
  const std::vector<KinectCalibration> kinects =
      LoadKinectCalibration(calib_dir);
  const std::set<int> selected = SelectKinects(options_.kinect_ids, kinects);

  // hd_NNNNNNNN directories in frame order.
  std::vector<std::pair<uint32_t, fs::path>> hd_dirs;
  if (fs::exists(frames_dir)) {
    for (const fs::directory_entry& e : fs::directory_iterator(frames_dir)) {
      const std::string name = e.path().filename().string();
      if (e.is_directory() && name.rfind("hd_", 0) == 0) {
        hd_dirs.emplace_back(std::stoul(name.substr(3)), e.path());
      }
    }
  }
  if (hd_dirs.empty()) {
    throw std::runtime_error("No hd_* directories found in: " +
                             frames_dir.string());
  }
  std::sort(hd_dirs.begin(), hd_dirs.end());

  for (const auto& [frame, hd_dir] : hd_dirs) {
    std::vector<SourceFramePtr> views;
    for (const KinectCalibration& calib : kinects) {
      if (selected.count(calib.id) == 0) {
        continue;
      }
      const std::string prefix = std::string("k") + (calib.id < 10 ? "0" : "") +
                                 std::to_string(calib.id);
      auto view = std::make_shared<SourceFrame>();
      view->color_path = hd_dir / (prefix + "_rgb.jpg");
      view->depth_path = hd_dir / (prefix + "_depth.png");
      // A Kinect missing in this frame is skipped.
      if (!fs::exists(view->color_path) || !fs::exists(view->depth_path)) {
        continue;
      }
      view->color_calibration = calib.color;
      view->depth_calibration = calib.depth;
      view->color_to_world = calib.color_to_world;
      view->depth_to_world = calib.depth_to_world;
      view->depth_to_color =
          view->color_to_world.inverse() * view->depth_to_world;
      views.push_back(std::move(view));
    }
    AddFrame(frame, std::move(views));
  }
}

}  // namespace grtmesh::app
