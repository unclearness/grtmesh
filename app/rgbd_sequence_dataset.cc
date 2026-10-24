// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "app/rgbd_sequence_dataset.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

namespace fs = std::filesystem;

namespace grtmesh::app {
namespace {

Eigen::Vector3f ParseVector3(const nlohmann::json& j) {
  return Eigen::Vector3f(j.at(0).get<float>(), j.at(1).get<float>(),
                         j.at(2).get<float>());
}

RgbdSequenceDataset::Camera ParseCamera(const nlohmann::json& j) {
  RgbdSequenceDataset::Camera camera;
  camera.name = j.at("name").get<std::string>();
  CameraCalibration& calibration = camera.calibration;
  calibration.width = j.at("width").get<uint32_t>();
  calibration.height = j.at("height").get<uint32_t>();
  calibration.intrinsics.fx = j.at("fx").get<float>();
  calibration.intrinsics.fy = j.at("fy").get<float>();
  calibration.intrinsics.cx = j.at("cx").get<float>();
  calibration.intrinsics.cy = j.at("cy").get<float>();
  if (j.contains("distortion")) {
    const nlohmann::json& d = j["distortion"];
    for (size_t i = 0; i < std::min<size_t>(d.size(), 8); ++i) {
      calibration.distortion[i] = d[i].get<float>();
    }
  }
  const nlohmann::json& m = j.at("camera_to_world");
  Eigen::Matrix4f matrix;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      matrix(r, c) = m.at(r).at(c).get<float>();
    }
  }
  camera.camera_to_world.matrix() = matrix;
  return camera;
}

// <camera>_color.png, or .jpg when there is no PNG.
fs::path ColorPath(const fs::path& frame_dir, const std::string& camera) {
  const fs::path png = frame_dir / (camera + "_color.png");
  return fs::exists(png) ? png : frame_dir / (camera + "_color.jpg");
}

}  // namespace

void RgbdSequenceDataset::Load(const fs::path& root,
                               const std::vector<std::string>& camera_names) {
  const fs::path cameras_path = root / "cameras.json";
  std::ifstream ifs(cameras_path);
  if (!ifs) {
    throw std::runtime_error("Missing " + cameras_path.string());
  }
  const nlohmann::json j = nlohmann::json::parse(ifs);
  depth_scale_ = j.value("depth_scale", 1000.f);
  bounds_min_ = ParseVector3(j.at("bounds").at("min"));
  bounds_max_ = ParseVector3(j.at("bounds").at("max"));

  std::vector<Camera> all;
  for (const nlohmann::json& c : j.at("cameras")) {
    all.push_back(ParseCamera(c));
  }
  cameras_.clear();
  if (camera_names.empty()) {
    cameras_ = all;
  }
  for (const std::string& name : camera_names) {
    const auto it = std::find_if(all.begin(), all.end(), [&](const Camera& c) {
      return c.name == name;
    });
    if (it == all.end()) {
      throw std::runtime_error("Unknown camera '" + name + "' in " +
                               cameras_path.string());
    }
    cameras_.push_back(*it);
  }

  std::vector<std::pair<uint32_t, fs::path>> frame_dirs;
  const fs::path frames_root = root / "frames";
  if (fs::exists(frames_root)) {
    for (const fs::directory_entry& e : fs::directory_iterator(frames_root)) {
      const std::string name = e.path().filename().string();
      if (e.is_directory() && !name.empty() &&
          std::all_of(name.begin(), name.end(),
                      [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
        frame_dirs.emplace_back(std::stoul(name), e.path());
      }
    }
  }
  if (frame_dirs.empty()) {
    throw std::runtime_error("No frame directories in " + frames_root.string());
  }
  std::sort(frame_dirs.begin(), frame_dirs.end());

  for (const auto& [frame, dir] : frame_dirs) {
    std::vector<SourceFramePtr> views;
    for (const Camera& camera : cameras_) {
      auto view = std::make_shared<SourceFrame>();
      view->color_path = ColorPath(dir, camera.name);
      view->depth_path = dir / (camera.name + "_depth.png");
      view->color_calibration = camera.calibration;
      view->depth_calibration = camera.calibration;
      view->color_to_world = camera.camera_to_world;
      view->depth_to_world = camera.camera_to_world;
      view->depth_to_color = Eigen::Affine3f::Identity();
      views.push_back(std::move(view));
    }
    AddFrame(frame, std::move(views));
  }
  std::cout << "[INFO] " << root.string() << ": " << cameras_.size() << " of "
            << all.size() << " cameras, " << frame_numbers().size() << " frames"
            << std::endl;
}

}  // namespace grtmesh::app
