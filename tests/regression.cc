// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "grtmesh/grtmesh.h"
#include "grtmesh/tex/color_correction_internal.h"
#include "grtmesh/tex/view_selection.cuh"

void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class T>
struct DeviceData {
  T* ptr = nullptr;
  explicit DeviceData(const std::vector<T>& data) {
    Require(cudaMalloc(&ptr, data.size() * sizeof(T)) == cudaSuccess,
            "cudaMalloc");
    Require(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess,
            "upload");
  }
  ~DeviceData() { cudaFree(ptr); }
};
std::vector<uint8_t> Pixels(const grtmesh::ColorImage& image) {
  return {image.data(), image.data() + image.size()};
}
grtmesh::ColorImage Atlas(int first, int second) {
  auto a = grtmesh::ColorImage::Create(32, 64);
  std::fill_n(a.data(), 32 * 32 * 3, static_cast<uint8_t>(first));
  std::fill_n(a.data() + 32 * 32 * 3, 32 * 32 * 3,
              static_cast<uint8_t>(second));
  return a;
}
void AllocationFailureRecovery() {
  grtmesh::tex::ColorCorrectionContext context;
  const std::vector<float> gains{0.05f, -0.05f};
  auto bake = [&](grtmesh::ColorImage& atlas) {
    return context.Bake(atlas, gains, 32, 32, 2, 1, 1, 1, false, 0.15f,
                        grtmesh::tex::AtlasLayout::kVertical);
  };
  auto expected = Atlas(80, 145);
  Require(bake(expected) >= 0.f, "initial bake allocation");
  // Impossible size is rejected by cudaMalloc before any image transfer;
  // unlike exhausting VRAM, this does not consume the shared GPU's memory.
  auto impossible = Atlas(80, 145);
  impossible.width = impossible.height = 2147483647u;
  Require(
      context.Bake(impossible, gains, 2147483647, 1073741823, 2, 1, 1, 1, false,
                   0.15f, grtmesh::tex::AtlasLayout::kVertical) < 0.f,
      "allocation failure should be recoverable");
  auto actual = Atlas(80, 145);
  Require(bake(actual) >= 0.f, "bake after failed buffer growth");
  Require(Pixels(expected) == Pixels(actual),
          "failed growth corrupted buffers");
}

std::vector<grtmesh::RgbdFramePtr> Frames(int value, float focal = 60.f) {
  std::vector<grtmesh::RgbdFramePtr> frames;
  for (int i = 0; i < 2; ++i) {
    auto f = std::make_shared<grtmesh::RgbdFrame>();
    f->depth = grtmesh::DepthImage::Create(64, 48);
    std::fill_n(f->depth.data(), 64 * 48, 1.f);
    f->depth_in_color = f->depth;
    f->color = grtmesh::ColorImage::Create(64, 48);
    std::fill_n(f->color.data(), 64 * 48 * 3, static_cast<uint8_t>(value));
    f->depth_intrinsics.fx = f->depth_intrinsics.fy = focal;
    f->depth_intrinsics.cx = 32;
    f->depth_intrinsics.cy = 24;
    f->color_intrinsics = f->depth_intrinsics;
    f->depth_to_world = f->color_to_world = Eigen::Affine3f::Identity();
    frames.push_back(f);
  }
  return frames;
}

void RetainedCorrectionModesAndLayouts() {
  using grtmesh::tex::AtlasLayout;
  using grtmesh::tex::ColorCorrectionConfig;
  using grtmesh::tex::ColorCorrectionContext;
  using grtmesh::tex::ColorCorrectionMode;
  using grtmesh::tex::ColorCorrectionStats;
  using grtmesh::tex::PackAtlas;
  for (auto layout :
       {AtlasLayout::kVertical, AtlasLayout::kHorizontal, AtlasLayout::kTile}) {
    for (auto mode : {ColorCorrectionMode::kLowFreqField,
                      ColorCorrectionMode::kGlobalOnly}) {
      auto frames = std::vector<grtmesh::RgbdFramePtr>{};
      for (int value : {80, 145}) {
        auto f = std::make_shared<grtmesh::RgbdFrame>();
        f->color = grtmesh::ColorImage::Create(32, 32);
        std::fill_n(f->color.data(), 32 * 32 * 3, static_cast<uint8_t>(value));
        frames.push_back(f);
      }
      std::vector<grtmesh::ColorImage> images = {frames[0]->color,
                                                 frames[1]->color};
      grtmesh::Mesh mesh;
      mesh.faces = {{0, 1, 2}, {1, 3, 2}};
      mesh.face_normals = {{0, 0, -1}, {0, 0, -1}};
      mesh.face_camera_ids = {0, 1};
      mesh.face_image_coords = {
          {{Eigen::Vector2f(8, 8), Eigen::Vector2f(24, 8),
            Eigen::Vector2f(8, 24)}},
          {{Eigen::Vector2f(24, 8), Eigen::Vector2f(24, 24),
            Eigen::Vector2f(8, 24)}}};
      ColorCorrectionConfig cfg;
      cfg.mode = mode;
      cfg.field_grid_w = cfg.field_grid_h = 4;
      cfg.proxy_downscale = 1;
      ColorCorrectionContext cpu, gpu;
      ColorCorrectionStats cpu_stats, gpu_stats;
      cfg.cpu_bake = true;
      PackAtlas(images, mesh.atlas, layout);
      auto original = Pixels(mesh.atlas);
      ApplyColorCorrectionFieldCpu(cpu, mesh, frames, cfg, layout, cpu_stats);
      Require(cpu_stats.seam_edge_count > 0,
              "CPU correction must find the synthetic seam");
      auto expected = Pixels(mesh.atlas);
      Require(expected != original, "CPU correction must change the atlas");
      cfg.cpu_bake = false;
      PackAtlas(images, mesh.atlas, layout);
      ApplyColorCorrectionFieldCpu(gpu, mesh, frames, cfg, layout, gpu_stats);
      auto actual = Pixels(mesh.atlas);
      for (size_t i = 0; i < actual.size(); ++i)
        Require(std::abs(static_cast<int>(actual[i]) -
                         static_cast<int>(expected[i])) <= 1,
                "CPU/GPU bake disagrees for retained layout or mode");
    }
  }
}

void ContextIsolation() {
  using grtmesh::tex::AtlasLayout;
  using grtmesh::tex::ColorCorrectionConfig;
  using grtmesh::tex::ColorCorrectionContext;
  using grtmesh::tex::ColorCorrectionMode;
  using grtmesh::tex::ColorCorrectionStats;
  DeviceData<int> faces({0, 1, 2, 1, 3, 2});
  DeviceData<int> ids({0, 1});
  DeviceData<float3> normals({make_float3(0, 0, -1), make_float3(0, 0, -1)});
  DeviceData<float2> projected({make_float2(8, 8), make_float2(24, 8),
                                make_float2(8, 24), make_float2(24, 8),
                                make_float2(24, 24), make_float2(8, 24)});
  ColorCorrectionConfig cfg;
  cfg.mode = ColorCorrectionMode::kLowFreqField;
  cfg.field_grid_w = 4;
  cfg.field_grid_h = 4;
  cfg.proxy_downscale = 1;
  cfg.temporal_alpha = 0.15f;  // the EMA history must stay per context
  ColorCorrectionContext reference, a, b;
  auto prepare = [&](ColorCorrectionContext& ctx, int n = 2) {
    Require(ctx.PrepareGeometry(faces.ptr, n, normals.ptr, nullptr), "prepare");
  };
  auto bake = [&](ColorCorrectionContext& ctx, grtmesh::ColorImage& atlas) {
    ColorCorrectionStats stats;
    Require(ctx.SolveAndBake(atlas, ids.ptr, projected.ptr, 2, cfg, 32, 32,
                             AtlasLayout::kVertical, stats) >= 0,
            "bake");
    return stats;
  };
  // Alter the second context's face count between preparation and baking A.
  // A singleton mixes B's one-face geometry with A's two-face projection data.
  for (int frame = 0; frame < 3; ++frame) {
    auto expected = Atlas(80 + frame * 5, 145);
    prepare(reference);
    auto expected_stats = bake(reference, expected);
    auto actual = Atlas(80 + frame * 5, 145);
    prepare(a);
    prepare(b, 1);
    auto other = Atlas(180, 60);
    bake(b, other);
    auto actual_stats = bake(a, actual);
    Require(expected_stats.seam_edge_count > 0, "synthetic seam missing");
    Require(actual_stats.seam_edge_count == expected_stats.seam_edge_count,
            "cross-instance geometry");
    auto x = Pixels(expected), y = Pixels(actual);
    for (size_t i = 0; i < x.size(); ++i)
      Require(std::abs(static_cast<int>(x[i]) - static_cast<int>(y[i])) <= 1,
              "cross-instance EMA");
  }
  a.Reset();
  auto invalid = Atlas(80, 145);
  ColorCorrectionStats stats;
  Require(a.SolveAndBake(invalid, ids.ptr, projected.ptr, 2, cfg, 32, 32,
                         AtlasLayout::kVertical, stats) < 0,
          "reset must invalidate prepared geometry");
  ColorCorrectionContext fresh;
  auto x = Atlas(100, 145), y = Atlas(100, 145);
  prepare(a);
  prepare(fresh);
  bake(a, x);
  bake(fresh, y);
  Require(Pixels(x) == Pixels(y), "reset must clear EMA history");
}

void ProjectionIsolation() {
  // Fixed geometry avoids marching-cubes ordering and inpainting ambiguity.
  // Test the actual projection against its analytic pixel coordinates, with a
  // second selector holding different intrinsics.
  auto frames = Frames(90);
  std::vector<grtmesh::CameraIntrinsics> intrinsics(
      2, frames[0]->color_intrinsics);
  std::vector<Eigen::Affine3f> transforms(2, Eigen::Affine3f::Identity());
  std::vector<Eigen::Vector3f> directions(2, Eigen::Vector3f(0, 0, 1));
  grtmesh::tex::ViewSelector a, b;
  a.Init(1, 64, 48, intrinsics, transforms, directions);
  for (auto& k : intrinsics) k.fx = k.fy = 80.f;
  b.Init(1, 64, 48, intrinsics, transforms, directions);
  DeviceData<float3> vertices({make_float3(-.1f, -.1f, 1.f),
                               make_float3(.1f, -.1f, 1.f),
                               make_float3(0, .1f, 1.f)});
  DeviceData<int> faces({0, 1, 2});
  DeviceData<float3> normals({make_float3(0, 0, -1)});
  b.AssignViews(frames, vertices.ptr, faces.ptr, normals.ptr, 1, .1f, 80.f, 0);
  a.AssignViews(frames, vertices.ptr, faces.ptr, normals.ptr, 1, .1f, 80.f, 0);
  std::vector<std::array<Eigen::Vector2f, 3>> projected;
  a.DownloadFaceImageCoords(projected);
  Require(projected.size() == 1, "fixed triangle projection");
  const Eigen::Vector2f expected[] = {{26, 18}, {38, 18}, {32, 30}};
  for (int i = 0; i < 3; ++i)
    Require((projected[0][i] - expected[i]).norm() < 1e-5f,
            "camera parameters leaked across instances");
}

void RetainedAtlasAndMesherSwitch() {
  using grtmesh::Mesh;
  using grtmesh::Mesher;
  using grtmesh::MesherParams;
  using grtmesh::VolumeParams;
  VolumeParams init;
  init.min_corner = {-.5f, -.4f, .8f};
  init.max_corner = {.5f, .4f, 1.2f};
  init.voxel_count = {32, 24, 16};
  MesherParams params;
  params.color_correction.mode =
      grtmesh::tex::ColorCorrectionMode::kLowFreqField;
  params.color_correction.temporal_alpha = 0.15f;
  Mesh retained;
  std::vector<uint8_t> expected;
  {
    Mesher a, b;
    auto fa = Frames(90), fb = Frames(180, 80.f);
    Require(a.Init(init, fa), "init A");
    Require(a.Process(fa, params, retained), "process A");
    Require(!retained.atlas.empty(), "atlas must have storage");
    expected = Pixels(retained.atlas);
    auto original_pointer = retained.atlas.data();
    Require(b.Init(init, fb), "init B");
    Mesh next, other;
    Require(a.Process(fa, params, next), "process A after B Init");
    Require(b.Process(fb, params, other), "process B interleaved");
    Require(next.atlas.data() != original_pointer, "retained output reused");
    Require(Pixels(retained.atlas) == expected, "previous frame overwritten");
    Require(Pixels(next.atlas) == expected, "interleaved atlas changed");
    for (auto uv : next.uvs)
      Require(uv.allFinite(), "invalid UV after instance switch");
    // Reusing the same output object must keep the no-allocation fast path.
    auto reusable = next.atlas.data();
    Require(a.Process(fa, params, next), "repeat Process");
    Require(next.atlas.data() == reusable,
            "unshared output should reuse its pinned allocation");
    params.view_selection.inpaint_iterations = 0;
    Require(a.Process(fa, params, next), "Process with propagation disabled");
    Require(Pixels(next.atlas) == expected,
            "fully visible atlas changed without propagation");
    for (auto uv : next.uvs)
      Require(uv.allFinite(), "invalid UV without propagation");
    params.download_face_assignments = true;
    Require(a.Process(fa, params, next), "Process with face assignments");
    Require(next.face_camera_ids.size() == next.faces.size() &&
                next.face_image_coords.size() == next.faces.size(),
            "face assignments not downloaded");
    params.build_atlas = false;
    Require(a.Process(fa, params, next), "geometry-only Process");
    Require(next.atlas.empty() && next.uvs.empty(), "geometry-only atlas");
  }
  Require(Pixels(retained.atlas) == expected,
          "atlas invalid after Mesher destruction");
  grtmesh::ColorImage copy = retained.atlas.Clone();
  Require(Pixels(copy) == expected, "deep copy of pinned atlas");
}

void PlainTexturedTransition() {
  grtmesh::VolumeParams volume;
  volume.min_corner = {-.5f, -.4f, .8f};
  volume.max_corner = {.5f, .4f, 1.2f};
  volume.voxel_count = {32, 24, 16};
  auto frames = Frames(90);
  grtmesh::Mesher mesher;
  Require(mesher.Init(volume, frames), "plain transition Init");
  grtmesh::MesherParams params;
  params.view_selection.occlusion_threshold = .04f;
  params.view_selection.inpaint_iterations = 5;
  params.noise_removal.min_component_faces = 1;
  params.smooth_face_normals = true;
  params.color_correction.mode =
      grtmesh::tex::ColorCorrectionMode::kLowFreqField;
  grtmesh::Mesh reference;
  grtmesh::FrameReport report;
  Require(mesher.Process(frames, params, reference, &report),
          "reference textured Process");
  Require(!reference.faces.empty() && !reference.atlas.empty() &&
              !reference.uvs.empty(),
          "reference must produce geometry, atlas and UVs");
  auto pixels = Pixels(reference.atlas);
  // Reuse an output that owns an atlas; plain must clear it without changing
  // a separately retained output, and must reset per-frame correction/timers.
  grtmesh::Mesh mesh = reference;
  params.with_texturing = false;
  Require(mesher.Process(frames, params, mesh, &report), "plain Process");
  Require(!mesh.faces.empty() && mesh.atlas.empty() && mesh.uvs.empty() &&
              mesh.uv_faces.empty() && mesh.face_camera_ids.empty() &&
              mesh.face_image_coords.empty(),
          "plain must produce only geometry");
  Require(report.color_correction.total_added_ms == 0 &&
              report.color_correction.seam_edge_count == 0,
          "plain must not retain correction statistics");
  Require(report.timings.view_selection_ms == 0 &&
              report.timings.atlas_uv_ms == 0 &&
              report.timings.download_uv_ms == 0 &&
              report.timings.atlas_ms == 0,
          "plain texture stage timings must be zero");
  Require(report.detailed.view_input_setup_ms == 0 &&
              report.detailed.view_selection_ms == 0 &&
              report.detailed.texture_adjacency_ms == 0 &&
              report.detailed.inpaint_kernel_ms == 0,
          "plain detailed texture timings must reset");
  Require(std::isfinite(report.detailed.depth_upload_ms) &&
              report.detailed.depth_upload_ms >= 0,
          "plain upload timing valid");
  Require(Pixels(reference.atlas) == pixels, "plain preserves retained atlas");
  params.with_texturing = true;
  Require(mesher.Process(frames, params, mesh, &report),
          "textured Process after plain");
  Require(!mesh.atlas.empty() && !mesh.uvs.empty() &&
              mesh.uv_faces.size() == mesh.faces.size(),
          "texture output recovers");
  Require(Pixels(mesh.atlas) == pixels, "atlas recovers after plain");
  Require(report.detailed.view_selection_ms > 0 &&
              report.detailed.texture_adjacency_ms > 0 &&
              report.detailed.inpaint_kernel_ms > 0,
          "detailed timings recover after plain");
}

// FNV-1a 64 over the little-endian bytes of the maps (xy, then frac).
uint64_t MapHash(const std::vector<int16_t>& xy,
                 const std::vector<uint16_t>& frac) {
  uint64_t h = 0xcbf29ce484222325ull;
  auto feed = [&h](uint16_t v) {
    for (int byte : {v & 0xff, v >> 8}) {
      h ^= static_cast<uint64_t>(byte);
      h *= 0x100000001b3ull;
    }
  };
  for (int16_t v : xy) feed(static_cast<uint16_t>(v));
  for (uint16_t v : frac) feed(v);
  return h;
}

// grtmesh::BuildUndistortMaps against hashes of the maps of OpenCV 4.11
// (cv::initUndistortRectifyMap, CV_16SC2, AVX2 dispatch) for the same float
// calibrations; the second width (61) also exercises the scalar row tail.
void UndistortMapsMatchOpenCv() {
  struct Case {
    uint32_t width, height;
    grtmesh::CameraIntrinsics k;
    std::array<float, 8> distortion;
    uint64_t opencv_hash;
  };
  const Case cases[] = {
      {640,
       480,
       {530.25f, 529.75f, 318.4f, 241.6f},
       {0.052f, -0.071f, 0.0013f, -0.00052f, 0.021f, 0.f, 0.f, 0.f},
       0x9c71d5b46c2c1136ull},
      {61,
       45,
       {50.5f, 49.25f, 30.2f, 22.7f},
       {-0.3f, 0.12f, 0.002f, -0.001f, -0.02f, 0.01f, 0.005f, -0.002f},
       0x9a6475c570ef21b1ull},
  };
  for (const Case& c : cases) {
    grtmesh::CameraCalibration calibration;
    calibration.intrinsics = c.k;
    calibration.width = c.width;
    calibration.height = c.height;
    calibration.distortion = c.distortion;
    std::vector<int16_t> xy;
    std::vector<uint16_t> frac;
    grtmesh::BuildUndistortMaps(calibration, xy, frac);
    Require(MapHash(xy, frac) == c.opencv_hash,
            "undistortion maps must match OpenCV bit for bit");
  }
}

// InputPreprocessor on registered, distortion-free input: the color passes
// through, depth beyond the far threshold is cut, and the device and host
// outputs agree.
void InputPreprocessorPassThrough() {
  constexpr uint32_t kW = 40, kH = 24;
  grtmesh::CameraCalibration calibration;
  calibration.intrinsics = {30.f, 30.f, 20.f, 12.f};
  calibration.width = kW;
  calibration.height = kH;
  std::vector<uint8_t> color(kW * kH * 3);
  std::vector<float> depth(kW * kH);
  for (size_t i = 0; i < color.size(); ++i) color[i] = (i * 7) % 251;
  for (size_t i = 0; i < depth.size(); ++i) depth[i] = 0.5f + 0.01f * i;
  std::vector<grtmesh::SensorFrame> sensor(2);
  for (grtmesh::SensorFrame& s : sensor) {
    s.color = color.data();
    s.depth = depth.data();
    s.color_calibration = calibration;
    s.depth_calibration = calibration;
  }
  grtmesh::InputPreprocessor preprocessor;
  preprocessor.options.depth_far_threshold = 5.f;
  Require(preprocessor.Init(sensor), "preprocessor init");
  std::vector<grtmesh::RgbdFramePtr> host;
  Require(preprocessor.Run(sensor, host), "preprocessor host run");
  Require(host.size() == 2, "one frame per sensor");
  for (const grtmesh::RgbdFramePtr& f : host) {
    Require(std::equal(color.begin(), color.end(), f->color.data()),
            "undistortion-free color must pass through");
    for (size_t i = 0; i < depth.size(); ++i) {
      const float expected = depth[i] > 5.f ? 0.f : depth[i];
      Require(f->depth.data()[i] == expected, "far depth must be cut");
      Require(f->depth_in_color.data()[i] == expected,
              "registered depth is the depth in color");
    }
  }
  grtmesh::InputPreprocessor::DeviceFrames device;
  std::vector<grtmesh::RgbdFramePtr> frames;
  Require(preprocessor.Run(sensor, frames, device), "preprocessor device run");
  const grtmesh::DeviceRgbdBatch batch = device.batch();
  std::vector<uint8_t> device_color(color.size() * 2);
  Require(cudaMemcpy(device_color.data(), batch.color, device_color.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "download device color");
  Require(std::equal(color.begin(), color.end(), device_color.begin()) &&
              std::equal(color.begin(), color.end(),
                         device_color.begin() + color.size()),
          "device output must match the host output");
  Require(frames[0]->color.width == kW && frames[0]->color.empty(),
          "device run sets sizes, not pixels");
}

int main() {
  try {
    UndistortMapsMatchOpenCv();
    InputPreprocessorPassThrough();
    AllocationFailureRecovery();
    RetainedCorrectionModesAndLayouts();
    ContextIsolation();
    ProjectionIsolation();
    RetainedAtlasAndMesherSwitch();
    PlainTexturedTransition();
    std::cout << "PASS: undistortion maps = OpenCV, input preprocessing, "
                 "context isolation, EMA reset, retained atlas, "
                 "interleaved Meshers, plain/textured transitions\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
