// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Reconstruction of a calibrated multi-view RGB-D sequence in the generic
// layout of RgbdSequenceDataset (cameras.json + frames/NNNNNN/), e.g. the
// synthetic sequence rendered by python/synthetic/.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "app/parallel_runtime.h"
#include "app/rgbd_sequence_dataset.h"
#include "app/util.h"
#include "grtmesh/grtmesh.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: multiview_rgbd <dataset_dir> <out_dir> [flags]\n"
         "  Reads <dataset_dir>/cameras.json and <dataset_dir>/frames/NNNNNN/\n"
         "  and writes <out_dir>/mesh/NNNNNN.{obj,mtl,jpg} plus a timing CSV.\n"
         "\n"
         "Flags:\n"
         "  --cameras <comma list>      cameras to reconstruct from, by name\n"
         "                              (default: all in cameras.json)\n"
         "  --voxel-size F              voxel edge in meters (default 0.02)\n"
         "  --occlusion-threshold F     depth tolerance of the view-selection\n"
         "                              visibility test in meters (default\n"
         "                              2 x voxel size)\n"
      << grtmesh::app::kColorCorrectionUsage
      << "  --start-frame N             first frame (default: the first one)\n"
         "  --end-frame N               exclusive end frame (default: all)\n"
         "  --max-frames N\n"
         "  --frame-stride N\n"
         "  --no-save-mesh\n"
         "  -h, --help                  print this message\n";
}

}  // namespace

using grtmesh::FrameReport;
using grtmesh::Mesh;
using grtmesh::Mesher;
using grtmesh::MesherParams;
using grtmesh::RgbdFramePtr;
using grtmesh::VolumeParams;
using grtmesh::app::RgbdSequenceDataset;
using grtmesh::app::Stopwatch;
using grtmesh::app::UndistortPreprocessor;

const int kOtherAppsThreadsNum = 2;  // For other apps on the PC

int main(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      PrintUsage();
      return 0;
    }
  }
  if (argc < 3) {
    PrintUsage();
    return 1;
  }
  const std::filesystem::path dataset_dir = argv[1];
  const std::filesystem::path out_dir = argv[2];

  grtmesh::tex::ColorCorrectionConfig color_correction_config;
  grtmesh::app::ApplyColorCorrectionEnvironment(color_correction_config);
  std::vector<std::string> camera_names;
  float voxel_size = 0.02f;
  float occlusion_threshold = -1.f;
  int start_frame = -1, end_frame = -1, max_frames = -1, frame_stride = 1;
  bool save_mesh = true;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (grtmesh::app::ParseColorCorrectionFlag(argc, argv, i,
                                               color_correction_config)) {
      continue;
    }
    if (a == "--cameras" && i + 1 < argc) {
      camera_names = grtmesh::app::ParseStringList(next());
    } else if (a == "--voxel-size" && i + 1 < argc) {
      voxel_size = std::stof(next());
    } else if (a == "--occlusion-threshold" && i + 1 < argc) {
      occlusion_threshold = std::stof(next());
    } else if (a == "--start-frame" && i + 1 < argc) {
      start_frame = std::stoi(next());
    } else if (a == "--end-frame" && i + 1 < argc) {
      end_frame = std::stoi(next());
    } else if (a == "--max-frames" && i + 1 < argc) {
      max_frames = std::stoi(next());
    } else if (a == "--frame-stride" && i + 1 < argc) {
      frame_stride = std::max(1, std::stoi(next()));
    } else if (a == "--no-save-mesh") {
      save_mesh = false;
    } else {
      std::cerr << "unknown argument: " << a << std::endl;
      PrintUsage();
      return 1;
    }
  }

  RgbdSequenceDataset dataset;
  dataset.Load(dataset_dir, camera_names);
  std::cout << "cameras       :";
  for (const RgbdSequenceDataset::Camera& camera : dataset.cameras()) {
    std::cout << " " << camera.name;
  }
  std::cout << std::endl;

  // Frames to reconstruct: the dataset's frame numbers within
  // [start, end), every `frame_stride`-th.
  std::vector<uint32_t> frames_to_process;
  for (uint32_t frame : dataset.frame_numbers()) {
    if ((start_frame < 0 || frame >= static_cast<uint32_t>(start_frame)) &&
        (end_frame < 0 || frame < static_cast<uint32_t>(end_frame))) {
      frames_to_process.push_back(frame);
    }
  }
  std::vector<uint32_t> strided;
  for (size_t i = 0; i < frames_to_process.size(); i += frame_stride) {
    strided.push_back(frames_to_process[i]);
  }
  if (max_frames > 0 && strided.size() > static_cast<size_t>(max_frames)) {
    strided.resize(max_frames);
  }
  frames_to_process = std::move(strided);
  if (frames_to_process.empty()) {
    std::cerr << "no frames to process" << std::endl;
    return 1;
  }

  std::filesystem::create_directories(out_dir / "mesh");
  const std::time_t now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::stringstream timestamp;
  timestamp << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
  std::ofstream timing_csv(out_dir / ("timing_" + timestamp.str() + ".csv"));
  timing_csv << "frame,all,host2device,tsdf,mc,denoise,texture,device2host,"
                "misc,texture_only,inpaint,cc_added_ms,cc_seam_before,cc_seam_"
                "after,gpu_mem_used_mb"
             << std::endl;

  VolumeParams volume_params;
  volume_params.min_corner = dataset.bounds_min();
  volume_params.max_corner = dataset.bounds_max();
  const Eigen::Vector3f extent =
      volume_params.max_corner - volume_params.min_corner;
  for (int a = 0; a < 3; ++a) {
    volume_params.voxel_count[a] =
        static_cast<uint32_t>(std::ceil(extent[a] / voxel_size));
  }
  const Eigen::Vector3f resolution =
      extent.cwiseQuotient(volume_params.voxel_count.cast<float>());
  std::cout << "volume        : " << volume_params.voxel_count.transpose()
            << " voxels of " << resolution.transpose() << " m" << std::endl;

  UndistortPreprocessor preprocessor;
  // Depth is in meters; keep everything inside the volume diagonal.
  preprocessor.options.depth_far_threshold = extent.norm() * 2.f;
  std::vector<RgbdFramePtr> frames;
  std::vector<RgbdFramePtr> next_frames;
  {
    const auto first = dataset.Get(frames_to_process.front());
    preprocessor.Init(first);
    preprocessor.Run(first, frames);
  }

  Mesher mesher;
  mesher.Init(volume_params, frames);

  // Same settings as the Panoptic runs of the paper, with the camera-ready
  // occlusion threshold (2 voxels).
  MesherParams params;
  params.noise_removal.label_iterations = 100;
  params.noise_removal.min_component_faces = 500;
  params.noise_removal.convergence_check_interval = 10;
  params.view_selection.inpaint_iterations = 5;
  params.view_selection.occlusion_threshold = occlusion_threshold > 0.f
                                                  ? occlusion_threshold
                                                  : 2.f * resolution.minCoeff();
  params.smooth_face_normals = true;
  params.fusion.truncation_distance = resolution.minCoeff() * 4.f * 2;
  params.color_correction = color_correction_config;

  grtmesh::app::SetHostThreadCount(
      std::max(1u, std::thread::hardware_concurrency() - kOtherAppsThreadsNum -
                       grtmesh::app::preprocess_thread_count));
  for (size_t n = 0; n < frames_to_process.size(); ++n) {
    const uint32_t frame = frames_to_process[n];
    // Load and preprocess the next frame while this one is reconstructed.
    std::thread preprocess_thread([&]() {
      if (n + 1 < frames_to_process.size()) {
        preprocessor.Run(dataset.Get(frames_to_process[n + 1]), next_frames);
      }
    });

    Mesh mesh;
    FrameReport report;
    Stopwatch watch;
    mesher.Process(frames, params, mesh, &report);
    const double all_time = watch.ElapsedMs();

    preprocess_thread.join();
    std::swap(frames, next_frames);

    if (save_mesh) {
      grtmesh::app::SaveAsObj(
          (out_dir / "mesh" / (grtmesh::app::Zfill(frame, 6) + ".obj"))
              .string(),
          mesh);
    }

    const grtmesh::StageTimings& t = report.timings;
    const double texture_all_time = t.atlas_uv_ms + t.view_selection_ms;
    const double device2host_time = t.download_geometry_ms + t.download_uv_ms;
    const double misc_time = all_time - t.upload_ms - t.fusion_ms -
                             t.marching_cubes_ms - t.noise_removal_ms -
                             texture_all_time - device2host_time;
    size_t gpu_free = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    const grtmesh::tex::ColorCorrectionStats& cc = report.color_correction;
    std::stringstream line;
    line << frame << "," << all_time << "," << t.upload_ms << "," << t.fusion_ms
         << "," << t.marching_cubes_ms << "," << t.noise_removal_ms << ","
         << texture_all_time << "," << device2host_time << "," << misc_time
         << "," << t.atlas_uv_ms << "," << t.view_selection_ms;
    std::cout << line.str() << std::endl;
    timing_csv << line.str() << "," << cc.total_added_ms << ","
               << cc.mean_abs_seam_residual_before << ","
               << cc.mean_abs_seam_residual_after << ","
               << (gpu_total - gpu_free) / (1024.0 * 1024.0) << std::endl;
  }
  return 0;
}
