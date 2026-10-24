// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CMU Panoptic (Kinoptic) reconstruction: the held-out evaluation and
// free-view reconstruction entry point of the paper.

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "app/panoptic_dataset.h"
#include "app/parallel_runtime.h"
#include "app/util.h"
#include "grtmesh/grtmesh.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: cmu_panoptic <image_root> <data_name> <calib_dir> <out_root> "
         "[flags]\n"
         "  Reads <image_root>/<data_name>/frames/hd_NNNNNNNN/ (written by\n"
         "  python/panoptic_prepare.py) and writes the meshes to\n"
         "  <out_root>/<data_name>mesh/NNNNNN.obj (NNNNNN = HD frame number)\n"
         "\n"
         "Flags:\n"
         "  --kinect-ids <comma list>   Kinect ids used for reconstruction,\n"
         "                              e.g. 2,4,6,8,10 (held-out eval:\n"
         "                              reconstruct from even, evaluate on\n"
         "                              odd). Default: all Kinects listed in\n"
         "                              the calibration.\n"
      << grtmesh::app::kColorCorrectionUsage
      << "  --start-frame N             first HD frame (default 1)\n"
         "  --end-frame N               exclusive end HD frame\n"
         "  --max-frames N\n"
         "  --frame-stride N\n"
         "  --no-save-mesh\n"
         "  --plain                   geometry-only pipeline (no texture "
         "stages)\n"
         "  --occlusion-threshold F    absolute depth tolerance in meters "
         "(default 0.10)\n"
         "  --profile-cuda            bracket frame processing for Nsight "
         "capture\n"
         "  --save-point-cloud          also write <data_name>pc.ply (the\n"
         "                              back-projected input of each frame)\n"
         "  --volume-preset cmu-room|wide (default cmu-room)\n"
         "  --preprocess cpu|gpu        input undistortion + depth-to-color\n"
         "                              projection on the CPU (OpenMP) or\n"
         "                              the GPU, where the results stay for\n"
         "                              reconstruction; same output (default\n"
         "                              gpu)\n"
         "  --preload                   read and decode all frames into host\n"
         "                              memory before the timed loop, as if\n"
         "                              they came from the sensors\n"
         "  --serial                    preprocess each frame right before\n"
         "                              reconstructing it (no overlap with\n"
         "                              the previous frame); e2e_ms is then\n"
         "                              the per-frame latency\n"
         "  -h, --help                  print this message\n"
         "\n"
         "Environment variables:\n"
         "  NUM_KINECTS=N          use the first N Kinects {1..N} "
         "(sensor-count\n"
         "                         sweep; when set > 0 it overrides "
         "--kinect-ids)\n"
         "  VOXEL_SCALE=F          multiply the per-axis voxel count by F\n"
         "  OCCLUSION_TH_SCALE=F   occlusion threshold = F * voxel size\n"
         "  CC_TEMPORAL, CC_DUMP_GAIN_DIR, CC_TRANSFER_PROFILE\n"
         "                         same as the --color-correction-temporal /\n"
         "                         -gain-dump-dir / -profile flags\n";
}

}  // namespace

using grtmesh::FrameReport;
using grtmesh::InputPreprocessor;
using grtmesh::Mesh;
using grtmesh::Mesher;
using grtmesh::MesherParams;
using grtmesh::RgbdFrame;
using grtmesh::RgbdFramePtr;
using grtmesh::VolumeParams;
using grtmesh::app::PanopticKinopticDataset;
using grtmesh::app::preprocess_thread_count;
using grtmesh::app::SaveAsObj;
using grtmesh::app::SavePointCloud;
using grtmesh::app::SetHostThreadCount;
using grtmesh::app::SourceFramePtr;
using grtmesh::app::Stopwatch;
using grtmesh::app::UndistortPreprocessor;
using grtmesh::app::Zfill;
using grtmesh::tex::ColorCorrectionConfig;

const int kOtherAppsThreadsNum = 2;  // For other apps on the PC

int main(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      PrintUsage();
      return 0;
    }
  }
  if (argc < 5) {
    PrintUsage();
    return 1;
  }

  const std::string image_root_dir = std::string(argv[1]) + "/";
  const std::string data_name = argv[2];
  const std::string calib_dir = std::string(argv[3]) + "/";
  const std::string out_root_dir = std::string(argv[4]) + "/";

  PanopticKinopticDataset::Options dataset_options;

  // ---- Color correction + eval CLI (argv[5..]). Environment variables are
  // applied first so the flags can override them.
  ColorCorrectionConfig color_correction_config;
  grtmesh::app::ApplyColorCorrectionEnvironment(color_correction_config);
  int cli_start_frame = -1, cli_end_frame = -1, cli_max_frames = -1;
  int cli_frame_stride = 1;
  bool cli_save_mesh = true;
  bool cli_plain = false;
  bool cli_profile_cuda = false;
  float cli_occlusion_threshold = 0.10f;
  bool cli_save_point_cloud = false;
  std::string cli_volume_preset = "cmu-room";
  bool cli_gpu_preprocess = true;
  bool cli_preload = false;
  bool cli_serial = false;
  for (int i = 5; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (grtmesh::app::ParseColorCorrectionFlag(argc, argv, i,
                                               color_correction_config)) {
      continue;
    }
    if (a == "--kinect-ids" && i + 1 < argc) {
      // Kinects used for reconstruction (empty = all in the calibration).
      // Held-out eval: --kinect-ids 2,4,6,8,10 (evaluate on the odd ones).
      dataset_options.kinect_ids = grtmesh::app::ParseIntList(next());
    } else if (a == "--start-frame" && i + 1 < argc) {
      cli_start_frame = std::stoi(next());
    } else if (a == "--end-frame" && i + 1 < argc) {
      cli_end_frame = std::stoi(next());
    } else if (a == "--max-frames" && i + 1 < argc) {
      cli_max_frames = std::stoi(next());
    } else if (a == "--frame-stride" && i + 1 < argc) {
      cli_frame_stride = std::max(1, std::stoi(next()));
    } else if (a == "--no-save-mesh") {
      cli_save_mesh = false;
    } else if (a == "--plain") {
      cli_plain = true;
    } else if (a == "--profile-cuda") {
      cli_profile_cuda = true;
    } else if (a == "--occlusion-threshold" && i + 1 < argc) {
      cli_occlusion_threshold = std::stof(next());
    } else if (a == "--save-point-cloud") {
      cli_save_point_cloud = true;
    } else if (a == "--volume-preset" && i + 1 < argc) {
      cli_volume_preset = next();
    } else if (a == "--preprocess" && i + 1 < argc) {
      cli_gpu_preprocess = next() == "gpu";
    } else if (a == "--preload") {
      cli_preload = true;
    } else if (a == "--serial") {
      cli_serial = true;
    } else {
      std::cerr << "unknown argument: " << a << std::endl;
      PrintUsage();
      return 1;
    }
  }
  if (cli_plain) {
    color_correction_config.mode = grtmesh::tex::ColorCorrectionMode::kOff;
  }
  if (!(cli_occlusion_threshold > 0.f)) {
    std::cerr << "Occlusion threshold must be positive\n";
    return 1;
  }
  std::cout << "plain=" << cli_plain
            << " occlusion_threshold_m=" << cli_occlusion_threshold
            << " preprocess=" << (cli_gpu_preprocess ? "gpu" : "cpu")
            << " preload=" << cli_preload << " serial=" << cli_serial
            << std::endl;
  std::cout << "color correction: mode="
            << static_cast<int>(color_correction_config.mode)
            << " grid=" << color_correction_config.field_grid_w << "x"
            << color_correction_config.field_grid_h
            << " cpu_solve=" << color_correction_config.cpu_solve
            << " temporal_alpha=" << color_correction_config.temporal_alpha
            << std::endl;
  std::cout << "kinect ids    : ";
  if (dataset_options.kinect_ids.empty()) {
    std::cout << "all (from calibration)";
  } else {
    for (size_t k = 0; k < dataset_options.kinect_ids.size(); ++k) {
      std::cout << (k > 0 ? "," : "") << dataset_options.kinect_ids[k];
    }
  }
  std::cout << std::endl;

  std::cout << "image_root_dir: " << image_root_dir << std::endl;
  std::cout << "data_name     : " << data_name << std::endl;
  std::cout << "calib_dir     : " << calib_dir << std::endl;
  std::cout << "out_root_dir  : " << out_root_dir << std::endl;
  const std::string image_dir = image_root_dir + data_name + "/frames/";
  const std::string out_dir = out_root_dir + data_name;
  std::filesystem::create_directories(out_root_dir);
  std::filesystem::create_directories(out_dir + "mesh/");
  PanopticKinopticDataset dataset(dataset_options);
  dataset.Load(image_dir, calib_dir);

  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  // Timestamp used for the per-run timing log name
  ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");

  std::ofstream outputfile(out_dir + ss.str() + ".csv");
  outputfile << "frame,all,host2device,tsdf,mc,denoise,texture,device2host,"
                "misc,texture_only,inpaint,cc_added_ms,cc_seam_before,cc_seam_"
                "after,gpu_mem_used_mb,pipeline_wall_ms,preprocess_ms,vertices,"
                "faces,occlusion_threshold_m,cc_field_solve_ms,depth_upload_ms,"
                "view_input_setup_ms,view_selection_ms,texture_adjacency_ms,"
                "inpaint_kernel_ms,sdf_truncation_band_m,voxel_size_m,"
                "decode_ms,undistort_ms,e2e_ms,prep_staging_ms,prep_upload_ms,"
                "prep_kernel_ms,download_geometry_ms,download_uv_ms,"
                "view_input_host_copy_ms,view_input_upload_ms,atlas_pack_ms,"
                "atlas_pack_wait_ms,atlas_device_assemble_ms,"
                "cc_atlas_upload_ms,cc_atlas_download_ms,"
                "atlas_final_download_ms"
             << std::endl;

  int start = cli_start_frame >= 0 ? cli_start_frame : 1;
  int end = cli_end_frame >= 0 ? cli_end_frame : start + 1;
  if (cli_max_frames > 0) end = start + cli_max_frames;
  for (int idx = start; idx < end; idx += cli_frame_stride) {
    if (!dataset.Has(static_cast<uint32_t>(idx))) {
      const std::vector<uint32_t>& numbers = dataset.frame_numbers();
      std::cerr << "HD frame " << idx << " is not in " << image_dir
                << " (exported: " << numbers.front() << ".." << numbers.back()
                << ", " << numbers.size() << " frames)" << std::endl;
      return 1;
    }
  }

  // Input frames: read + decoded per frame, or all at once before the loop
  // (--preload) so the timed loop starts from images in host memory.
  std::vector<std::vector<SourceFramePtr>> preloaded;
  std::vector<double> preload_ms;
  auto slot = [&](uint32_t idx) { return (idx - start) / cli_frame_stride; };
  if (cli_preload) {
    const int count = (end - start + cli_frame_stride - 1) / cli_frame_stride;
    preloaded.resize(count);
    preload_ms.resize(count);
    Stopwatch preload_watch;
#pragma omp parallel for schedule(dynamic) num_threads(32)
    for (int k = 0; k < count; ++k) {
      Stopwatch watch;
      preloaded[k] = dataset.Get(start + k * cli_frame_stride);
      preload_ms[k] = watch.ElapsedMs();
    }
    std::cout << "preloaded " << count << " frames in "
              << preload_watch.ElapsedMs() / 1000.0 << " s" << std::endl;
  }
  auto get_source = [&](uint32_t idx, double& decode_ms) {
    if (cli_preload) {
      decode_ms = preload_ms[slot(idx)];
      return preloaded[slot(idx)];
    }
    Stopwatch watch;
    std::vector<SourceFramePtr> source = dataset.Get(idx);
    decode_ms = watch.ElapsedMs();
    return source;
  };

  // GPU preprocessing writes into one of two device slots (the frame being
  // reconstructed and the next one), swapped together with the frames.
  UndistortPreprocessor cpu_preprocessor;
  InputPreprocessor gpu_preprocessor;
  InputPreprocessor::DeviceFrames device_slots[2];
  InputPreprocessor::DeviceFrames* device_slot = &device_slots[0];
  InputPreprocessor::DeviceFrames* next_device_slot = &device_slots[1];
  auto preprocess = [&](const std::vector<SourceFramePtr>& source,
                        std::vector<RgbdFramePtr>& out,
                        InputPreprocessor::DeviceFrames& device,
                        InputPreprocessor::Timing* timing) {
    if (!cli_gpu_preprocess) return cpu_preprocessor.Run(source, out);
    gpu_preprocessor.options.staging_threads = preprocess_thread_count;
    const bool ok =
        gpu_preprocessor.Run(grtmesh::app::ToSensorFrames(source), out, device);
    if (timing != nullptr) *timing = gpu_preprocessor.last_timing();
    return ok;
  };
  double first_decode_ms = 0;
  const std::vector<SourceFramePtr> first_source =
      get_source(start, first_decode_ms);
  if (!(cli_gpu_preprocess
            ? gpu_preprocessor.Init(grtmesh::app::ToSensorFrames(first_source))
            : cpu_preprocessor.Init(first_source))) {
    std::cerr << "Preprocessor initialization failed" << std::endl;
    return 1;
  }
  std::vector<RgbdFramePtr> next_frames;
  preprocess(first_source, next_frames, *next_device_slot, nullptr);
  std::vector<RgbdFramePtr> frames;
  preprocess(first_source, frames, *device_slot, nullptr);

  VolumeParams volume_params;
  if (cli_volume_preset == "wide") {
    volume_params.max_corner = {2.0f, 1.5f, 4.5f};
    volume_params.min_corner = {-2.0f, -1.5f, 1.5f};
    volume_params.voxel_count = {400u, 300u, 300u};
  } else {  // cmu-room
    volume_params.max_corner = {3.0f, 1.5f, 5.6f};
    volume_params.min_corner = {-2.8f, -1.2f, -0.2f};
    volume_params.voxel_count = {290u, 135u, 290u};
  }

  // Eval-only: VOXEL_SCALE multiplies the per-axis voxel count (finer grid for
  // larger values), so the runtime sweep over voxel size can be reproduced
  // without rebuilding. Voxel size = bbox / (voxel_count * VOXEL_SCALE).
  if (const char* vs = std::getenv("VOXEL_SCALE")) {
    float s = std::atof(vs);
    if (s > 0.f) {
      volume_params.voxel_count = (volume_params.voxel_count.cast<float>() * s)
                                      .array()
                                      .round()
                                      .cast<uint32_t>();
      std::cout << "VOXEL_SCALE=" << s
                << " voxel_count=" << volume_params.voxel_count.transpose()
                << std::endl;
    }
  }

  Eigen::Vector3f resolution =
      (volume_params.max_corner - volume_params.min_corner)
          .cwiseQuotient(volume_params.voxel_count.cast<float>());

  size_t gpu_free_before_init = 0, gpu_total = 0;
  cudaMemGetInfo(&gpu_free_before_init, &gpu_total);
  Mesher mesher;
  mesher.Init(volume_params, frames);
  {
    size_t gpu_free_after_init = 0, gpu_total2 = 0;
    cudaMemGetInfo(&gpu_free_after_init, &gpu_total2);
    std::cout << "GPU mem static alloc (Init): "
              << (gpu_free_before_init - gpu_free_after_init) /
                     (1024.0 * 1024.0)
              << " MB, used total: "
              << (gpu_total - gpu_free_after_init) / (1024.0 * 1024.0)
              << " MB / " << gpu_total / (1024.0 * 1024.0) << " MB"
              << std::endl;
  }

  MesherParams params;
  params.noise_removal.label_iterations = 100;
  params.noise_removal.min_component_faces = 500;
  params.noise_removal.convergence_check_interval = 10;
  // Paper-era value (commit 6476910^, 2026-03-04 measurement). A later
  // commit raised this to 100, which inflated the inpainting stage ~1.8x;
  // 5 reproduces the published 24.9 ms breakdown and the published quality.
  params.view_selection.inpaint_iterations = 5;
  params.smooth_face_normals = true;
  params.fusion.truncation_distance = resolution.minCoeff() * 4.f;
  // Fixed absolute depth tolerance (--occlusion-threshold, default 0.10 m)
  // across voxel sizes. OCCLUSION_TH_SCALE=k is the legacy override that
  // reproduces the old k * voxel_size sweeps and takes precedence.
  params.view_selection.occlusion_threshold = cli_occlusion_threshold;
  params.with_texturing = !cli_plain;
  params.build_atlas = !cli_plain;
  if (const char* oth = std::getenv("OCCLUSION_TH_SCALE")) {
    params.view_selection.occlusion_threshold =
        static_cast<float>(std::atof(oth)) * resolution.minCoeff();
  }
  params.color_correction = color_correction_config;

  grtmesh::app::SetHostThreadCount(
      std::max(1u, std::thread::hardware_concurrency() - kOtherAppsThreadsNum -
                       preprocess_thread_count));
  const uint32_t stride = static_cast<uint32_t>(cli_frame_stride);
  if (cli_profile_cuda) cudaProfilerStart();
  // Reused across frames: its vectors keep their capacity, so the per-frame
  // downloads write into already-touched memory (a fresh Mesh costs ~2 ms
  // of first-touch page faults per frame on ~0.6 M-face meshes).
  Mesh mesh;
  for (uint32_t idx = start; idx < end; idx += stride) {
    Stopwatch frame_watch;
    double prepro_time = 0, decode_time = 0, undistort_time = 0;
    double e2e_time = -1;
    InputPreprocessor::Timing gpu_timing;
    std::thread preprocess_thread;
    if (cli_serial) {
      // Latency of one frame: preprocessing + reconstruction, back to back.
      const std::vector<SourceFramePtr> source = get_source(idx, decode_time);
      Stopwatch undistort;
      preprocess(source, frames, *device_slot, &gpu_timing);
      undistort_time = undistort.ElapsedMs();
      prepro_time = undistort_time;
    } else {
      // Preprocess the next frame on another thread while this one is
      // reconstructed (decode/undistort times are those of the next frame).
      preprocess_thread = std::thread([&]() {
        Stopwatch preprocessing;
        if (idx + stride < static_cast<uint32_t>(end)) {
          const std::vector<SourceFramePtr> source =
              get_source(idx + stride, decode_time);
          Stopwatch undistort;
          preprocess(source, next_frames, *next_device_slot, &gpu_timing);
          undistort_time = undistort.ElapsedMs();
        }
        prepro_time = preprocessing.ElapsedMs();
      });
    }

    if (cli_save_point_cloud) {
      SavePointCloud(out_dir + "pc.ply", frames);
    }

    FrameReport report;
    Stopwatch watch;
    const grtmesh::DeviceRgbdBatch device_images = device_slot->batch();
    mesher.Process(frames, params, mesh, &report,
                   cli_gpu_preprocess ? &device_images : nullptr);
    const double all_time = watch.ElapsedMs();
    if (cli_serial) e2e_time = undistort_time + all_time;

    if (preprocess_thread.joinable()) {
      preprocess_thread.join();
      // Hand the preprocessed frames over without copying pixels.
      std::swap(frames, next_frames);
      std::swap(device_slot, next_device_slot);
    }

    if (cli_save_mesh) {
      SaveAsObj(out_dir + "mesh/" + Zfill(idx, 6) + ".obj", mesh);
    }

    const grtmesh::StageTimings& t = report.timings;
    const double texture_all_time = t.atlas_uv_ms + t.view_selection_ms;
    const double device2host_time = t.download_geometry_ms + t.download_uv_ms;
    const double misc_time = all_time - t.upload_ms - t.fusion_ms -
                             t.marching_cubes_ms - t.noise_removal_ms -
                             texture_all_time - device2host_time;
    std::stringstream line;
    line << idx << "," << all_time << "," << t.upload_ms << "," << t.fusion_ms
         << "," << t.marching_cubes_ms << "," << t.noise_removal_ms << ","
         << texture_all_time << "," << device2host_time << "," << misc_time
         << "," << t.atlas_uv_ms << "," << t.view_selection_ms;
    std::cout << line.str() << std::endl;
    double gpu_mem_used_mb = -1.0;
    {
      size_t gpu_free_now = 0, gpu_total_now = 0;
      cudaMemGetInfo(&gpu_free_now, &gpu_total_now);
      gpu_mem_used_mb = (gpu_total_now - gpu_free_now) / (1024.0 * 1024.0);
    }
    const grtmesh::tex::ColorCorrectionStats& cc = report.color_correction;
    outputfile << line.str() << "," << cc.total_added_ms << ","
               << cc.mean_abs_seam_residual_before << ","
               << cc.mean_abs_seam_residual_after << "," << gpu_mem_used_mb
               << "," << frame_watch.ElapsedMs() << "," << prepro_time << ","
               << mesh.vertices.size() << "," << mesh.faces.size() << ","
               << params.view_selection.occlusion_threshold << ","
               << cc.field_solve_ms << "," << report.detailed.depth_upload_ms
               << "," << report.detailed.view_input_setup_ms << ","
               << report.detailed.view_selection_ms << ","
               << report.detailed.texture_adjacency_ms << ","
               << report.detailed.inpaint_kernel_ms << ","
               << *params.fusion.truncation_distance << ","
               << resolution.minCoeff() << "," << decode_time << ","
               << undistort_time << "," << e2e_time << ","
               << gpu_timing.staging_ms << "," << gpu_timing.upload_ms << ","
               << gpu_timing.kernel_ms << "," << t.download_geometry_ms << ","
               << t.download_uv_ms << ","
               << report.detailed.view_input_host_copy_ms << ","
               << report.detailed.view_input_upload_ms << ","
               << report.detailed.atlas_pack_ms << ","
               << report.detailed.atlas_pack_wait_ms << ","
               << report.detailed.atlas_device_assemble_ms << ","
               << cc.atlas_upload_ms << "," << cc.atlas_download_ms << ","
               << report.detailed.atlas_final_download_ms << std::endl;
  }

  if (cli_profile_cuda) cudaProfilerStop();
  return 0;
}
