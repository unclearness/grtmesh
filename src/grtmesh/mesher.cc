// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// GPU meshing + texturing orchestrator (grtmesh::Mesher). One Process call:
// depth upload -> TSDF fusion -> marching cubes -> connected-component noise
// removal -> per-face view selection + propagation -> atlas UVs. Meanwhile a
// host thread packs the atlas, and the seam correction is solved and baked
// into it at the end.

#include "grtmesh/mesher.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/device_buffer.h"
#include "common/stopwatch.h"
#include "grtmesh/camera.h"
#include "grtmesh/frame.h"
#include "grtmesh/mesh.h"
#include "grtmesh/tex/atlas.h"
#include "grtmesh/tex/atlas_tiling.h"
#include "grtmesh/tex/color_correction_internal.h"
#include "grtmesh/tex/view_selection.cuh"
#include "grtmesh/tsdf_volume.h"

namespace grtmesh {
namespace {

// Pinned host buffers for the atlas. A buffer is handed out as the shared
// storage of Mesh::atlas; when the last Image holding it goes away it returns
// to the pool (or is freed when the pool is already gone). Retained output
// meshes therefore keep their pixels, and an output object that nobody else
// shares is reused in place without a new allocation.
class PinnedAtlasPool {
 public:
  PinnedAtlasPool() : state_(std::make_shared<State>()) {}

  std::shared_ptr<uint8_t[]> Acquire(size_t bytes) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    Block block{};
    for (auto it = state_->free_blocks.begin(); it != state_->free_blocks.end();
         ++it) {
      if (it->capacity >= bytes) {
        block = *it;
        state_->free_blocks.erase(it);
        break;
      }
    }
    if (block.ptr == nullptr) {
      // Every free block is too small: drop them instead of hoarding.
      for (const Block& stale : state_->free_blocks) {
        state_->Forget(stale.ptr);
        cudaFreeHost(stale.ptr);
      }
      state_->free_blocks.clear();
      if (cudaMallocHost(&block.ptr, bytes) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
      }
      block.capacity = bytes;
      state_->all_blocks.push_back(block);
    }
    std::weak_ptr<State> weak = state_;
    return std::shared_ptr<uint8_t[]>(block.ptr, [weak, block](uint8_t*) {
      if (std::shared_ptr<State> state = weak.lock()) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->free_blocks.push_back(block);
      } else {
        cudaFreeHost(block.ptr);
      }
    });
  }

  // Capacity of a block handed out by this pool; 0 for foreign storage.
  size_t Capacity(const uint8_t* ptr) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (const Block& block : state_->all_blocks) {
      if (block.ptr == ptr) {
        return block.capacity;
      }
    }
    return 0;
  }

 private:
  struct Block {
    uint8_t* ptr = nullptr;
    size_t capacity = 0;
  };
  struct State {
    std::mutex mutex;
    std::vector<Block> free_blocks;
    std::vector<Block> all_blocks;
    void Forget(const uint8_t* ptr) {
      all_blocks.erase(
          std::remove_if(all_blocks.begin(), all_blocks.end(),
                         [ptr](const Block& b) { return b.ptr == ptr; }),
          all_blocks.end());
    }
    ~State() {
      for (const Block& block : free_blocks) {
        cudaFreeHost(block.ptr);
      }
    }
  };
  std::shared_ptr<State> state_;
};

std::vector<ColorImage> ColorImagesOf(const std::vector<RgbdFramePtr>& frames) {
  std::vector<ColorImage> images(frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    images[i] = frames[i]->color;
  }
  return images;
}

}  // namespace

class Mesher::Impl {
 public:
  bool Init(const VolumeParams& params,
            const std::vector<RgbdFramePtr>& initial_frames) {
    if (initial_frames.empty()) {
      std::cerr << "Mesher::Init: no frames" << std::endl;
      return false;
    }
    num_images_ = static_cast<int>(initial_frames.size());
    depth_width_ = static_cast<int>(initial_frames[0]->depth.width);
    depth_height_ = static_cast<int>(initial_frames[0]->depth.height);
    color_width_ = static_cast<int>(initial_frames[0]->color.width);
    color_height_ = static_cast<int>(initial_frames[0]->color.height);

    correction_context_.Reset();
    if (!h_depths_.ResizeDiscard(static_cast<size_t>(depth_width_) *
                                 depth_height_ * num_images_)) {
      return false;
    }

    voxel_size_ = (params.max_corner - params.min_corner)
                      .cwiseQuotient(params.voxel_count.cast<float>());
    voxel_grid_ = std::make_unique<TsdfVolume>();
    if (!voxel_grid_->Init(params.min_corner, params.max_corner,
                           params.voxel_count)) {
      return false;
    }

    // Color cameras for view selection: intrinsics, world-to-camera and the
    // viewing direction (the +Z column of camera-to-world, OpenCV convention).
    std::vector<CameraIntrinsics> intrinsics(num_images_);
    std::vector<Eigen::Affine3f> world_to_camera(num_images_);
    std::vector<Eigen::Vector3f> directions(num_images_);
    for (int i = 0; i < num_images_; ++i) {
      const Eigen::Affine3f& color_to_world = initial_frames[i]->color_to_world;
      intrinsics[i] = initial_frames[i]->color_intrinsics;
      directions[i] = color_to_world.rotation().col(2);
      world_to_camera[i] = color_to_world.inverse();
    }

#ifdef GRTMESH_LARGE_GRID
    // Grids around 5 mm voxels (~7e8 cells) overflow int32 in total_cells * 15.
    const int64_t total_cells =
        static_cast<int64_t>(params.voxel_count.x() - 1) *
        (params.voxel_count.y() - 1) * (params.voxel_count.z() - 1);
    const int64_t max_face_num = total_cells * 15;  // Theoretical max
    const int reduced_max_face_num = static_cast<int>(std::min<int64_t>(
        max_face_num / params.voxel_count.maxCoeff(),
        std::numeric_limits<int>::max()));  // Practically enough
#else
    const int total_cells = (params.voxel_count.x() - 1) *
                            (params.voxel_count.y() - 1) *
                            (params.voxel_count.z() - 1);
    const int max_face_num = total_cells * 15;  // Theoretical max
    const int reduced_max_face_num =
        max_face_num /
        static_cast<int>(params.voxel_count.maxCoeff());  // Practically enough
#endif
    view_selector_ = std::make_unique<grtmesh::tex::ViewSelector>();
    view_selector_->Init(reduced_max_face_num, color_width_, color_height_,
                         intrinsics, world_to_camera, directions);
    return true;
  }

  bool Process(const std::vector<RgbdFramePtr>& frames,
               const MesherParams& params, Mesh& mesh, FrameReport* report,
               const DeviceRgbdBatch* device_images) {
    FrameReport local_report;
    FrameReport& rep = report != nullptr ? *report : local_report;
    rep = FrameReport{};
    StageTimings& t = rep.timings;

    if (static_cast<int>(frames.size()) != num_images_ ||
        voxel_grid_ == nullptr) {
      std::cerr << "Mesher::Process: expected " << num_images_
                << " frames (Init), got " << frames.size() << std::endl;
      return false;
    }

    const grtmesh::tex::ColorCorrectionConfig& cc = params.color_correction;
    const bool correct = params.with_texturing && params.build_atlas &&
                         cc.mode != grtmesh::tex::ColorCorrectionMode::kOff &&
                         num_images_ > 1;
    // kGlobalOnly is the same solve on a 1x1 field.
    const bool cpu_solve = correct && cc.cpu_solve;
    if (device_images != nullptr && cpu_solve) {
      std::cerr << "Mesher::Process: cpu_solve needs the host images"
                << std::endl;
      return false;
    }
    const grtmesh::tex::AtlasTiling tiling = grtmesh::tex::MakeAtlasTiling(
        params.atlas_layout, num_images_, color_width_, color_height_);

    // Pack the atlas on a host thread while the GPU works; the color images are
    // never modified by the pipeline.
    std::thread pack_thread;
    if (params.with_texturing && params.build_atlas) {
      PrepareAtlasStorage(mesh, tiling);
      if (device_images == nullptr) {
        pack_thread = std::thread([&frames, &mesh, &params, &rep]() {
          Stopwatch pack;
          grtmesh::tex::PackAtlas(ColorImagesOf(frames), mesh.atlas,
                                  params.atlas_layout);
          rep.detailed.atlas_pack_ms = pack.ElapsedMs();
        });
      }
    } else {
      mesh.atlas = ColorImage();
    }
    mesh.uvs.clear();
    mesh.uv_faces.clear();
    mesh.face_camera_ids.clear();
    mesh.face_image_coords.clear();

    Stopwatch stage;

    // Depth batch into pinned memory (unless it is already on the device).
    const size_t depth_pixels =
        static_cast<size_t>(depth_width_) * depth_height_;
    if (device_images == nullptr) {
      for (int i = 0; i < num_images_; ++i) {
        std::memcpy(h_depths_.data() + depth_pixels * i,
                    frames[i]->depth.data(), sizeof(float) * depth_pixels);
      }
    }
    t.upload_ms = stage.ElapsedMs();
    stage.Restart();

    // TSDF fusion.
    TsdfFusionParams fusion_params;
    fusion_params.truncation_distance =
        (params.fusion.truncation_distance &&
         *params.fusion.truncation_distance > 0.f)
            ? *params.fusion.truncation_distance
            : DefaultTruncationDistance(voxel_size_.minCoeff());
    std::vector<DepthView> views(num_images_);
    for (int i = 0; i < num_images_; ++i) {
      views[i].intrinsics = frames[i]->depth_intrinsics;
      views[i].camera_to_world = frames[i]->depth_to_world;
    }
    constexpr bool kSync = true;
    const bool depth_on_device = device_images != nullptr;
    voxel_grid_->FuseDepthMulti(
        depth_on_device ? device_images->depth : h_depths_.data(), depth_width_,
        depth_height_, views, fusion_params, kSync,
        &rep.detailed.depth_upload_ms, depth_on_device);
    t.fusion_ms = stage.ElapsedMs();
    stage.Restart();

    // Marching cubes.
    constexpr bool kWithFaceNormals = true;
    voxel_grid_->ExtractMesh(kWithFaceNormals);
    t.marching_cubes_ms = stage.ElapsedMs();
    stage.Restart();

    // Connected-component noise removal.
    if (params.noise_removal.min_component_faces > 0 &&
        params.noise_removal.label_iterations > 0) {
      voxel_grid_->RemoveSmallConnectedComponents(
          params.noise_removal.label_iterations,
          params.noise_removal.min_component_faces,
          params.noise_removal.convergence_check_interval);
    }
    t.noise_removal_ms = stage.ElapsedMs();

    if (params.smooth_face_normals) {
      voxel_grid_->ComputeVertexNormals();
      voxel_grid_->SmoothFaceNormalsWithVertexNormals();
    }

    // Geometry to the host.
    stage.Restart();
    voxel_grid_->GetVerticesCpu(mesh.vertices);
    voxel_grid_->GetFacesCpu(mesh.faces);
    voxel_grid_->GetFaceNormalsCpu(mesh.face_normals,
                                   params.smooth_face_normals);
    t.download_geometry_ms = stage.ElapsedMs();

    if (mesh.vertices.size() < 3 || mesh.faces.empty() ||
        mesh.face_normals.empty()) {
      voxel_grid_->Clear();
      if (pack_thread.joinable()) {
        pack_thread.join();
      }
      return false;
    }

    if (!params.with_texturing) {
      voxel_grid_->Clear();
      return true;
    }
    // Per-face view selection + propagation into unassigned faces (the
    // fixed-size face adjacency is built on the device).
    const int num_faces = static_cast<int>(mesh.faces.size());
    const float3* d_face_normals = params.smooth_face_normals
                                       ? voxel_grid_->GetSmoothFaceNormalsGpu()
                                       : voxel_grid_->GetFaceNormalsGpu();
    stage.Restart();
    view_selector_->AssignViews(
        frames, voxel_grid_->GetVerticesGpu(), voxel_grid_->GetFacesGpu(),
        d_face_normals, num_faces, params.view_selection.occlusion_threshold,
        params.view_selection.max_view_angle_deg,
        params.view_selection.inpaint_iterations,
        device_images != nullptr ? device_images->depth_in_color : nullptr);
    t.view_selection_ms = stage.ElapsedMs();

    if (params.download_face_assignments || cpu_solve) {
      stage.Restart();
      view_selector_->DownloadFaceCameraIds(mesh.face_camera_ids);
      view_selector_->DownloadFaceImageCoords(mesh.face_image_coords);
      t.download_uv_ms += stage.ElapsedMs();
    }

    const auto view_times = view_selector_->GetLastTimes();
    rep.detailed.view_input_setup_ms = view_times.input_setup_ms;
    rep.detailed.view_input_host_copy_ms = view_times.input_host_copy_ms;
    rep.detailed.view_input_upload_ms = view_times.input_upload_ms;
    rep.detailed.view_selection_ms = view_times.view_selection_ms;
    rep.detailed.texture_adjacency_ms = view_times.adjacency_ms;
    rep.detailed.inpaint_kernel_ms = view_times.inpaint_ms;
    // GPU seam correction: persist the faces (+ raw normals, so seam detection
    // can skip sharp folds / veil triangles at occlusion edges) and the face
    // adjacency while the device geometry is still live. The propagation
    // stage's adjacency is reused when it was built for this mesh.
    float cc_prep_ms = 0.f;
    bool gpu_prepared = false;
    if (correct && !cpu_solve) {
      Stopwatch prep;
      const int* reuse_face_adjacency =
          (params.view_selection.inpaint_iterations > 0 &&
           view_selector_->FaceAdjacencyFaceCount() == num_faces)
              ? view_selector_->DeviceFaceAdjacency()
              : nullptr;
      gpu_prepared = correction_context_.PrepareGeometry(
          voxel_grid_->GetFacesGpu(), num_faces,
          voxel_grid_->GetFaceNormalsGpu(), reuse_face_adjacency);
      cudaDeviceSynchronize();
      cc_prep_ms = static_cast<float>(prep.ElapsedMs());
    }

    voxel_grid_->Clear();

    if (!params.build_atlas) {
      return true;
    }

    // Atlas UVs.
    stage.Restart();
    view_selector_->GenerateAtlasUvs(tiling);
    t.atlas_uv_ms = stage.ElapsedMs();
    stage.Restart();
    view_selector_->DownloadAtlasUvs(mesh.uvs, mesh.uv_faces);
    t.download_uv_ms += stage.ElapsedMs();

    // Atlas + seam correction. Device input: the atlas is assembled from the
    // device images and only the final atlas is downloaded.
    stage.Restart();
    if (pack_thread.joinable()) {
      pack_thread.join();
    }
    rep.detailed.atlas_pack_wait_ms = stage.ElapsedMs();
    const unsigned char* d_atlas = nullptr;
    if (device_images != nullptr) {
      Stopwatch assemble;
      d_atlas = AssembleDeviceAtlas(device_images->color, tiling);
      if (d_atlas == nullptr) {
        return false;
      }
      cudaDeviceSynchronize();
      rep.detailed.atlas_device_assemble_ms = assemble.ElapsedMs();
    }
    bool atlas_on_host = d_atlas == nullptr;
    if (correct) {
      Stopwatch correction;
      bool done = false;
      if (gpu_prepared) {
        const float ms = correction_context_.SolveAndBake(
            mesh.atlas, view_selector_->DeviceFaceCameraIds(),
            view_selector_->DeviceFaceImageCoords(), num_images_, cc,
            color_width_, color_height_, params.atlas_layout,
            rep.color_correction, d_atlas);
        if (ms >= 0.f) {
          rep.color_correction.field_solve_ms = ms;
          done = true;
          atlas_on_host = true;
        }
      }
      if (!done && d_atlas != nullptr) {
        // The CPU reference needs the host images: keep the atlas
        // uncorrected.
        std::cerr << "Mesher::Process: GPU seam correction failed; atlas "
                     "left uncorrected"
                  << std::endl;
      } else if (!done) {
        // CPU reference, also the fallback after a failed GPU bake (which may
        // have modified part of the atlas: repack it from the source images).
        if (static_cast<int>(mesh.face_camera_ids.size()) != num_faces) {
          view_selector_->DownloadFaceCameraIds(mesh.face_camera_ids);
          view_selector_->DownloadFaceImageCoords(mesh.face_image_coords);
        }
        if (gpu_prepared) {
          grtmesh::tex::PackAtlas(ColorImagesOf(frames), mesh.atlas,
                                  params.atlas_layout);
        }
        grtmesh::tex::ApplyColorCorrectionFieldCpu(
            correction_context_, mesh, frames, cc, params.atlas_layout,
            rep.color_correction);
      }
      rep.color_correction.total_added_ms +=
          static_cast<float>(correction.ElapsedMs()) + cc_prep_ms;
    }
    if (!atlas_on_host) {
      Stopwatch download;
      cudaMemcpy(mesh.atlas.data(), d_atlas, mesh.atlas.size_bytes(),
                 cudaMemcpyDeviceToHost);
      rep.detailed.atlas_final_download_ms = download.ElapsedMs();
    }
    t.atlas_ms = stage.ElapsedMs();
    return true;
  }

 private:
  // Gives mesh.atlas storage of the atlas size: the object's own storage when
  // nobody else shares it, otherwise a (pooled) pinned buffer.
  void PrepareAtlasStorage(Mesh& mesh,
                           const grtmesh::tex::AtlasTiling& tiling) {
    const uint32_t aw = static_cast<uint32_t>(tiling.Width());
    const uint32_t ah = static_cast<uint32_t>(tiling.Height());
    const size_t bytes = static_cast<size_t>(aw) * ah * 3;
    if (!mesh.atlas.empty() && mesh.atlas.pixels.use_count() == 1 &&
        atlas_pool_.Capacity(mesh.atlas.data()) >= bytes) {
      mesh.atlas.width = aw;
      mesh.atlas.height = ah;
      return;
    }
    std::shared_ptr<uint8_t[]> storage = atlas_pool_.Acquire(bytes);
    if (storage) {
      mesh.atlas = ColorImage::Wrap(std::move(storage), aw, ah);
    } else {
      mesh.atlas = ColorImage::Create(aw, ah);
    }
  }

  // Packs the device color images into the device atlas (tile layout).
  const unsigned char* AssembleDeviceAtlas(
      const uint8_t* d_color, const grtmesh::tex::AtlasTiling& tiling) {
    const size_t row_bytes = static_cast<size_t>(color_width_) * 3;
    const size_t atlas_bytes =
        static_cast<size_t>(tiling.Width()) * tiling.Height() * 3;
    if (!d_atlas_.Reserve(atlas_bytes)) {
      std::cerr << "Mesher::Process: device atlas allocation failed"
                << std::endl;
      return nullptr;
    }
    if (tiling.columns == 1) {
      // Vertical layout: the atlas is the images back to back.
      cudaMemcpy(d_atlas_.data(), d_color,
                 row_bytes * color_height_ * num_images_,
                 cudaMemcpyDeviceToDevice);
    } else {
      if (tiling.rows * tiling.columns > num_images_) {
        cudaMemset(d_atlas_.data(), 0, atlas_bytes);
      }
      for (int k = 0; k < num_images_; ++k) {
        const int2 o = tiling.Origin(k);
        cudaMemcpy2D(d_atlas_.data() +
                         (static_cast<size_t>(o.y) * tiling.Width() + o.x) * 3,
                     static_cast<size_t>(tiling.Width()) * 3,
                     d_color + row_bytes * color_height_ * k, row_bytes,
                     row_bytes, color_height_, cudaMemcpyDeviceToDevice);
      }
    }
    return cudaGetLastError() == cudaSuccess ? d_atlas_.data() : nullptr;
  }

  int num_images_ = 0;
  int depth_width_ = 0, depth_height_ = 0;
  int color_width_ = 0, color_height_ = 0;
  Eigen::Vector3f voxel_size_ = Eigen::Vector3f::Zero();
  std::unique_ptr<TsdfVolume> voxel_grid_;
  std::unique_ptr<grtmesh::tex::ViewSelector> view_selector_;
  grtmesh::tex::ColorCorrectionContext correction_context_;
  PinnedHostBuffer<float> h_depths_;
  DeviceBuffer<unsigned char> d_atlas_;
  PinnedAtlasPool atlas_pool_;
};

Mesher::Mesher() : impl_(std::make_unique<Impl>()) {}

Mesher::~Mesher() = default;

bool Mesher::Init(const VolumeParams& params,
                  const std::vector<RgbdFramePtr>& initial_frames) {
  return impl_->Init(params, initial_frames);
}

bool Mesher::Process(const std::vector<RgbdFramePtr>& frames,
                     const MesherParams& params, Mesh& mesh,
                     FrameReport* report,
                     const DeviceRgbdBatch* device_images) {
  return impl_->Process(frames, params, mesh, report, device_images);
}

}  // namespace grtmesh
