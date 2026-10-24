// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#include "common/cuda_check.h"
#include "common/stopwatch.h"
#include "grtmesh/mesh_topology.cuh"
#include "grtmesh/tex/view_selection.cuh"

namespace {

using grtmesh::DeviceCamera;

// Per face: keep the K cameras that face it best, test each for the in-image
// and depth (occlusion) conditions and assign the first that passes.
__global__ void AssignViewsKernel(
    const DeviceCamera* __restrict__ cameras, int num_images, int image_width,
    int image_height, int depth_width, int depth_height, float max_angle_dot,
    float occlusion_threshold, const float* __restrict__ d_depth_in_color,
    const float3* __restrict__ d_vertices,      // [num_vertices]
    const int* __restrict__ d_faces,            // [num_faces*3]
    const float3* __restrict__ d_face_normals,  // [num_faces]
    int* __restrict__ d_face_camera_ids,        // [num_faces]
    float2* __restrict__ d_face_image_coords,   // [num_faces*3]
    int num_faces) {
  int fid = blockIdx.x * blockDim.x + threadIdx.x;
  if (fid >= num_faces) {
    return;
  }

  const float3 fn = d_face_normals[fid];

  // Keep K views
  constexpr int kViewCandidates = 4;
  int cand_cam[kViewCandidates];
  float cand_dot[kViewCandidates];
  int cand_cnt = 0;

  for (int c = 0; c < num_images; ++c) {
    const float* cd = cameras[c].dir;
    // dot product of face normal and camera direction (OpenCV, Z-toward) must
    // be negative if the camera is looking at the face
    float dot = fn.x * cd[0] + fn.y * cd[1] + fn.z * cd[2];
    if (dot >= 0.f || -dot < max_angle_dot) {
      continue;
    }
    // Keep K views by inserting the current camera
    int pos = cand_cnt;
    while (pos > 0 && dot < cand_dot[pos - 1]) {
      if (pos < kViewCandidates) {
        cand_dot[pos] = cand_dot[pos - 1];
        cand_cam[pos] = cand_cam[pos - 1];
      }
      --pos;
    }
    if (pos < kViewCandidates) {
      cand_dot[pos] = dot;
      cand_cam[pos] = c;
      if (cand_cnt < kViewCandidates) ++cand_cnt;
    }
  }

  // Initial values
  d_face_camera_ids[fid] = grtmesh::kNoCamera;
  for (int v = 0; v < 3; ++v) {
    d_face_image_coords[fid * 3 + v] = make_float2(-1.f, -1.f);
  }

  // Test each candidate camera
  for (int i = 0; i < cand_cnt; ++i) {
    int c = cand_cam[i];
    const float* camera_rotation = cameras[c].r;
    const float* t = cameras[c].t;
    const float fx = cameras[c].fx;
    const float fy = cameras[c].fy;
    const float cx_ = cameras[c].cx;
    const float cy_ = cameras[c].cy;
    const int dw = depth_width;
    const int dh = depth_height;
    const float* depth = d_depth_in_color + (c * dw * dh);

    bool all_ok = true;
    float2 tri[3];

    for (int v = 0; v < 3; ++v) {
      int vid = d_faces[fid * 3 + v];
      float3 w = d_vertices[vid];

      // world -> camera
      float cx = camera_rotation[0] * w.x + camera_rotation[1] * w.y +
                 camera_rotation[2] * w.z + t[0];
      float cy = camera_rotation[3] * w.x + camera_rotation[4] * w.y +
                 camera_rotation[5] * w.z + t[1];
      float cz = camera_rotation[6] * w.x + camera_rotation[7] * w.y +
                 camera_rotation[8] * w.z + t[2];

      float px = fx * (cx / cz) + cx_;
      float py = fy * (cy / cz) + cy_;

      // Inside image test
      if (px < 0.f || px >= image_width || py < 0.f || py >= image_height) {
        all_ok = false;
        break;
      }
      tri[v] = make_float2(px, py);

      // Depth test
      int ix = min(max(static_cast<int>(px + 0.5f), 0), dw - 1);
      int iy = min(max(static_cast<int>(py + 0.5f), 0), dh - 1);
      float dmap = depth[iy * dw + ix];
      if (dmap <= 0.f || fabsf(dmap - cz) > occlusion_threshold) {
        all_ok = false;
        break;
      }
    }
    if (all_ok) {
      d_face_camera_ids[fid] = c;
      for (int v = 0; v < 3; ++v) {
        d_face_image_coords[fid * 3 + v] = tri[v];
      }
      break;
    }
  }
}

// One propagation round: an unassigned face takes the camera of its first
// assigned neighbor and that neighbor's first image coordinate for all three
// vertices (a uniform color, like a vertex color).
__global__ void PropagateViewsKernel(
    const int* __restrict__ d_face_adjacency,  // [num_faces*3], -1 if none
    const int* __restrict__ d_camera_ids_in,   // [num_faces]
    const float2* __restrict__ d_coords_in,    // [num_faces*3]
    int* __restrict__ d_camera_ids_out,        // [num_faces]
    float2* __restrict__ d_coords_out,         // [num_faces*3]
    int num_faces, int* __restrict__ d_updated) {
  int fid = blockIdx.x * blockDim.x + threadIdx.x;
  if (fid >= num_faces) return;

  int current = d_camera_ids_in[fid];

  // If the face is already valid, just copy.
  if (current != grtmesh::kNoCamera) {
    d_camera_ids_out[fid] = current;
#pragma unroll
    for (int v = 0; v < 3; ++v) {
      d_coords_out[fid * 3 + v] = d_coords_in[fid * 3 + v];
    }
    return;
  }

// If invalid, copy the value of the first valid neighbor.
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    int nb = d_face_adjacency[fid * 3 + k];
    if (nb < 0) continue;

    int nb_id = d_camera_ids_in[nb];
    if (nb_id != grtmesh::kNoCamera) {
      d_camera_ids_out[fid] = nb_id;

      // Replicate a single UV to force "uniform color" like vertex-color.
      float2 rep = d_coords_in[nb * 3 + 0];
#pragma unroll
      for (int v = 0; v < 3; ++v) {
        d_coords_out[fid * 3 + v] = rep;
      }

      atomicExch(d_updated, 1);
      return;
    }
  }

  // Still invalid.
  d_camera_ids_out[fid] = grtmesh::kNoCamera;
#pragma unroll
  for (int v = 0; v < 3; ++v) {
    d_coords_out[fid * 3 + v] = d_coords_in[fid * 3 + v];
  }
}

// Atlas UVs of each face from its image coordinates and its camera's tile.
// The tile layout samples pixel centers (+0.5), the band layouts do not.
__global__ void GenerateAtlasUvKernel(
    const int* __restrict__ d_face_camera_ids,
    const float2* __restrict__ d_face_image_coords, float2* __restrict__ d_uvs,
    int3* __restrict__ d_uv_faces, int num_faces,
    grtmesh::tex::AtlasTiling tiling, int pixel_center) {
  int fid = blockIdx.x * blockDim.x + threadIdx.x;
  if (fid >= num_faces) {
    return;
  }

  int camera = d_face_camera_ids[fid];
  if (camera < 0 || tiling.num_images <= camera) {
    // If the face is not assigned to any camera, skip with zero UVs
    for (int k = 0; k < 3; ++k) {
      d_uvs[fid * 3 + k] = make_float2(0.f, 0.f);
    }
    d_uv_faces[fid] = make_int3(fid * 3, fid * 3 + 1, fid * 3 + 2);
    return;
  }

  const int2 origin = tiling.Origin(camera);
  const float offset_x = static_cast<float>(origin.x);
  const float offset_y = static_cast<float>(origin.y);
  const float atlas_w = static_cast<float>(tiling.Width());
  const float atlas_h = static_cast<float>(tiling.Height());

  for (int k = 0; k < 3; ++k) {
    float2 p = d_face_image_coords[fid * 3 + k];
    float u, v;
    if (pixel_center != 0) {
      u = ((p.x + offset_x) + 0.5f) / atlas_w;
      v = 1.0f - (((p.y + offset_y) + 0.5f) / atlas_h);
    } else {
      u = (p.x + offset_x) / atlas_w;
      v = 1.f - (p.y + offset_y) / atlas_h;
    }
    d_uvs[fid * 3 + k] = make_float2(u, v);
  }
  d_uv_faces[fid] = make_int3(fid * 3, fid * 3 + 1, fid * 3 + 2);
}

}  // namespace

namespace grtmesh::tex {

namespace {

void Require(bool ok, const char* what) {
  if (!ok) {
    GRTMESH_CHECK_CUDA_ERRORS(cudaErrorMemoryAllocation);
  }
  (void)what;
}

}  // namespace

ViewSelector::ViewSelector() = default;

ViewSelector::~ViewSelector() = default;

void ViewSelector::Init(
    const int max_face_num, int image_width, int image_height,
    const std::vector<grtmesh::CameraIntrinsics>& color_intrinsics,
    const std::vector<Eigen::Affine3f>& color_world_to_camera,
    const std::vector<Eigen::Vector3f>& color_camera_directions) {
  const int num_images = static_cast<int>(color_intrinsics.size());
  num_images_ = num_images;
  image_width_ = image_width;
  image_height_ = image_height;
  face_capacity_ = 0;
  face_adjacency_capacity_ = 0;
  face_adjacency_face_count_ = 0;

  EnsureTriangleMemory(max_face_num);

  std::vector<DeviceCamera> cameras(num_images);
  for (int i = 0; i < num_images; i++) {
    const Eigen::Affine3f& w2c = color_world_to_camera[i];
    const Eigen::Vector3f& direction = color_camera_directions[i];
    const grtmesh::CameraIntrinsics& intrinsics = color_intrinsics[i];
    DeviceCamera& cam = cameras[i];
    cam.fx = intrinsics.fx;
    cam.fy = intrinsics.fy;
    cam.cx = intrinsics.cx;
    cam.cy = intrinsics.cy;
    // Eigen is column-major in memory; the accessor gives row-major order.
    const Eigen::Matrix3f rotation = w2c.rotation().cast<float>();
    const Eigen::Vector3f t = w2c.translation().cast<float>();
    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < 3; j++) {
        cam.r[k * 3 + j] = rotation(k, j);
      }
      cam.t[k] = t[k];
      cam.dir[k] = direction[k];
    }
  }
  Require(d_cameras_.ResizeDiscard(num_images), "cameras");
  cudaMemcpy(d_cameras_.data(), cameras.data(),
             sizeof(DeviceCamera) * num_images, cudaMemcpyHostToDevice);

  const size_t depth_elements =
      static_cast<size_t>(image_width) * image_height * num_images;
  Require(d_depth_in_color_.ResizeDiscard(depth_elements), "depth");
  Require(h_depth_in_color_.ResizeDiscard(depth_elements), "depth");
  Require(d_propagation_updated_.ResizeDiscard(1), "flag");
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
}

void ViewSelector::AssignViews(const std::vector<grtmesh::RgbdFramePtr>& frames,
                               const float3* d_vertices, const int* d_faces,
                               const float3* d_face_normals, int num_faces,
                               float occlusion_threshold,
                               float max_view_angle_deg, int inpaint_iterations,
                               const float* d_depth_in_color) {
  const int num_images = static_cast<int>(frames.size());
  last_times_ = {};
  grtmesh::Stopwatch stage_timer;

  if (num_images != num_images_) {
    std::cerr << "Number of images does not match: " << num_images
              << " != " << num_images_;
    return;
  }

  if (num_faces < 1) {
    return;
  }

  num_faces_ = num_faces;

  EnsureTriangleMemory(num_faces_);

  const float max_angle_dot = std::cos(grtmesh::Radians(max_view_angle_deg));

  const int depth_width = static_cast<int>(frames[0]->depth_in_color.width);
  const int depth_height = static_cast<int>(frames[0]->depth_in_color.height);
  const size_t depth_pixels = static_cast<size_t>(depth_width) * depth_height;
  if (d_depth_in_color == nullptr) {
    Require(d_depth_in_color_.ResizeDiscard(depth_pixels * num_images),
            "depth");
    Require(h_depth_in_color_.ResizeDiscard(depth_pixels * num_images),
            "depth");

    // Copy to concatenated pinned memory
    grtmesh::Stopwatch copy_timer;
    for (int i = 0; i < num_images; ++i) {
      std::memcpy(h_depth_in_color_.data() + i * depth_pixels,
                  frames[i]->depth_in_color.data(),
                  sizeof(float) * depth_pixels);
    }
    last_times_.input_host_copy_ms = copy_timer.ElapsedMs();
    copy_timer.Restart();

    cudaMemcpy(d_depth_in_color_.data(), h_depth_in_color_.data(),
               sizeof(float) * depth_pixels * num_images,
               cudaMemcpyHostToDevice);
    last_times_.input_upload_ms = copy_timer.ElapsedMs();
    d_depth_in_color = d_depth_in_color_.data();
  }
  last_times_.input_setup_ms = stage_timer.ElapsedMs();
  stage_timer.Restart();

  // Initial projection
  const int threads = 256;
  const int blocks = std::max(1, (num_faces + threads - 1) / threads);
  AssignViewsKernel<<<blocks, threads>>>(
      d_cameras_.data(), num_images_, image_width_, image_height_, depth_width,
      depth_height, max_angle_dot, occlusion_threshold, d_depth_in_color,
      d_vertices, d_faces, d_face_normals, d_face_camera_ids_.data(),
      d_face_image_coords_.data(), num_faces);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  GRTMESH_CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  face_adjacency_face_count_ = 0;  // invalidate; set below once (re)built
  last_times_.view_selection_ms = stage_timer.ElapsedMs();
  if (inpaint_iterations < 1) {
    return;
  }

  stage_timer.Restart();
  EnsureFaceAdjacencyMemory(num_faces);
  grtmesh::BuildFaceAdjacency(d_faces, num_faces, d_edge_key_.data(),
                              d_edge_face_.data(), d_face_adjacency_.data());
  face_adjacency_face_count_ = num_faces;
  last_times_.adjacency_ms = stage_timer.ElapsedMs();
  stage_timer.Restart();

  // Propagation with ping-pong buffers
  int* d_in_ids = d_face_camera_ids_.data();
  int* d_out_ids = d_face_camera_ids_scratch_.data();
  float2* d_in_coords = d_face_image_coords_.data();
  float2* d_out_coords = d_face_image_coords_scratch_.data();

  int h_updated = 0;
  for (int iter = 0; iter < inpaint_iterations; ++iter) {
    cudaMemset(d_propagation_updated_.data(), 0, sizeof(int));

    PropagateViewsKernel<<<blocks, threads>>>(
        d_face_adjacency_.data(), d_in_ids, d_in_coords, d_out_ids,
        d_out_coords, num_faces, d_propagation_updated_.data());
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    GRTMESH_CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    cudaMemcpy(&h_updated, d_propagation_updated_.data(), sizeof(int),
               cudaMemcpyDeviceToHost);

    // If no update, break
    if (h_updated == 0) {
      break;
    }

    // Swap addresses to ping-pong
    std::swap(d_in_ids, d_out_ids);
    std::swap(d_in_coords, d_out_coords);
  }

  // If the input buffer is not the primary one after the ping-pong, copy it
  // back to the primary one.
  if (d_in_ids != d_face_camera_ids_.data()) {
    cudaMemcpy(d_face_camera_ids_.data(), d_in_ids, sizeof(int) * num_faces,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_face_image_coords_.data(), d_in_coords,
               sizeof(float2) * num_faces * 3, cudaMemcpyDeviceToDevice);
  }
  last_times_.inpaint_ms = stage_timer.ElapsedMs();
}

void ViewSelector::DownloadFaceCameraIds(
    std::vector<int32_t>& face_camera_ids) {
  face_camera_ids.resize(num_faces_);
  if (num_faces_ < 1) {
    return;
  }
  cudaMemcpy(h_face_camera_ids_.data(), d_face_camera_ids_.data(),
             sizeof(int32_t) * num_faces_, cudaMemcpyDeviceToHost);
  std::memcpy(face_camera_ids.data(), h_face_camera_ids_.data(),
              sizeof(int32_t) * num_faces_);
}

void ViewSelector::DownloadFaceImageCoords(
    std::vector<std::array<Eigen::Vector2f, 3>>& face_image_coords) {
  face_image_coords.resize(num_faces_);
  if (num_faces_ < 1) {
    return;
  }
  cudaMemcpy(h_face_image_coords_.data(), d_face_image_coords_.data(),
             sizeof(float2) * num_faces_ * 3, cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_faces_; ++i) {
    for (int j = 0; j < 3; ++j) {
      face_image_coords[i][j] =
          Eigen::Vector2f(h_face_image_coords_.data()[i * 3 + j].x,
                          h_face_image_coords_.data()[i * 3 + j].y);
    }
  }
}

void ViewSelector::EnsureTriangleMemory(int num_faces) {
  if (num_faces <= face_capacity_) {
    return;
  }
  face_capacity_ = num_faces;
  const size_t f = static_cast<size_t>(num_faces);
  Require(d_face_camera_ids_.ResizeDiscard(f), "face camera ids");
  Require(d_face_image_coords_.ResizeDiscard(f * 3), "face image coords");
  Require(h_face_camera_ids_.ResizeDiscard(f), "face camera ids");
  Require(h_face_image_coords_.ResizeDiscard(f * 3), "face image coords");
  Require(d_face_camera_ids_scratch_.ResizeDiscard(f), "scratch");
  Require(d_face_image_coords_scratch_.ResizeDiscard(f * 3), "scratch");
  Require(d_uvs_.ResizeDiscard(f * 3), "uvs");
  Require(d_uv_faces_.ResizeDiscard(f), "uv faces");
  Require(h_uvs_.ResizeDiscard(f * 3), "uvs");
  Require(h_uv_faces_.ResizeDiscard(f), "uv faces");
}

void ViewSelector::EnsureFaceAdjacencyMemory(int num_faces) {
  if (num_faces <= face_adjacency_capacity_) {
    return;
  }
  face_adjacency_capacity_ = num_faces;
  const size_t e = static_cast<size_t>(3) * num_faces;
  Require(d_edge_key_.ResizeDiscard(e), "edge keys");
  Require(d_edge_face_.ResizeDiscard(e), "edge faces");
  Require(d_face_adjacency_.ResizeDiscard(e), "face adjacency");
}

void ViewSelector::GenerateAtlasUvs(const AtlasTiling& tiling) {
  const int num_faces = num_faces_;
  if (num_faces < 1) {
    return;
  }

  const int threads = 256;
  const int blocks = (num_faces + threads - 1) / threads;
  const int pixel_center = (tiling.columns > 1 && tiling.rows > 1) ? 1 : 0;
  GenerateAtlasUvKernel<<<blocks, threads>>>(
      d_face_camera_ids_.data(), d_face_image_coords_.data(), d_uvs_.data(),
      d_uv_faces_.data(), num_faces, tiling, pixel_center);

  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  GRTMESH_CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}

void ViewSelector::DownloadAtlasUvs(std::vector<Eigen::Vector2f>& uvs,
                                    std::vector<Eigen::Vector3i>& uv_faces) {
  const int num_faces = num_faces_;
  if (num_faces < 1) {
    uvs.clear();
    uv_faces.clear();
    return;
  }

  cudaMemcpy(h_uvs_.data(), d_uvs_.data(), sizeof(float2) * num_faces * 3,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_uv_faces_.data(), d_uv_faces_.data(), sizeof(int3) * num_faces,
             cudaMemcpyDeviceToHost);

  // Eigen::Vector2f / Vector3i are packed float[2] / int[3], bit-compatible
  // with float2 / int3: one memcpy each instead of a per-element repack loop
  // (~1 ms/frame on ~0.6 M-face meshes).
  static_assert(sizeof(Eigen::Vector2f) == sizeof(float2));
  static_assert(sizeof(Eigen::Vector3i) == sizeof(int3));
  uvs.resize(num_faces * 3);
  std::memcpy(uvs.data(), h_uvs_.data(), sizeof(float2) * num_faces * 3);
  uv_faces.resize(num_faces);
  std::memcpy(uv_faces.data(), h_uv_faces_.data(), sizeof(int3) * num_faces);
}

}  // namespace grtmesh::tex
