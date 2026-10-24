// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_SRC_GRTMESH_TEX_VIEW_SELECTION_CUH_
#define GRTMESH_SRC_GRTMESH_TEX_VIEW_SELECTION_CUH_

// GPU per-face view selection (+ propagation of views into the faces no camera
// sees) and atlas UV generation. This header is included from host-only
// translation units (grtmesh/mesher.cc), so it must stay free of
// __global__/__device__ code; <cuda_runtime.h> is only needed for
// float2/float3/int3/cudaStream_t.

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <vector>

#include "Eigen/Geometry"
#include "common/device_buffer.h"
#include "common/device_camera.h"
#include "grtmesh/camera.h"
#include "grtmesh/frame.h"
#include "grtmesh/mesh.h"
#include "grtmesh/tex/atlas.h"
#include "grtmesh/tex/atlas_tiling.h"

namespace grtmesh::tex {
struct ViewSelectionTimes {
  double input_setup_ms = 0;
  double input_host_copy_ms = 0;  // depth-in-color into pinned memory
  double input_upload_ms = 0;     // its H2D copy
  double view_selection_ms = 0;
  double adjacency_ms = 0;
  double inpaint_ms = 0;
};

class ViewSelector {
 public:
  ViewSelector();
  ~ViewSelector();
  ViewSelector(const ViewSelector&) = delete;
  ViewSelector& operator=(const ViewSelector&) = delete;

  // Allocates the per-face buffers for up to max_face_num faces and uploads the
  // color cameras. image_width / image_height is the color image size shared
  // by all cameras (in-image test); color_world_to_camera and
  // color_camera_directions (the +Z axis of camera-to-world, world frame) have
  // one entry per camera.
  void Init(int max_face_num, int image_width, int image_height,
            const std::vector<grtmesh::CameraIntrinsics>& color_intrinsics,
            const std::vector<Eigen::Affine3f>& color_world_to_camera,
            const std::vector<Eigen::Vector3f>& color_camera_directions);

  // Per-face view selection (cameras within max_view_angle_deg of the
  // face normal, tested against frames[k]->depth_in_color), then
  // inpaint_iterations rounds of propagating views into unassigned faces (the
  // face adjacency is built when inpaint_iterations > 0). A non-null
  // d_depth_in_color (all cameras, concatenated, on the device) is used
  // instead of uploading the frames' depth_in_color.
  void AssignViews(const std::vector<grtmesh::RgbdFramePtr>& frames,
                   const float3* d_vertices, const int* d_faces,
                   const float3* d_face_normals, int num_faces,
                   float occlusion_threshold, float max_view_angle_deg,
                   int inpaint_iterations,
                   const float* d_depth_in_color = nullptr);

  void DownloadFaceCameraIds(std::vector<int32_t>& face_camera_ids);
  void DownloadFaceImageCoords(
      std::vector<std::array<Eigen::Vector2f, 3>>& face_image_coords);

  // Device pointers, valid until the next AssignViews.
  const int* DeviceFaceCameraIds() const { return d_face_camera_ids_.data(); }
  const float2* DeviceFaceImageCoords() const {
    return d_face_image_coords_.data();
  }
  // Face adjacency built by AssignViews when inpaint_iterations > 0, valid
  // only when FaceAdjacencyFaceCount() equals the current face count; the
  // seam correction reuses it instead of rebuilding it.
  const int* DeviceFaceAdjacency() const { return d_face_adjacency_.data(); }
  int FaceAdjacencyFaceCount() const { return face_adjacency_face_count_; }

  ViewSelectionTimes GetLastTimes() const { return last_times_; }
  void GenerateAtlasUvs(const AtlasTiling& tiling);
  void DownloadAtlasUvs(std::vector<Eigen::Vector2f>& uvs,
                        std::vector<Eigen::Vector3i>& uv_faces);

 private:
  ViewSelectionTimes last_times_;
  void EnsureTriangleMemory(int num_faces);
  void EnsureFaceAdjacencyMemory(int num_faces);

  int image_width_ = 0, image_height_ = 0;
  int num_images_ = 0;
  int face_capacity_ = 0;
  int num_faces_ = 0;
  int face_adjacency_capacity_ = 0;
  int face_adjacency_face_count_ = 0;  // 0 = stale

  grtmesh::DeviceBuffer<grtmesh::DeviceCamera> d_cameras_;
  grtmesh::DeviceBuffer<float> d_depth_in_color_;
  grtmesh::PinnedHostBuffer<float> h_depth_in_color_;

  grtmesh::DeviceBuffer<int> d_face_camera_ids_, d_face_camera_ids_scratch_;
  grtmesh::DeviceBuffer<float2> d_face_image_coords_,
      d_face_image_coords_scratch_;
  grtmesh::PinnedHostBuffer<int> h_face_camera_ids_;
  grtmesh::PinnedHostBuffer<float2> h_face_image_coords_;

  grtmesh::DeviceBuffer<uint64_t> d_edge_key_;
  grtmesh::DeviceBuffer<int> d_edge_face_, d_face_adjacency_;
  grtmesh::DeviceBuffer<int> d_propagation_updated_;

  grtmesh::DeviceBuffer<float2> d_uvs_;
  grtmesh::DeviceBuffer<int3> d_uv_faces_;
  grtmesh::PinnedHostBuffer<float2> h_uvs_;
  grtmesh::PinnedHostBuffer<int3> h_uv_faces_;
};

}  // namespace grtmesh::tex

#endif  // GRTMESH_SRC_GRTMESH_TEX_VIEW_SELECTION_CUH_
