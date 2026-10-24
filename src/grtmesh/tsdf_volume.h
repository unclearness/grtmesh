// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_SRC_GRTMESH_TSDF_VOLUME_H_
#define GRTMESH_SRC_GRTMESH_TSDF_VOLUME_H_

#include <cuda_runtime.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <memory>
#include <vector>

#include "grtmesh/camera.h"

namespace grtmesh {

struct TsdfFusionParams {
  float truncation_distance =
      -1.f;  // depth units; see DefaultTruncationDistance
};

// Truncation distance derived from the voxel size: 4 voxels, the paper
// setting (OpenCV kinfu uses 7).
inline constexpr float kDefaultTruncationVoxels = 4.f;
inline float DefaultTruncationDistance(float voxel_size) {
  return kDefaultTruncationVoxels * voxel_size;
}

// One depth camera of a fusion batch.
struct DepthView {
  CameraIntrinsics intrinsics;
  Eigen::Affine3f camera_to_world;
};

// Dense TSDF volume on the GPU: multi-view fusion from raw depth maps, marching
// cubes extraction, connected-component noise removal and normal computation.
// All mesh buffers stay on the device between calls; the Get*Cpu() accessors
// copy through persistent pinned host buffers.
class TsdfVolume {
 public:
  TsdfVolume();
  ~TsdfVolume();

  bool Init(const Eigen::Vector3f& min_corner,
            const Eigen::Vector3f& max_corner,
            const Eigen::Vector<uint32_t, 3>& voxel_count);

  // Fuses views.size() depth images of width x height (concatenated in
  // `depth`, host memory, or device memory when depth_on_device) into the
  // volume.
  void FuseDepthMulti(const float* depth, int width, int height,
                      const std::vector<DepthView>& views,
                      const TsdfFusionParams& params, bool sync = true,
                      double* depth_upload_ms = nullptr,
                      bool depth_on_device = false);
  void ExtractMesh(bool with_face_normals = true);
  void RemoveSmallConnectedComponents(int max_iter, int min_face,
                                      int early_exit_check_interval = -1);
  void ComputeVertexNormals();
  void SmoothFaceNormalsWithVertexNormals();
  void GetVerticesCpu(std::vector<Eigen::Vector3f>& vertices);
  void GetFacesCpu(std::vector<Eigen::Vector3i>& faces);
  void GetFaceNormalsCpu(std::vector<Eigen::Vector3f>& face_normals,
                         bool smoothing = false);

  const float3* GetVerticesGpu() const;
  const int* GetFacesGpu() const;
  const float3* GetFaceNormalsGpu() const;
  const float3* GetSmoothFaceNormalsGpu() const;

  // Resets the voxels (the mesh buffers stay allocated).
  void Clear();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace grtmesh

#endif  // GRTMESH_SRC_GRTMESH_TSDF_VOLUME_H_
