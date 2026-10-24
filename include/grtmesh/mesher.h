// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_INCLUDE_GRTMESH_MESHER_H_
#define GRTMESH_INCLUDE_GRTMESH_MESHER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "Eigen/Geometry"
#include "grtmesh/frame.h"
#include "grtmesh/mesh.h"
#include "grtmesh/tex/atlas.h"
#include "grtmesh/tex/color_correction.h"

namespace grtmesh {

// Axis-aligned reconstruction volume and its voxel grid, allocated once in
// Mesher::Init. Corners are in depth units.
struct VolumeParams {
  Eigen::Vector3f min_corner;
  Eigen::Vector3f max_corner;
  Eigen::Vector<uint32_t, 3> voxel_count = {64u, 64u, 64u};
};

// Multi-view depth fusion into the TSDF volume.
struct FusionParams {
  // Truncation distance in depth units. Unset (or <= 0) derives it from the
  // voxel size: 4 voxels, the paper setting.
  std::optional<float> truncation_distance;
};

// Connected-component noise removal on the extracted mesh: GPU label
// propagation over the edge-connected faces, then small components are
// dropped.
struct NoiseRemovalParams {
  // Components with fewer faces than this are removed; <= 0 disables the
  // stage. 100-1000 is a sensible range.
  int min_component_faces = 0;
  // Maximum label-propagation iterations; <= 0 disables the stage.
  int label_iterations = 100;
  // Check for convergence every N iterations and stop early; <= 0 means 8.
  int convergence_check_interval = 8;
};

// Per-face view selection, visibility test and propagation of views into
// faces no camera sees.
struct ViewSelectionParams {
  // Depth tolerance of the per-face visibility test, in depth units.
  float occlusion_threshold = 0.1f;
  // Cameras whose viewing direction makes more than this angle with the face
  // normal are not candidates for that face.
  float max_view_angle_deg = 80.f;
  // Iterations of view propagation into unassigned faces; 0 disables it.
  int inpaint_iterations = 20;
};

struct MesherParams {
  FusionParams fusion;
  NoiseRemovalParams noise_removal;
  ViewSelectionParams view_selection;
  // Average vertex normals into the face normals used for view selection.
  bool smooth_face_normals = false;

  // Run the texture stages after the geometry: per-face view selection and
  // propagation, UVs, atlas and seam correction. When false only geometry is
  // produced; Mesh::atlas, uvs and face assignments are left empty.
  bool with_texturing = true;
  // Pack the per-camera color images into Mesh::atlas, generate the UVs and
  // apply the seam correction when enabled. When false only geometry (and the
  // per-face view assignment on the device) is produced.
  bool build_atlas = true;
  grtmesh::tex::AtlasLayout atlas_layout = grtmesh::tex::AtlasLayout::kVertical;
  // Also copy the per-face view assignment into Mesh::face_camera_ids /
  // face_image_coords (one extra device-to-host transfer per frame).
  bool download_face_assignments = false;

  grtmesh::tex::ColorCorrectionConfig color_correction;
};

struct MesherTimingDetails {
  double depth_upload_ms = 0;
  double view_input_setup_ms = 0;
  double view_input_host_copy_ms = 0;  // depth-in-color into pinned
  double view_input_upload_ms = 0;     // its H2D copy
  // Host thread packing the color images into the (pinned) atlas; it runs
  // during the GPU stages and atlas_pack_wait_ms is the part waited for.
  double atlas_pack_ms = 0;
  double atlas_pack_wait_ms = 0;
  double atlas_device_assemble_ms = 0;  // D2D atlas from device images
  double atlas_final_download_ms = 0;   // uncorrected atlas D2H
  double view_selection_ms = 0;
  double texture_adjacency_ms = 0;
  double inpaint_kernel_ms = 0;
};

// Wall-clock time of each stage of one Process call, in milliseconds. The
// atlas is packed on a host thread while the GPU stages run; atlas_ms is the
// time spent waiting for it plus the seam-correction solve and bake.
struct StageTimings {
  double upload_ms = 0.0;  // depth batch into pinned host memory
  double fusion_ms = 0.0;  // H2D of the depth batch + TSDF fusion
  double marching_cubes_ms = 0.0;
  double noise_removal_ms = 0.0;
  double download_geometry_ms = 0.0;  // vertices, faces, normals to the host
  double view_selection_ms = 0.0;     // per-face view selection + propagation
  double atlas_uv_ms = 0.0;           // UV generation on the device
  double download_uv_ms = 0.0;
  double atlas_ms = 0.0;
};

struct FrameReport {
  StageTimings timings;
  MesherTimingDetails detailed;
  grtmesh::tex::ColorCorrectionStats
      color_correction;  // zero when correction is off
};

// GPU pipeline: TSDF fusion -> marching cubes -> connected-component noise
// removal -> per-face view selection -> view propagation -> atlas packing +
// UVs -> seam color correction. Instances are independent and may be
// interleaved; do not call one instance concurrently or change its CUDA
// device after Init.
class Mesher {
 public:
  Mesher();
  ~Mesher();

  // Allocates the volume and the per-camera state. All later frames must have
  // the same camera count, depth size and color size as initial_frames.
  bool Init(const VolumeParams& params,
            const std::vector<RgbdFramePtr>& initial_frames);

  // Reconstructs and textures one frame. Returns false (with mesh.faces empty)
  // when the volume contains no surface. With `device_images` the pixels are
  // read from the device instead of the host images of `frames` (only the
  // GPU seam correction is available then, not cpu_solve).
  bool Process(const std::vector<RgbdFramePtr>& frames,
               const MesherParams& params, Mesh& mesh,
               FrameReport* report = nullptr,
               const DeviceRgbdBatch* device_images = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace grtmesh

#endif  // GRTMESH_INCLUDE_GRTMESH_MESHER_H_
