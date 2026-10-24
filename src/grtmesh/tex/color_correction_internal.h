// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_SRC_GRTMESH_TEX_COLOR_CORRECTION_INTERNAL_H_
#define GRTMESH_SRC_GRTMESH_TEX_COLOR_CORRECTION_INTERNAL_H_

// Internal (library-private) interface for the real-time texture color
// correction: the per-sequence GPU context and the CPU reference solver.

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "grtmesh/frame.h"
#include "grtmesh/image.h"
#include "grtmesh/mesh.h"
#include "grtmesh/tex/atlas.h"
#include "grtmesh/tex/color_correction.h"

namespace grtmesh::tex {

// Persistent device buffers and temporal (EMA) state of one reconstruction
// sequence. Owned by a Mesher; Reset() on Init / sequence changes. Nothing is
// shared between contexts.
class ColorCorrectionContext {
 public:
  ColorCorrectionContext();
  ~ColorCorrectionContext();
  ColorCorrectionContext(const ColorCorrectionContext&) = delete;
  ColorCorrectionContext& operator=(const ColorCorrectionContext&) = delete;
  void Reset();

  // GPU path, two calls per frame:
  //  (1) While the device geometry is live: persist the faces (+ per-face
  //      normals for the fold / veil seam skip) and the face adjacency. When
  //      d_prebuilt_face_adjacency (the view-propagation stage's adjacency of
  //      the same mesh) is given it is copied instead of rebuilt.
  bool PrepareGeometry(const int* d_faces, int num_faces,
                       const float3* d_face_normals,
                       const int* d_prebuilt_face_adjacency);
  //  (2) Once the atlas is packed: upload it, build the low-res proxy on the
  //      device, detect seams, sample residuals, solve the field (Jacobi, then
  //      the EMA when enabled), bake it into the atlas in place and download.
  //      Returns the GPU time in ms, or -1 on failure (the caller falls back to
  //      the CPU path; the atlas may then be partially modified). With
  //      d_atlas_source (the packed atlas already on the device) the input is
  //      copied from there instead of uploaded from `atlas`, which then only
  //      receives the corrected result.
  float SolveAndBake(grtmesh::ColorImage& atlas, const int* d_face_camera_ids,
                     const float2* d_face_image_coords, int num_cams,
                     const ColorCorrectionConfig& config, int image_width,
                     int image_height, AtlasLayout layout,
                     ColorCorrectionStats& stats,
                     const unsigned char* d_atlas_source = nullptr);

  // Bake an already solved per-camera log-gain field (host, laid out
  // [(cam * cells + cell) * nch + ch], cells = grid_w * grid_h, nch = 1 or 3)
  // into the atlas bands on the GPU. Returns the GPU time in ms or -1 on
  // failure (caller should bake on the CPU).
  float Bake(grtmesh::ColorImage& atlas, const std::vector<float>& field,
             int image_width, int image_height, int num_cams, int grid_w,
             int grid_h, int nch, bool rgb, float max_log_gain,
             AtlasLayout layout);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// CPU reference: estimates the per-camera gain field from the host mesh data
// (mesh.faces, face_camera_ids, face_image_coords) and the per-camera source
// images (frames[k]->color), then bakes it into mesh.atlas in place (on the
// GPU unless config.cpu_bake). No-op when config.mode == kOff or there is a
// single camera.
void ApplyColorCorrectionFieldCpu(
    ColorCorrectionContext& context, grtmesh::Mesh& mesh,
    const std::vector<grtmesh::RgbdFramePtr>& frames,
    const ColorCorrectionConfig& config, AtlasLayout layout,
    ColorCorrectionStats& stats);

}  // namespace grtmesh::tex

#endif  // GRTMESH_SRC_GRTMESH_TEX_COLOR_CORRECTION_INTERNAL_H_
