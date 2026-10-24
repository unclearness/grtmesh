// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_INCLUDE_GRTMESH_TEX_COLOR_CORRECTION_H_
#define GRTMESH_INCLUDE_GRTMESH_TEX_COLOR_CORRECTION_H_

#include <cstdint>
#include <string>

namespace grtmesh::tex {

// ---------------------------------------------------------------------------
// Real-time texture color correction (per-camera low-frequency radiometric
// field). The correction is a smooth, low-frequency, per-camera multiplicative
// gain estimated from texture seams and baked into the per-camera bands of the
// atlas.
// ---------------------------------------------------------------------------
enum class ColorCorrectionSpace {
  kLuminanceOnly = 0,  // single log-luminance gain (chroma preserved)
  kLinearRgb,          // independent per-channel log gain
};

enum class ColorCorrectionMode {
  kOff = 0,       // no correction; output identical to the base pipeline
  kGlobalOnly,    // one gain per camera (1x1 field): weak baseline
  kLowFreqField,  // per-camera low-resolution gain field: main method
};

struct ColorCorrectionConfig {
  ColorCorrectionMode mode = ColorCorrectionMode::kOff;
  ColorCorrectionSpace space = ColorCorrectionSpace::kLuminanceOnly;

  // Per-camera low-resolution gain grid (enforces low-frequency-only).
  int field_grid_w = 24;
  int field_grid_h = 16;

  // Matrix-free weighted Jacobi solver (fixed iterations, no convergence wait).
  int field_iterations = 12;
  float field_omega = 0.7f;  // under-relaxation
  float lambda_smooth =
      30.0f;  // per-camera Laplacian (low-freq + fill-in). A
              // strong weight is important: with a weak weight
              // the field over-fits noisy per-cell seam
              // residuals into a blotchy/streaky pattern; a
              // strong weight keeps it smooth and low-frequency
              // (fewer artifacts, better held-out quality).
  float lambda_anchor = 0.02f;  // identity tether (g=0 => gain=1)
  float seam_data_weight =
      1.0f;  // multiplier on the seam data term: raising it
             // makes the field fit the cross-seam residual
             // more strongly (sharper seam correction) at the
             // cost of less smoothing at seam cells.

  int reference_camera = -1;   // gauge anchor; -1 => zero-mean over cameras
  float max_log_gain = 0.15f;  // |g| clamp => gain in [0.86, 1.16]. A tight
                               // cap is essential: a loose limit lets the
                               // field over-correct -- it removes seams but
                               // shifts appearance away from the GT exposure
                               // and hurts re-rendering quality. ~+-15% both
                               // removes seams and improves held-out metrics.

  // Temporal smoothing of the field across frames (EMA):
  // field = temporal_alpha * current + (1 - temporal_alpha) * previous.
  // Values outside (0, 1) disable it. The paper applies 0.15 on every frame,
  // for the videos and the held-out evaluation alike (the default).
  float temporal_alpha = 0.15f;

  // Seam residual measurement (low-res radiometric proxy + edge sampling).
  int proxy_downscale = 8;
  int seam_samples = 8;
  float sample_inset_proxy_pixels =
      0.5f;  // small inset: sample both cameras at
             // ~the shared 3D edge point so the
             // residual is the pure inter-camera
             // exposure difference, not a mix of
             // two different surface points.
  float seam_max_dihedral_cos =
      0.5f;  // skip seams whose two faces fold sharply
             // (n_a.n_b < this): occlusion/veil edges,
             // not a real continuous surface.

  // Reference paths, for verification: solve the field on the CPU instead of
  // CUDA, and/or bake the solved field on the CPU instead of CUDA.
  bool cpu_solve = false;
  bool cpu_bake = false;

  // Diagnostics. gain_dump_dir: write the per-frame gain field to
  // <dir>/gain_NNNNNN.txt (layout [cam][gy][gx][ch], log gains before the
  // clamp). profile_transfers: print the upload / compute / download split of
  // the GPU path to stderr.
  std::string gain_dump_dir;
  bool profile_transfers = false;
};

struct ColorCorrectionStats {
  uint32_t face_count = 0;
  uint32_t adjacency_edge_count = 0;
  uint32_t seam_edge_count = 0;
  uint32_t non_manifold_edge_count = 0;
  uint32_t boundary_edge_count = 0;
  uint32_t invalid_seam_sample_count = 0;
  float mean_abs_seam_residual_before = 0.f;
  float mean_abs_seam_residual_after = 0.f;
  // Stage timings (ms), added on top of the base pipeline.
  float seam_extract_ms = 0.f;
  float residual_sample_ms = 0.f;
  float field_solve_ms = 0.f;
  float bake_ms = 0.f;
  float total_added_ms = 0.f;
  // Inside the GPU solve: the atlas copy into the solver (H2D from the
  // host atlas, or D2D from a device atlas) and the corrected atlas D2H.
  float atlas_upload_ms = 0.f;
  float atlas_download_ms = 0.f;
};

}  // namespace grtmesh::tex

#endif  // GRTMESH_INCLUDE_GRTMESH_TEX_COLOR_CORRECTION_H_
