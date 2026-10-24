// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Texture atlas packing and per-camera color image cleanup.

#include "grtmesh/tex/atlas.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "grtmesh/frame.h"
#include "grtmesh/image.h"
#include "grtmesh/mesh.h"
#include "grtmesh/tex/atlas_tiling.h"

namespace grtmesh::tex {
namespace {

auto CalcProjectedBoundingBoxes(
    const grtmesh::Mesh& mesh,
    const std::vector<grtmesh::RgbdFramePtr>& frames) {
  std::vector<Eigen::Vector2f> bb_mins(
      frames.size(),
      Eigen::Vector2f(static_cast<float>(frames[0]->color.width - 1),
                      static_cast<float>(frames[0]->color.height - 1)));
  std::vector<Eigen::Vector2f> bb_maxs(frames.size(),
                                       Eigen::Vector2f(0.f, 0.f));
  for (size_t i = 0; i < mesh.face_image_coords.size(); i++) {
    const int32_t frame_idx = mesh.face_camera_ids[i];
    if (frame_idx < 0) {
      // Skip faces no camera sees (even after inpainting)
      continue;
    }
    assert(static_cast<size_t>(frame_idx) < frames.size());

    Eigen::Vector2f& bb_min = bb_mins[frame_idx];
    Eigen::Vector2f& bb_max = bb_maxs[frame_idx];

    for (size_t j = 0; j < 3; j++) {
      const Eigen::Vector2f& p = mesh.face_image_coords[i][j];
      if (p.x() < bb_min.x() && 0.f <= p.x()) {
        bb_min.x() = p.x();
      }
      if (p.y() < bb_min.y() && 0.f <= p.y()) {
        bb_min.y() = p.y();
      }
      if (bb_max.x() < p.x() &&
          p.x() <= static_cast<float>(frames[0]->color.width - 1)) {
        bb_max.x() = p.x();
      }
      if (bb_max.y() < p.y() &&
          p.y() <= static_cast<float>(frames[0]->color.height - 1)) {
        bb_max.y() = p.y();
      }
    }
  }

  return std::make_pair(bb_mins, bb_maxs);
}

void FillOutsideBlack(grtmesh::ColorImage& img, uint32_t min_x, uint32_t max_x,
                      uint32_t min_y, uint32_t max_y) {
  uint8_t* pixels = img.data();
  // Top
  if (0 < min_y) {
    std::memset(pixels, 0, (min_y - 1) * img.width * 3 * sizeof(uint8_t));
  }

  // Middle
  const size_t bb_width_size = (max_x - min_x) * 3 * sizeof(uint8_t);
  const size_t l_size = min_x * 3 * sizeof(uint8_t);
  const size_t r_size = ((img.width - 1) - max_x) * 3 * sizeof(uint8_t);
  uint32_t first_row = min_y;
  if (0 < first_row) {
    first_row = first_row - 1;
  }
  for (uint32_t j = first_row; j <= max_y; j++) {
    // Left
    const size_t l_offset = j * img.width * 3 * sizeof(uint8_t);
    std::memset(pixels + l_offset, 0, l_size);

    // Right
    const size_t r_offset = l_offset + l_size + bb_width_size;
    std::memset(pixels + r_offset, 0, r_size);
  }

  // Bottom
  if (max_y < img.height - 1) {
    const size_t offset = max_y * img.width * 3 * sizeof(uint8_t);
    std::memset(pixels + offset, 0,
                ((img.height - 1) - max_y) * img.width * 3 * sizeof(uint8_t));
  }
}

}  // namespace

bool FillColorAreaOutsideMeshBlack(const grtmesh::Mesh& mesh,
                                   std::vector<grtmesh::RgbdFramePtr>& frames,
                                   uint32_t padding) {
  auto [bb_mins, bb_maxs] = CalcProjectedBoundingBoxes(mesh, frames);

  float padding_f = static_cast<float>(padding);
  for (size_t i = 0; i < frames.size(); i++) {
    uint32_t min_x = static_cast<uint32_t>(
        std::max(std::floor(bb_mins[i].x()) - padding_f, 0.f));
    uint32_t min_y = static_cast<uint32_t>(
        std::max(std::floor(bb_mins[i].y()) - padding_f, 0.f));

    uint32_t max_x =
        std::min(static_cast<uint32_t>(std::ceil(bb_maxs[i].x())) + padding,
                 frames[i]->color.width - 1);
    max_x = std::max(max_x, min_x);

    uint32_t max_y =
        std::min(static_cast<uint32_t>(std::ceil(bb_maxs[i].y())) + padding,
                 frames[i]->color.height - 1);
    max_y = std::max(max_y, min_y);

    FillOutsideBlack(frames[i]->color, min_x, max_x, min_y, max_y);
  }
  return true;
}

bool PackAtlas(const std::vector<grtmesh::ColorImage>& images,
               grtmesh::ColorImage& atlas, AtlasLayout layout) {
  if (images.empty()) {
    return false;
  }

  const uint32_t width = images[0].width;
  const uint32_t height = images[0].height;
  const AtlasTiling tiling =
      MakeAtlasTiling(layout, static_cast<int>(images.size()),
                      static_cast<int>(width), static_cast<int>(height));
  const uint32_t atlas_width = static_cast<uint32_t>(tiling.Width());
  const uint32_t atlas_height = static_cast<uint32_t>(tiling.Height());

  // Allocate only when needed: an atlas that already has the right size and
  // storage (e.g. a pinned buffer) is reused.
  if (atlas.empty() || atlas.width != atlas_width ||
      atlas.height != atlas_height) {
    atlas = grtmesh::ColorImage::Create(atlas_width, atlas_height);
  }

  const size_t row_bytes = static_cast<size_t>(width) * 3;
  for (size_t i = 0; i < images.size(); ++i) {
    const grtmesh::ColorImage& src = images[i];
    const int2 origin = tiling.Origin(static_cast<int>(i));
    uint8_t* dst = atlas.data() +
                   (static_cast<size_t>(origin.y) * atlas_width + origin.x) * 3;
    for (uint32_t y = 0; y < height; ++y) {
      std::memcpy(dst + static_cast<size_t>(y) * atlas_width * 3,
                  src.data() + static_cast<size_t>(y) * row_bytes, row_bytes);
    }
  }
  return true;
}

}  // namespace grtmesh::tex
