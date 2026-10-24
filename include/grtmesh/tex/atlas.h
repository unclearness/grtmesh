// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_INCLUDE_GRTMESH_TEX_ATLAS_H_
#define GRTMESH_INCLUDE_GRTMESH_TEX_ATLAS_H_

#include <cstdint>
#include <vector>

#include "grtmesh/frame.h"
#include "grtmesh/image.h"
#include "grtmesh/mesh.h"

namespace grtmesh::tex {

// How the per-camera images are packed into the single texture atlas.
enum class AtlasLayout { kVertical = 0, kHorizontal, kTile };

// Packs the per-camera images (all the same size) into `atlas`. `atlas` is
// reused when it already has the right size and non-null pixel storage.
bool PackAtlas(const std::vector<grtmesh::ColorImage>& images,
               grtmesh::ColorImage& atlas,
               AtlasLayout layout = AtlasLayout::kVertical);

bool FillColorAreaOutsideMeshBlack(const grtmesh::Mesh& mesh,
                                   std::vector<grtmesh::RgbdFramePtr>& frames,
                                   uint32_t padding = 1);

}  // namespace grtmesh::tex

#endif  // GRTMESH_INCLUDE_GRTMESH_TEX_ATLAS_H_
