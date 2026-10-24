// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Where each camera's image sits inside the atlas, shared by host code and
// kernels.

#ifndef GRTMESH_SRC_GRTMESH_TEX_ATLAS_TILING_H_
#define GRTMESH_SRC_GRTMESH_TEX_ATLAS_TILING_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include "grtmesh/tex/atlas.h"

namespace grtmesh::tex {

// The atlas is a `columns` x `rows` grid of equally sized camera images,
// camera k at column k % columns, row k / columns. Vertical layout: one
// column; horizontal: one row; tile: ceil(sqrt(N)) columns.
struct AtlasTiling {
  int image_width = 0;
  int image_height = 0;
  int num_images = 0;
  int columns = 1;
  int rows = 1;

  __host__ __device__ int Width() const { return columns * image_width; }
  __host__ __device__ int Height() const { return rows * image_height; }
  __host__ __device__ int2 Origin(int camera) const {
    return make_int2((camera % columns) * image_width,
                     (camera / columns) * image_height);
  }
  // Camera whose tile contains atlas pixel (ax, ay), and the pixel's
  // coordinates inside that image. The result is >= num_images for pixels in
  // an unused tile of the tile layout.
  __host__ __device__ int CameraAt(int ax, int ay, int* x, int* y) const {
    const int kx = ax / image_width;
    const int ky = ay / image_height;
    *x = ax - kx * image_width;
    *y = ay - ky * image_height;
    return ky * columns + kx;
  }
};

inline AtlasTiling MakeAtlasTiling(AtlasLayout layout, int num_images,
                                   int image_width, int image_height) {
  AtlasTiling tiling;
  tiling.image_width = image_width;
  tiling.image_height = image_height;
  tiling.num_images = num_images;
  switch (layout) {
    case AtlasLayout::kHorizontal:
      tiling.columns = num_images;
      tiling.rows = 1;
      break;
    case AtlasLayout::kTile:
      tiling.columns = std::max(1, static_cast<int>(std::ceil(std::sqrt(
                                       static_cast<float>(num_images)))));
      tiling.rows = (num_images + tiling.columns - 1) / tiling.columns;
      break;
    case AtlasLayout::kVertical:
    default:
      tiling.columns = 1;
      tiling.rows = num_images;
      break;
  }
  return tiling;
}

}  // namespace grtmesh::tex

#endif  // GRTMESH_SRC_GRTMESH_TEX_ATLAS_TILING_H_
