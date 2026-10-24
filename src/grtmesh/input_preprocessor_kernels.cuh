// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CUDA kernels of InputPreprocessor. They reproduce the CPU reference of the
// sample apps (OpenCV remap with CV_16SC2 fixed-point maps, the
// far-depth cut and the depth-to-color z-buffer) operation for operation, so
// the two paths give the same images.
#ifndef GRTMESH_SRC_GRTMESH_INPUT_PREPROCESSOR_KERNELS_CUH_
#define GRTMESH_SRC_GRTMESH_INPUT_PREPROCESSOR_KERNELS_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace grtmesh::preprocess_kernels {

// Fixed-point undistortion map as built by cv::initUndistortRectifyMap with
// CV_16SC2: integer source pixel (map1) and 5-bit sub-pixel index (map2,
// fy * 32 + fx). Null maps mean the camera has no lens distortion.
struct RemapMaps {
  const short2* map1 = nullptr;
  const uint16_t* map2 = nullptr;
};

// Pinhole projection of the depth camera into the color camera.
struct DepthToColorParams {
  float depth_fx, depth_fy, depth_cx, depth_cy;
  float color_fx, color_fy, color_cx, color_cy;
  float rotation[9];  // row-major
  float translation[3];
  float scale;  // z-buffer size relative to the color image
};

// Bilinear BGR8 remap (cv::INTER_LINEAR, cv::BORDER_CONSTANT 0).
void RemapColor(const uint8_t* src, uint8_t* dst, int width, int height,
                RemapMaps maps, cudaStream_t stream);

// Nearest-neighbor depth remap (cv::INTER_NEAREST, cv::BORDER_CONSTANT 0)
// followed by depth > far_threshold -> 0.
void RemapDepth(const float* src, float* dst, int width, int height,
                RemapMaps maps, float far_threshold, cudaStream_t stream);

// Depth reprojected into the color image: a z-buffer of zbuffer_width x
// zbuffer_height (scale x the color size) upsampled with nearest neighbor to
// the color size through x_offsets / y_offsets (cv::resize INTER_NEAREST).
// `zbuffer` is scratch of zbuffer_width * zbuffer_height ints.
void ProjectDepthToColor(const float* depth, int depth_width, int depth_height,
                         const DepthToColorParams& params, int* zbuffer,
                         int zbuffer_width, int zbuffer_height,
                         const int* x_offsets, const int* y_offsets,
                         float* depth_in_color, int color_width,
                         int color_height, cudaStream_t stream);

}  // namespace grtmesh::preprocess_kernels

#endif  // GRTMESH_SRC_GRTMESH_INPUT_PREPROCESSOR_KERNELS_CUH_
