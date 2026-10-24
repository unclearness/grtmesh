// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cub/cub.cuh>
#include <iostream>
#include <memory>
#include <vector>

#include "common/cuda_check.h"
#include "common/device_buffer.h"
#include "common/device_camera.h"
#include "grtmesh/mesh_topology.cuh"
#include "grtmesh/tsdf_volume.h"

namespace {

// Marching Cubes lookup tables (Lorensen and Cline edge / triangle tables).
__device__ __constant__ int d_edge_table[256] = {
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f,
    0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99,  0x393, 0x29a, 0x596, 0x49f,
    0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230,
    0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936,
    0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5,
    0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66,  0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a,
    0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453,
    0x55a, 0x256, 0x35f, 0x55,  0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
    0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,  0xfcc,
    0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca,
    0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc,  0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
    0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55,
    0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6,
    0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f,
    0x66,  0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af,
    0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
    0x13a, 0x33,  0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895,
    0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190, 0xf00, 0xe09,
    0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a,
    0x203, 0x109, 0x0};

__device__ __constant__ int d_tri_table[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

//
// Device-side utility functions
//
__device__ inline float dot(const float3 a, const float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float3 operator+(const float3 a, const float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(const float3 a, const float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 operator*(const float3 a, const float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ inline float3 operator/(const float3 a, const float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3 cross(const float3 a, const float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ inline float3 normalize(const float3 a) {
  float len = sqrtf(dot(a, a));
  return (len > 0.0f) ? a / len : make_float3(0.0f, 0.0f, 0.0f);
}

__device__ inline float3 VoxelIndexToPosition(int3 idx, float3 bb_min,
                                              float3 resolution) {
  return make_float3(bb_min.x + idx.x * resolution.x,
                     bb_min.y + idx.y * resolution.y,
                     bb_min.z + idx.z * resolution.z);
}

struct TsdfVoxel {
  float tsdf{0.f};    // weighted mean of the fused truncated signed distances
  float weight{0.f};  // accumulated observation weight; 0 = never observed
};

__global__ void FuseDepthKernel(const float* d_depth, int width, int height,
                                int num_images,
                                const grtmesh::DeviceCamera* cameras,
                                TsdfVoxel* d_voxel, float3 voxel_size,
                                float3 bb_min, int3 voxel_num,
                                float truncation_distance) {
  int x_index = blockIdx.x * blockDim.x + threadIdx.x;
  int y_index = blockIdx.y * blockDim.y + threadIdx.y;
  int z_index = blockIdx.z * blockDim.z + threadIdx.z;

  if (x_index < 0 || voxel_num.x <= x_index || y_index < 0 ||
      voxel_num.y <= y_index || z_index < 0 || voxel_num.z <= z_index) {
    return;
  }

  float3 voxel_pos;
  voxel_pos.x = bb_min.x + x_index * voxel_size.x;
  voxel_pos.y = bb_min.y + y_index * voxel_size.y;
  voxel_pos.z = bb_min.z + z_index * voxel_size.z;

  int vidx =
      x_index + y_index * voxel_num.x + z_index * voxel_num.x * voxel_num.y;
  TsdfVoxel* voxel = &d_voxel[vidx];

  for (int n = 0; n < num_images; n++) {
    const grtmesh::DeviceCamera& cam = cameras[n];
    const float* r_inv = cam.r;
    const float* t_inv = cam.t;

    float3 voxel_pos_cam;
    voxel_pos_cam.z = voxel_pos.x * r_inv[6] + voxel_pos.y * r_inv[7] +
                      voxel_pos.z * r_inv[8] + t_inv[2];

    voxel_pos_cam.x = voxel_pos.x * r_inv[0] + voxel_pos.y * r_inv[1] +
                      voxel_pos.z * r_inv[2] + t_inv[0];
    voxel_pos_cam.y = voxel_pos.x * r_inv[3] + voxel_pos.y * r_inv[4] +
                      voxel_pos.z * r_inv[5] + t_inv[1];

    const float fx = cam.fx;
    const float fy = cam.fy;
    const float cx = cam.cx;
    const float cy = cam.cy;

    if (voxel_pos_cam.z <= 0.f) {
      continue;
    }
    float invz = 1.f / voxel_pos_cam.z;
    float u_f = voxel_pos_cam.x * fx * invz + cx;
    float v_f = voxel_pos_cam.y * fy * invz + cy;

    int x = __float2int_rn(u_f);
    int y = __float2int_rn(v_f);

    if ((unsigned)x >= (unsigned)width || (unsigned)y >= (unsigned)height) {
      continue;
    }

    int img_idx = (n * height + y) * width + x;

    float d = d_depth[img_idx];

    if (d <= 0) {
      continue;
    }

    float sdf_cam = d - voxel_pos_cam.z;
    float xn = (u_f - cx) / fx;
    float yn = (v_f - cy) / fy;
    float sdf = sdf_cam * sqrtf(xn * xn + yn * yn + 1.f);

    if (sdf >= -truncation_distance) {
      sdf = fminf(1.0f, sdf / truncation_distance);
    } else {
      continue;
    }

    // Running weighted average (KinectFusion); every observation has weight 1.
    const float new_weight = voxel->weight + 1.f;
    voxel->tsdf = (voxel->tsdf * voxel->weight + sdf) / new_weight;
    voxel->weight = new_weight;
  }
}

__device__ float3 VertexInterp(float3 p1, float3 p2, float valp1, float valp2,
                               float iso_level = 0.f) {
  float diff = valp2 - valp1;
  float t = (fabsf(diff) < 1e-6f) ? 0.5f : (iso_level - valp1) / diff;
  return make_float3(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y),
                     p1.z + t * (p2.z - p1.z));
}

// edgeId: 0-11 (follows the Marching Cubes convention)
// The edge table has ~3N entries, which exceeds int32 for room-scale grids
// around 5 mm voxels (N ~ 7e8). GRTMESH_LARGE_GRID widens the edge key; the
// default build keeps the int arithmetic unchanged.
#ifdef GRTMESH_LARGE_GRID
using EdgeKey = int64_t;
#else
using EdgeKey = int;
#endif

__device__ EdgeKey ComputeEdgeKey(int ix, int iy, int iz, int edge_id,
                                  int source_nx, int source_ny, int source_nz) {
  const EdgeKey nx = source_nx, ny = source_ny, nz = source_nz;
  EdgeKey x_count = (nx - 1) * ny * nz;
  EdgeKey y_count = nx * (ny - 1) * nz;
  // zCount = nx*ny*(nz-1)  // used by the cases below

  switch (edge_id) {
    // --- X and Y edges on the bottom face (z) ---
    case 0:  // corner 0-1, X edge at (ix, iy, iz)
      return ix + iy * (nx - 1) + iz * (nx - 1) * ny;
    case 1:  // corner 1-2, Y edge at (ix+1, iy, iz)
      return x_count + (ix + 1) + iy * nx + iz * nx * (ny - 1);
    case 2:  // corner 2-3, X edge at (ix, iy+1, iz)
      return ix + (iy + 1) * (nx - 1) + iz * (nx - 1) * ny;
    case 3:  // corner 3-0, Y edge at (ix, iy, iz)
      return x_count + ix + iy * nx + iz * nx * (ny - 1);

    // --- X and Y edges on the top face (z+1) ---
    case 4:  // corner 4-5, X edge at (ix, iy, iz+1)
      return ix + iy * (nx - 1) + (iz + 1) * (nx - 1) * ny;
    case 5:  // corner 5-6, Y edge at (ix+1, iy, iz+1)
      return x_count + (ix + 1) + iy * nx + (iz + 1) * nx * (ny - 1);
    case 6:  // corner 6-7, X edge at (ix, iy+1, iz+1)
      return ix + (iy + 1) * (nx - 1) + (iz + 1) * (nx - 1) * ny;
    case 7:  // corner 7-4, Y edge at (ix, iy, iz+1)
      return x_count + ix + iy * nx + (iz + 1) * nx * (ny - 1);

    // --- Vertical (Z) edges ---
    // There are nx*ny*(nz-1) Z edges; they follow after the Y-edge offsets.
    case 8:  // corner 0-4, Z edge at (ix, iy, iz)
      return x_count + y_count + ix + iy * nx + iz * nx * ny;
    case 9:  // corner 1-5, Z edge at (ix+1, iy, iz)
      return x_count + y_count + (ix + 1) + iy * nx + iz * nx * ny;
    case 10:  // corner 2-6, Z edge at (ix+1, iy+1, iz)
      return x_count + y_count + (ix + 1) + (iy + 1) * nx + iz * nx * ny;
    case 11:  // corner 3-7, Z edge at (ix, iy+1, iz)
      return x_count + y_count + ix + (iy + 1) * nx + iz * nx * ny;
  }
  return -1;  // invalid edgeId
}

// Compute cube index based on iso threshold
__device__ int CalcCubeIndex(const TsdfVoxel* voxels, int ix, int iy, int iz,
                             int3 vn, float iso_level) {
  // Future work: Use size_t for cube/voxel index
  // int32 index cannot handle over 1290 x 1290 x 1290 voxels

  float sdf[8];
  // offsets for 8 corners
  const int offs[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                          {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  for (int k = 0; k < 8; k++) {
    int x = ix + offs[k][0];
    int y = iy + offs[k][1];
    int z = iz + offs[k][2];
    int idx = z * vn.y * vn.x + y * vn.x + x;
    // skip if empty
    if (voxels[idx].weight <= 0.f) return -1;
    sdf[k] = voxels[idx].tsdf;
  }
  int cube_index = 0;
  for (int k = 0; k < 8; k++) {
    if (sdf[k] < iso_level) cube_index |= (1 << k);
  }
  return cube_index;
}

__device__ const int kCornerOffset[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                            {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                            {1, 1, 1}, {0, 1, 1}};

// For edgeId 0-11, the two corner ids that form the edge
__device__ const int kEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // bottom face
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // top face
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // vertical edges
};

// Cube index stored per cell: 255 marks a cell that yields nothing (a corner
// without observations, all corners inside, whose cube index is 255 too, or
// all corners outside, whose cube index 0 is folded into 255 as well). The
// other cells are the surface cells.
constexpr unsigned char kInertCell = 255;

// One for a surface cell, zero for an inert one: the flag that compacts the
// surface cells and counts them.
struct IsSurfaceCell {
  __host__ __device__ __forceinline__ int operator()(unsigned char cube) const {
    return cube != kInertCell ? 1 : 0;
  }
};

__device__ inline int CellIndex(int ix, int iy, int iz, int3 vn) {
  return ix + iy * (vn.x - 1) + iz * (vn.x - 1) * (vn.y - 1);
}

__device__ inline int3 CellCoords(int cell, int3 vn) {
  const int nx = vn.x - 1;
  const int slice = nx * (vn.y - 1);
  const int iz = cell / slice;
  const int rest = cell - iz * slice;
  return make_int3(rest % nx, rest / nx, iz);
}

// Grid edge `e` of cell (ix, iy, iz): its axis (0 = x, 1 = y, 2 = z) and the
// grid point at its lower end.
__device__ inline int EdgeAxisAndLowerEnd(int ix, int iy, int iz, int e,
                                          int3* lower) {
  const int* a = kCornerOffset[kEdgeCorners[e][0]];
  const int* b = kCornerOffset[kEdgeCorners[e][1]];
  const int axis = (a[0] != b[0]) ? 0 : ((a[1] != b[1]) ? 1 : 2);
  *lower = make_int3(ix + min(a[0], b[0]), iy + min(a[1], b[1]),
                     iz + min(a[2], b[2]));
  return axis;
}

// A crossed edge is shared by up to four cells; the one with the smallest
// linear index among the cells that yield output (cube index != kInertCell)
// emits its vertex. Every such cell crosses the edge, since the crossing
// depends only on the two end points.
__device__ inline bool OwnsEdge(const unsigned char* cell_cube, int3 vn,
                                int cell, int axis, int3 lower) {
  for (int s = 0; s < 2; ++s) {
    for (int t = 0; t < 2; ++t) {
      int ox = lower.x, oy = lower.y, oz = lower.z;
      if (axis == 0) {
        oy -= s;
        oz -= t;
      } else if (axis == 1) {
        ox -= s;
        oz -= t;
      } else {
        ox -= s;
        oy -= t;
      }
      if (ox < 0 || oy < 0 || oz < 0 || ox > vn.x - 2 || oy > vn.y - 2 ||
          oz > vn.z - 2) {
        continue;
      }
      const int other = CellIndex(ox, oy, oz, vn);
      if (other < cell && cell_cube[other] != kInertCell) {
        return false;
      }
    }
  }
  return true;
}

__device__ inline int TriangleCount(int cube_index) {
  const int* tri = &d_tri_table[cube_index][0];
  int n = 0;
  while (tri[n] != -1) n += 3;
  return n / 3;
}

// Pass 1: cube index per cell.
__global__ void ClassifyCellsKernel(const TsdfVoxel* voxels, int3 vn,
                                    float iso_level, unsigned char* cell_cube) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= vn.x - 1 || iy >= vn.y - 1 || iz >= vn.z - 1) return;
  const int cube_index = CalcCubeIndex(voxels, ix, iy, iz, vn, iso_level);
  cell_cube[CellIndex(ix, iy, iz, vn)] =
      (cube_index <= 0 || cube_index >= 255)
          ? kInertCell
          : static_cast<unsigned char>(cube_index);
}

// Pass 3: vertices (owned crossed edges) and triangles emitted by each
// surface cell. Pass 2 is the compaction of the surface cells on the host
// side (CompactSurfaceCells).
__global__ void CountCellOutputsKernel(const unsigned char* cell_cube, int3 vn,
                                       const int* surface_cells,
                                       int num_surface_cells, int* vertex_count,
                                       int* triangle_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_surface_cells) return;
  const int cell = surface_cells[i];
  const int3 c = CellCoords(cell, vn);
  const int ix = c.x, iy = c.y, iz = c.z;
  const unsigned char cube = cell_cube[cell];
  const int edges = d_edge_table[cube];
  int vertices = 0;
  for (int e = 0; e < 12; e++) {
    if (!(edges & (1 << e))) continue;
    int3 lower;
    const int axis = EdgeAxisAndLowerEnd(ix, iy, iz, e, &lower);
    if (OwnsEdge(cell_cube, vn, cell, axis, lower)) ++vertices;
  }
  vertex_count[i] = vertices;
  triangle_count[i] = TriangleCount(cube);
}

// Pass 4: vertices at the scanned offsets; the edge -> vertex id map is
// written for every owned edge, and every crossed edge of a surface cell has
// an owner among the surface cells.
__global__ void EmitVerticesKernel(const TsdfVoxel* voxels,
                                   const unsigned char* cell_cube, int3 vn,
                                   float3 bb_min, float3 resolution,
                                   float iso_level, const int* surface_cells,
                                   int num_surface_cells,
                                   const int* vertex_offset,
                                   int* d_edge_vertex_ids, float3* d_vertices) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_surface_cells) return;
  const int cell = surface_cells[i];
  const int3 c = CellCoords(cell, vn);
  const int ix = c.x, iy = c.y, iz = c.z;
  const unsigned char cube = cell_cube[cell];
  const int edges = d_edge_table[cube];
  int vid = vertex_offset[i];
  for (int e = 0; e < 12; e++) {
    if (!(edges & (1 << e))) continue;
    int3 lower;
    const int axis = EdgeAxisAndLowerEnd(ix, iy, iz, e, &lower);
    if (!OwnsEdge(cell_cube, vn, cell, axis, lower)) continue;

    int3 upper = lower;
    if (axis == 0) {
      upper.x += 1;
    } else if (axis == 1) {
      upper.y += 1;
    } else {
      upper.z += 1;
    }
    const int flat0 = lower.z * vn.y * vn.x + lower.y * vn.x + lower.x;
    const int flat1 = upper.z * vn.y * vn.x + upper.y * vn.x + upper.x;
    // Both ends are corners of this (yielding) cell, so they are observed.
    const float v1 = voxels[flat0].tsdf;
    const float v2 = voxels[flat1].tsdf;
    const float3 p1 = VoxelIndexToPosition(lower, bb_min, resolution);
    const float3 p2 = VoxelIndexToPosition(upper, bb_min, resolution);

    d_vertices[vid] = VertexInterp(p1, p2, v1, v2, iso_level);
    d_edge_vertex_ids[ComputeEdgeKey(ix, iy, iz, e, vn.x, vn.y, vn.z)] = vid;
    ++vid;
  }
}

// Pass 5: triangles at the scanned offsets (and their normals).
__global__ void EmitFacesKernel(const unsigned char* cell_cube, int3 vn,
                                const int* surface_cells, int num_surface_cells,
                                const int* d_edge_vertex_ids,
                                const int* triangle_offset,
                                const float3* d_vertices, int* d_faces,
                                float3* d_face_normals) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_surface_cells) return;
  const int cell = surface_cells[i];
  const int3 c = CellCoords(cell, vn);
  const int ix = c.x, iy = c.y, iz = c.z;
  const unsigned char cube = cell_cube[cell];
  const int* tri = &d_tri_table[cube][0];
  int face = triangle_offset[i];
  for (int i = 0; tri[i] != -1; i += 3, ++face) {
    const int v0 =
        d_edge_vertex_ids[ComputeEdgeKey(ix, iy, iz, tri[i], vn.x, vn.y, vn.z)];
    const int v1 = d_edge_vertex_ids[ComputeEdgeKey(ix, iy, iz, tri[i + 1],
                                                    vn.x, vn.y, vn.z)];
    const int v2 = d_edge_vertex_ids[ComputeEdgeKey(ix, iy, iz, tri[i + 2],
                                                    vn.x, vn.y, vn.z)];
    const int idx = face * 3;
    d_faces[idx + 0] = v2;
    d_faces[idx + 1] = v1;
    d_faces[idx + 2] = v0;
    if (d_face_normals != nullptr) {
      float3 face_normal = cross(d_vertices[v2] - d_vertices[v0],
                                 d_vertices[v1] - d_vertices[v0]);
      d_face_normals[face] = normalize(face_normal);
    }
  }
}

__global__ void ComputeVertexNormalsKernel(
    const float3* __restrict__ face_normals,  // [numFaces]
    const int* __restrict__ faces,            // [numFaces*3]
    float3* vertex_normals,                   // [numVertices]
    int num_faces) {
  int fid = blockIdx.x * blockDim.x + threadIdx.x;
  if (fid >= num_faces) {
    return;
  }

  // 3 vertices of the face
  int3 v =
      make_int3(faces[fid * 3 + 0], faces[fid * 3 + 1], faces[fid * 3 + 2]);
  float3 fn = face_normals[fid];

  // Add to each vertex normal
  atomicAdd(&vertex_normals[v.x].x, fn.x);
  atomicAdd(&vertex_normals[v.x].y, fn.y);
  atomicAdd(&vertex_normals[v.x].z, fn.z);

  atomicAdd(&vertex_normals[v.y].x, fn.x);
  atomicAdd(&vertex_normals[v.y].y, fn.y);
  atomicAdd(&vertex_normals[v.y].z, fn.z);

  atomicAdd(&vertex_normals[v.z].x, fn.x);
  atomicAdd(&vertex_normals[v.z].y, fn.y);
  atomicAdd(&vertex_normals[v.z].z, fn.z);
}

__global__ void NormalizeVertexNormalsKernel(
    float3* vertex_normals,  // [numVertices]
    int num_vertices) {
  int vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= num_vertices) {
    return;
  }

  float3 n = vertex_normals[vid];
  float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
  if (len > 1e-6f) {
    n.x /= len;
    n.y /= len;
    n.z /= len;
  }
  vertex_normals[vid] = n;
}

__global__ void SmoothFaceNormalsKernel(
    const int* __restrict__ faces,              // [numFaces*3]
    const float3* __restrict__ vertex_normals,  // [numVertices]
    float3* smooth_face_normals,                // [numFaces]  output
    int num_faces) {
  int fid = blockIdx.x * blockDim.x + threadIdx.x;
  if (fid >= num_faces) {
    return;
  }

  int3 v =
      make_int3(faces[fid * 3 + 0], faces[fid * 3 + 1], faces[fid * 3 + 2]);

  // Average of 3 vertex normals
  float3 n0 = vertex_normals[v.x];
  float3 n1 = vertex_normals[v.y];
  float3 n2 = vertex_normals[v.z];
  float3 avg =
      make_float3(n0.x + n1.x + n2.x, n0.y + n1.y + n2.y, n0.z + n1.z + n2.z);

  // Normalize
  float len = sqrtf(avg.x * avg.x + avg.y * avg.y + avg.z * avg.z);
  if (len > 1e-6f) {
    avg.x /= len;
    avg.y /= len;
    avg.z /= len;
  }

  smooth_face_normals[fid] = avg;
}

}  // namespace

namespace grtmesh {

namespace {

void Require(bool ok) {
  if (!ok) {
    GRTMESH_CHECK_CUDA_ERRORS(cudaErrorMemoryAllocation);
  }
}

}  // namespace

class TsdfVolume::Impl {
 public:
  bool Init(const Eigen::Vector3f& min_corner,
            const Eigen::Vector3f& max_corner,
            const Eigen::Vector<uint32_t, 3>& voxel_count) {
    const Eigen::Vector3f resolution =
        (max_corner - min_corner).cwiseQuotient(voxel_count.cast<float>());
    min_corner_ = make_float3(min_corner[0], min_corner[1], min_corner[2]);
    resolution_ = make_float3(resolution[0], resolution[1], resolution[2]);
    voxel_num_ = make_int3(static_cast<int>(voxel_count[0]),
                           static_cast<int>(voxel_count[1]),
                           static_cast<int>(voxel_count[2]));

    const size_t total_voxel_num =
        static_cast<size_t>(voxel_num_.x) * voxel_num_.y * voxel_num_.z;
    Require(d_voxels_.ResizeDiscard(total_voxel_num));
    cudaMemset(d_voxels_.data(), 0, sizeof(TsdfVoxel) * total_voxel_num);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

    size_t total_cells =
        (voxel_num_.x - 1) * (voxel_num_.y - 1) * (voxel_num_.z - 1);
    size_t max_tris = total_cells * 5 * 3;  // Theoretical max;
    const int max_dim =
        std::max(std::max(voxel_num_.x, voxel_num_.y), voxel_num_.z);

    // Practical max
    vertex_capacity_ = static_cast<int>(max_tris / max_dim);

    Require(d_vertices_.ResizeDiscard(vertex_capacity_));
    cudaMemset(d_vertices_.data(), 0, sizeof(float3) * vertex_capacity_);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    Require(d_vertex_normals_.ResizeDiscard(vertex_capacity_));

    // Grid size
    int nx = voxel_num_.x;
    int ny = voxel_num_.y;
    int nz = voxel_num_.z;

    // Edge counts per direction
    size_t x_count = (nx - 1) * ny * nz;  // X-direction edges
    size_t y_count = nx * (ny - 1) * nz;  // Y-direction edges
    size_t z_count = nx * ny * (nz - 1);  // Z-direction edges

    // Total edge count
    num_edges_ = x_count + y_count + z_count;

    Require(d_edge_vertex_ids_.ResizeDiscard(num_edges_));
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

    // Marching cubes: one byte per cell, the rest per surface cell (grown on
    // demand from the same practical maximum as the vertices).
    num_cells_ = total_cells;
    Require(d_cell_cube_.ResizeDiscard(num_cells_));
    Require(d_surface_counts_.ResizeDiscard(1));
    EnsureSurfaceCapacity(vertex_capacity_);
    // Face buffer and counter
    size_t max_f = total_cells * 15;

    // Practical max
    index_capacity_ = static_cast<int>(max_f / max_dim);

    Require(d_faces_.ResizeDiscard(index_capacity_));
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    Require(d_face_normals_.ResizeDiscard(index_capacity_ / 3));
    Require(d_face_smooth_normals_.ResizeDiscard(index_capacity_ / 3));
    Require(h_vertices_.ResizeDiscard(vertex_capacity_));
    Require(h_faces_.ResizeDiscard(index_capacity_));
    Require(h_face_normals_.ResizeDiscard(index_capacity_ / 3));
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

    return true;
  }

  void FuseDepthMulti(const float* depth, int width, int height,
                      const std::vector<DepthView>& views,
                      const TsdfFusionParams& params, bool sync,
                      double* depth_upload_ms, bool depth_on_device) {
    const int num_images = static_cast<int>(views.size());
    fusion_params_ = params;

    // World-to-camera of each depth camera: the transpose of the
    // camera-to-world rotation and -R^T t, in row-major order.
    std::vector<DeviceCamera> cameras(num_images);
    for (int n = 0; n < num_images; n++) {
      const Eigen::Matrix3f r =
          views[n].camera_to_world.rotation().cast<float>();
      const Eigen::Vector3f t =
          views[n].camera_to_world.translation().cast<float>();
      float camera_rotation[9];
      for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 3; j++) {
          // Eigen is col-major on memory as default
          // Use accesor to assume row-major
          camera_rotation[k * 3 + j] = r(k, j);
        }
      }
      DeviceCamera& cam = cameras[n];
      cam.fx = views[n].intrinsics.fx;
      cam.fy = views[n].intrinsics.fy;
      cam.cx = views[n].intrinsics.cx;
      cam.cy = views[n].intrinsics.cy;
      float* r_inv = cam.r;
      // Inverse rotation matrix
      r_inv[0] = camera_rotation[0];
      r_inv[1] = camera_rotation[3];
      r_inv[2] = camera_rotation[6];
      r_inv[3] = camera_rotation[1];
      r_inv[4] = camera_rotation[4];
      r_inv[5] = camera_rotation[7];
      r_inv[6] = camera_rotation[2];
      r_inv[7] = camera_rotation[5];
      r_inv[8] = camera_rotation[8];

      float* t_inv = cam.t;
      // Inverse translation vector
      t_inv[0] = -(r_inv[0] * t[0] + r_inv[1] * t[1] + r_inv[2] * t[2]);
      t_inv[1] = -(r_inv[3] * t[0] + r_inv[4] * t[1] + r_inv[5] * t[2]);
      t_inv[2] = -(r_inv[6] * t[0] + r_inv[7] * t[1] + r_inv[8] * t[2]);
      cam.dir[0] = cam.dir[1] = cam.dir[2] = 0.f;
    }
    Require(d_cameras_.ResizeDiscard(num_images));
    cudaMemcpy(d_cameras_.data(), cameras.data(),
               sizeof(DeviceCamera) * num_images, cudaMemcpyHostToDevice);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

    const size_t depth_elements =
        static_cast<size_t>(width) * height * num_images;
    const auto upload_start = std::chrono::steady_clock::now();
    const float* d_depth = depth;
    if (!depth_on_device) {
      Require(d_depth_.ResizeDiscard(depth_elements));
      cudaMemcpy(d_depth_.data(), depth, sizeof(float) * depth_elements,
                 cudaMemcpyHostToDevice);
      d_depth = d_depth_.data();
    }
    if (depth_upload_ms) {
      *depth_upload_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - upload_start)
                             .count();
    }
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    dim3 block(8, 8, 2);
    dim3 grid((voxel_num_.x + block.x - 1) / block.x,
              (voxel_num_.y + block.y - 1) / block.y,
              (voxel_num_.z + block.z - 1) / block.z);
    FuseDepthKernel<<<grid, block>>>(
        d_depth, width, height, num_images, d_cameras_.data(), d_voxels_.data(),
        resolution_, min_corner_, voxel_num_, params.truncation_distance);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    if (sync) {
      GRTMESH_CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    }
  }

  // Marching cubes in four deterministic passes: classify the cells, count
  // each cell's vertices and triangles, exclusive-scan the counts, then emit
  // into the exact offsets. No atomics, no retry on overflow: the buffers are
  // grown to the counted sizes before emitting.
  void ExtractMesh(bool with_face_normals) {
    const int num_cells = static_cast<int>(num_cells_);
    dim3 block(8, 8, 8);
    dim3 grid((voxel_num_.x + block.x - 1) / block.x,
              (voxel_num_.y + block.y - 1) / block.y,
              (voxel_num_.z + block.z - 1) / block.z);

    constexpr float kIsoLevel = 0.f;
    ClassifyCellsKernel<<<grid, block>>>(d_voxels_.data(), voxel_num_,
                                         kIsoLevel, d_cell_cube_.data());
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

    num_vertices_ = 0;
    num_faces_ = 0;
    const int num_surface = CompactSurfaceCells(num_cells);
    if (num_surface == 0) {
      return;
    }
    constexpr int kThreads = 256;
    const int blocks = (num_surface + kThreads - 1) / kThreads;
    CountCellOutputsKernel<<<blocks, kThreads>>>(
        d_cell_cube_.data(), voxel_num_, d_surface_cells_.data(), num_surface,
        d_surface_vertex_offset_.data(), d_surface_triangle_offset_.data());
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

    num_vertices_ = ExclusiveScanInPlace(d_surface_vertex_offset_, num_surface);
    num_faces_ = ExclusiveScanInPlace(d_surface_triangle_offset_, num_surface);
    if (num_vertices_ < 3 || num_faces_ < 1) {
      num_vertices_ = 0;
      num_faces_ = 0;
      return;
    }
    EnsureVertexCapacity(num_vertices_);
    EnsureIndexCapacity(num_faces_ * 3);

    EmitVerticesKernel<<<blocks, kThreads>>>(
        d_voxels_.data(), d_cell_cube_.data(), voxel_num_, min_corner_,
        resolution_, kIsoLevel, d_surface_cells_.data(), num_surface,
        d_surface_vertex_offset_.data(), d_edge_vertex_ids_.data(),
        d_vertices_.data());
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    EmitFacesKernel<<<blocks, kThreads>>>(
        d_cell_cube_.data(), voxel_num_, d_surface_cells_.data(), num_surface,
        d_edge_vertex_ids_.data(), d_surface_triangle_offset_.data(),
        d_vertices_.data(), d_faces_.data(),
        with_face_normals ? d_face_normals_.data() : nullptr);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    GRTMESH_CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  }

  // Pass 2: the surface cells (cube index != kInertCell) compacted in cell
  // order, which keeps the vertex and face order of a per-cell scan. Returns
  // their number.
  int CompactSurfaceCells(int num_cells) {
    const thrust::counting_iterator<int> cells(0);
    const auto flags = thrust::make_transform_iterator(
        static_cast<const unsigned char*>(d_cell_cube_.data()),
        IsSurfaceCell());
    int* d_count = d_surface_counts_.data();
    size_t reduce_bytes = 0;
    GRTMESH_CHECK_CUDA_ERRORS(cub::DeviceReduce::Sum(
        nullptr, reduce_bytes, flags, d_count, num_cells));
    size_t select_bytes = 0;
    GRTMESH_CHECK_CUDA_ERRORS(cub::DeviceSelect::Flagged(
        nullptr, select_bytes, cells, flags, static_cast<int*>(nullptr),
        d_count, num_cells));
    Require(d_scan_temp_.Reserve(std::max(reduce_bytes, select_bytes)));

    size_t temp_bytes = d_scan_temp_.capacity();
    GRTMESH_CHECK_CUDA_ERRORS(cub::DeviceReduce::Sum(
        d_scan_temp_.data(), temp_bytes, flags, d_count, num_cells));
    int num_surface = 0;
    GRTMESH_CHECK_CUDA_ERRORS(
        cudaMemcpy(&num_surface, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_surface == 0) {
      return 0;
    }
    EnsureSurfaceCapacity(num_surface);
    temp_bytes = d_scan_temp_.capacity();
    GRTMESH_CHECK_CUDA_ERRORS(cub::DeviceSelect::Flagged(
        d_scan_temp_.data(), temp_bytes, cells, flags, d_surface_cells_.data(),
        d_count, num_cells));
    return num_surface;
  }

  // Replaces the per-cell counts by their exclusive prefix sum and returns the
  // total.
  int ExclusiveScanInPlace(DeviceBuffer<int>& counts, int num_items) {
    int last_count = 0;
    cudaMemcpy(&last_count, counts.data() + (num_items - 1), sizeof(int),
               cudaMemcpyDeviceToHost);
    size_t temp_bytes = 0;
    GRTMESH_CHECK_CUDA_ERRORS(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes, counts.data(), counts.data(), num_items));
    Require(d_scan_temp_.Reserve(temp_bytes));
    GRTMESH_CHECK_CUDA_ERRORS(
        cub::DeviceScan::ExclusiveSum(d_scan_temp_.data(), temp_bytes,
                                      counts.data(), counts.data(), num_items));
    int last_offset = 0;
    cudaMemcpy(&last_offset, counts.data() + (num_items - 1), sizeof(int),
               cudaMemcpyDeviceToHost);
    return last_offset + last_count;
  }

  void RemoveSmallConnectedComponents(int max_iter, int min_faces,
                                      int early_exit_check_interval) {
    if (num_faces_ < 1 || num_vertices_ < 3) {
      return;
    }

    grtmesh::MeshDevice mesh_in, mesh_out;
    mesh_in.d_vertices = d_vertices_.data();
    mesh_in.num_vertices = num_vertices_;
    mesh_in.d_faces = d_faces_.data();
    mesh_in.d_face_normals = d_face_normals_.data();
    mesh_in.num_faces = num_faces_;

    cudaStream_t stream = 0;

    grtmesh::RemoveSmallConnectedComponents(mesh_in, max_iter, min_faces,
                                            early_exit_check_interval, mesh_out,
                                            components_scratch_, stream);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

    num_vertices_ = mesh_out.num_vertices;
    num_faces_ = mesh_out.num_faces;

    cudaMemcpyAsync(d_vertices_.data(), mesh_out.d_vertices,
                    sizeof(float3) * num_vertices_, cudaMemcpyDeviceToDevice,
                    stream);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    cudaMemcpyAsync(d_faces_.data(), mesh_out.d_faces,
                    sizeof(int) * num_faces_ * 3, cudaMemcpyDeviceToDevice,
                    stream);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    if (mesh_in.d_face_normals) {
      cudaMemcpyAsync(d_face_normals_.data(), mesh_out.d_face_normals,
                      sizeof(float3) * num_faces_, cudaMemcpyDeviceToDevice,
                      stream);
    }
    GRTMESH_CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  }

  void ComputeVertexNormals() {
    // Zero clear for summation

    if (num_faces_ < 1 || num_vertices_ < 3) {
      return;
    }
    cudaMemset(d_vertex_normals_.data(), 0, sizeof(float3) * num_vertices_);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    const int kThreads = 256;

    int blocks_f = std::max(1, (num_faces_ + kThreads - 1) / kThreads);
    int blocks_v = std::max(1, (num_vertices_ + kThreads - 1) / kThreads);

    // Both kernels run on the default stream, so the dependent normalize
    // pass needs no host-side synchronization in between.
    ComputeVertexNormalsKernel<<<blocks_f, kThreads>>>(
        d_face_normals_.data(), d_faces_.data(), d_vertex_normals_.data(),
        num_faces_);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    NormalizeVertexNormalsKernel<<<blocks_v, kThreads>>>(
        d_vertex_normals_.data(), num_vertices_);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    GRTMESH_CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  }

  void SmoothFaceNormalsWithVertexNormals() {
    // Return if no faces
    if (num_faces_ < 1 || num_vertices_ < 3) {
      return;
    }

    const int kThreads = 256;
    int blocks_f = std::max(1, (num_faces_ + kThreads - 1) / kThreads);

    SmoothFaceNormalsKernel<<<blocks_f, kThreads>>>(
        d_faces_.data(), d_vertex_normals_.data(),
        d_face_smooth_normals_.data(), num_faces_);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
    GRTMESH_CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  }

  void GetVerticesCpu(std::vector<Eigen::Vector3f>& vertices) {
    vertices.resize(num_vertices_);

    if (num_vertices_ < 1) {
      return;
    }

    // Copy data back
    cudaMemcpy(h_vertices_.data(), d_vertices_.data(),
               sizeof(float3) * num_vertices_, cudaMemcpyDeviceToHost);
    // Eigen::Vector3f is 3 packed floats (sizeof == 12 == sizeof(float3)), so
    // the pinned float3 buffer is bit-compatible with the destination. A single
    // memcpy replaces the per-element repack loop (verified bit-identical on
    // the same buffer; ~2 ms/frame faster on ~0.6 M-face meshes).
    std::memcpy(vertices.data(), h_vertices_.data(),
                sizeof(float3) * num_vertices_);
  }

  void GetFacesCpu(std::vector<Eigen::Vector3i>& faces) {
    faces.resize(num_faces_);
    if (num_faces_ < 1) {
      return;
    }

    cudaMemcpy(h_faces_.data(), d_faces_.data(), sizeof(int) * num_faces_ * 3,
               cudaMemcpyDeviceToHost);
    // Eigen::Vector3i is 3 packed ints (sizeof == 12): memcpy the pinned int[3]
    // buffer instead of the per-element repack loop.
    std::memcpy(faces.data(), h_faces_.data(), sizeof(int) * num_faces_ * 3);
  }

  void GetFaceNormalsCpu(std::vector<Eigen::Vector3f>& face_normals,
                         bool smoothing = false) {
    face_normals.resize(num_faces_);

    if (num_faces_ < 1) {
      return;
    }

    const float3* d_face_normals_source =
        smoothing ? d_face_smooth_normals_.data() : d_face_normals_.data();
    cudaMemcpy(h_face_normals_.data(), d_face_normals_source,
               sizeof(float3) * num_faces_, cudaMemcpyDeviceToHost);
    // bit-compatible float3 -> Eigen::Vector3f: single memcpy, no repack loop.
    std::memcpy(face_normals.data(), h_face_normals_.data(),
                sizeof(float3) * num_faces_);
  }

  const float3* GetVerticesGpu() const { return d_vertices_.data(); }

  const int* GetFacesGpu() const { return d_faces_.data(); }

  const float3* GetFaceNormalsGpu() const { return d_face_normals_.data(); }

  const float3* GetSmoothFaceNormalsGpu() const {
    return d_face_smooth_normals_.data();
  }

  void Clear() {
    const size_t total_voxel_num =
        static_cast<size_t>(voxel_num_.x) * voxel_num_.y * voxel_num_.z;
    cudaMemset(d_voxels_.data(), 0, sizeof(TsdfVoxel) * total_voxel_num);
  }

 private:
  void EnsureVertexCapacity(int num_vertices) {
    if (num_vertices <= vertex_capacity_) {
      return;
    }

    vertex_capacity_ = num_vertices;

    Require(d_vertices_.ResizeDiscard(vertex_capacity_));
    Require(d_vertex_normals_.ResizeDiscard(vertex_capacity_));
    Require(h_vertices_.ResizeDiscard(vertex_capacity_));
  }

  void EnsureSurfaceCapacity(int num_surface_cells) {
    if (num_surface_cells <= surface_capacity_) {
      return;
    }

    surface_capacity_ = num_surface_cells;

    Require(d_surface_cells_.ResizeDiscard(surface_capacity_));
    Require(d_surface_vertex_offset_.ResizeDiscard(surface_capacity_));
    Require(d_surface_triangle_offset_.ResizeDiscard(surface_capacity_));
  }

  void EnsureIndexCapacity(int num_indices) {
    if (num_indices <= index_capacity_) {
      return;
    }

    index_capacity_ = num_indices;

    Require(d_faces_.ResizeDiscard(index_capacity_));
    Require(d_face_normals_.ResizeDiscard(index_capacity_ / 3));
    Require(d_face_smooth_normals_.ResizeDiscard(index_capacity_ / 3));

    Require(h_faces_.ResizeDiscard(index_capacity_));
    Require(h_face_normals_.ResizeDiscard(index_capacity_ / 3));
  }

  DeviceBuffer<TsdfVoxel> d_voxels_;
  DeviceBuffer<float3> d_vertices_;
  DeviceBuffer<int> d_faces_;
  DeviceBuffer<int> d_edge_vertex_ids_;
  DeviceBuffer<float3> d_face_normals_;
  DeviceBuffer<float3> d_vertex_normals_;
  DeviceBuffer<float3> d_face_smooth_normals_;
  DeviceBuffer<float> d_depth_;
  DeviceBuffer<DeviceCamera> d_cameras_;
  // Marching cubes: cube index per cell; the surface cells compacted in cell
  // order with their vertex / triangle counts turned into exclusive offsets.
  DeviceBuffer<unsigned char> d_cell_cube_;
  DeviceBuffer<int> d_surface_counts_;  // the surface-cell count (device)
  DeviceBuffer<int> d_surface_cells_;
  DeviceBuffer<int> d_surface_vertex_offset_;
  DeviceBuffer<int> d_surface_triangle_offset_;
  DeviceBuffer<uint8_t> d_scan_temp_;  // CUB temp: reduce / select / scan

  PinnedHostBuffer<float3> h_vertices_;
  PinnedHostBuffer<int> h_faces_;
  PinnedHostBuffer<float3> h_face_normals_;

  size_t num_edges_ = 0;
  size_t num_cells_ = 0;
  float3 min_corner_{0.f, 0.f, 0.f};
  float3 resolution_{0.f, 0.f, 0.f};
  int3 voxel_num_{0, 0, 0};
  TsdfFusionParams fusion_params_;

  int vertex_capacity_ = 0;   // d_vertices_ / d_vertex_normals_ / h_vertices_
  int index_capacity_ = 0;    // d_faces_ / h_faces_ (3 per face)
  int surface_capacity_ = 0;  // d_surface_cells_ / d_surface_*_offset_

  int num_faces_ = 0;
  int num_vertices_ = 0;

  grtmesh::ConnectedComponentsScratch components_scratch_;
};

TsdfVolume::TsdfVolume() : impl_(std::make_unique<Impl>()) {}

TsdfVolume::~TsdfVolume() = default;

bool TsdfVolume::Init(const Eigen::Vector3f& min_corner,
                      const Eigen::Vector3f& max_corner,
                      const Eigen::Vector<uint32_t, 3>& voxel_count) {
  return impl_->Init(min_corner, max_corner, voxel_count);
}

void TsdfVolume::FuseDepthMulti(const float* depth, int width, int height,
                                const std::vector<DepthView>& views,
                                const TsdfFusionParams& params, bool sync,
                                double* depth_upload_ms, bool depth_on_device) {
  impl_->FuseDepthMulti(depth, width, height, views, params, sync,
                        depth_upload_ms, depth_on_device);
}

void TsdfVolume::ExtractMesh(bool with_face_normals) {
  impl_->ExtractMesh(with_face_normals);
}

void TsdfVolume::RemoveSmallConnectedComponents(int max_iter, int min_faces,
                                                int early_exit_check_interval) {
  impl_->RemoveSmallConnectedComponents(max_iter, min_faces,
                                        early_exit_check_interval);
}

void TsdfVolume::ComputeVertexNormals() { impl_->ComputeVertexNormals(); }

void TsdfVolume::SmoothFaceNormalsWithVertexNormals() {
  impl_->SmoothFaceNormalsWithVertexNormals();
}

void TsdfVolume::GetVerticesCpu(std::vector<Eigen::Vector3f>& vertices) {
  impl_->GetVerticesCpu(vertices);
}

void TsdfVolume::GetFacesCpu(std::vector<Eigen::Vector3i>& faces) {
  impl_->GetFacesCpu(faces);
}

void TsdfVolume::GetFaceNormalsCpu(std::vector<Eigen::Vector3f>& face_normals,
                                   bool smoothing) {
  impl_->GetFaceNormalsCpu(face_normals, smoothing);
}

const float3* TsdfVolume::GetVerticesGpu() const {
  return impl_->GetVerticesGpu();
}

const int* TsdfVolume::GetFacesGpu() const { return impl_->GetFacesGpu(); }

const float3* TsdfVolume::GetFaceNormalsGpu() const {
  return impl_->GetFaceNormalsGpu();
}

const float3* TsdfVolume::GetSmoothFaceNormalsGpu() const {
  return impl_->GetSmoothFaceNormalsGpu();
}

void TsdfVolume::Clear() { impl_->Clear(); }

}  // namespace grtmesh
