// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_SRC_GRTMESH_MESH_TOPOLOGY_CUH_
#define GRTMESH_SRC_GRTMESH_MESH_TOPOLOGY_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "common/device_buffer.h"

namespace grtmesh {

// Non-owning view of a triangle mesh in device memory.
struct MeshDevice {
  float3* d_vertices{nullptr};
  int* d_faces{nullptr};  // length = 3 * num_faces
  float3* d_face_normals{nullptr};
  int num_faces{0};
  int num_vertices{0};
};

// Device scratch of RemoveSmallConnectedComponents, grown on demand and reused
// across frames. Buffers are sized for the face / vertex counts passed to
// Reserve().
class ConnectedComponentsScratch {
 public:
  bool Reserve(int num_faces, int num_vertices);

  // Edge list (3 per face) and its sorted copy.
  DeviceBuffer<uint64_t> edge_key, edge_key_sorted;
  DeviceBuffer<int> edge_face, edge_face_sorted;
  DeviceBuffer<uint8_t> sort_temp;
  // Face adjacency, labels (ping-pong) and the per-label histogram.
  DeviceBuffer<int> face_adjacency, label, label_next, component_size,
      face_cc_size;
  // Compaction.
  DeviceBuffer<int> face_keep, face_scan, vertex_used, vertex_scan,
      vertex_new_id;
  DeviceBuffer<int> changed;
  DeviceBuffer<int> faces_out;
  DeviceBuffer<float3> face_normals_out, vertices_out;
};

// Drops every connected component (faces connected through shared edges) with
// fewer than min_faces faces. Labels are propagated for at most max_iterations
// rounds, checking for convergence every early_exit_check_interval rounds
// (<= 0 means 8). `out` points into `scratch` and stays valid until the next
// call.
void RemoveSmallConnectedComponents(const MeshDevice& in, int max_iterations,
                                    int min_faces,
                                    int early_exit_check_interval,
                                    MeshDevice& out,
                                    ConnectedComponentsScratch& scratch,
                                    cudaStream_t stream = 0);

// Fixed-size face adjacency: d_face_adjacency[3 * f + k] is the k-th
// edge-neighbor of face f, or -1. Non-manifold edges keep only the first two
// faces that share them. d_edge_key / d_edge_face are device workspace of
// 3 * num_faces entries each.
void BuildFaceAdjacency(const int* d_faces, int num_faces, uint64_t* d_edge_key,
                        int* d_edge_face, int* d_face_adjacency);

}  // namespace grtmesh

#endif  // GRTMESH_SRC_GRTMESH_MESH_TOPOLOGY_CUH_
