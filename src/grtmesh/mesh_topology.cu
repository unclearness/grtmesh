// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cub/cub.cuh>
#include <utility>

#include "common/cuda_check.h"
#include "grtmesh/mesh_topology.cuh"

namespace {

__host__ __device__ inline uint64_t PackEdgeKey(int a, int b) {
  uint32_t lo = static_cast<uint32_t>((a < b ? a : b));
  uint32_t hi = static_cast<uint32_t>((a < b ? b : a));
  return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

__global__ void FillIntKernel(int* a, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = v;
}

// ---------------------------
// (1) Build edge list (E=3F)
// ---------------------------
__global__ void BuildEdgesFromFacesKernel(const int* faces, int face_count,
                                          uint64_t* edge_key, int* edge_face) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;

  int i0 = faces[3 * f + 0];
  int i1 = faces[3 * f + 1];
  int i2 = faces[3 * f + 2];

  int base = 3 * f;
  edge_key[base + 0] = PackEdgeKey(i0, i1);
  edge_face[base + 0] = f;

  edge_key[base + 1] = PackEdgeKey(i1, i2);
  edge_face[base + 1] = f;

  edge_key[base + 2] = PackEdgeKey(i2, i0);
  edge_face[base + 2] = f;
}

// Puts v into the first free one of a face's three neighbor slots (-1 = free).
__device__ inline void TrySetNeighbor(int* slots, int v) {
  for (int k = 0; k < 3; ++k) {
    if (atomicCAS(&slots[k], -1, v) == -1) return;
  }
}

// Build adjacency from sorted edges:
// for each run(key) with len>=2, connect first two faces.
__global__ void BuildAdjacencyFromSortedEdgesKernel(const uint64_t* edge_key,
                                                    const int* edge_face,
                                                    int edge_count,
                                                    int* face_adjacency,
                                                    int face_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= edge_count) return;

  bool is_head = (i == 0) || (edge_key[i] != edge_key[i - 1]);
  if (!is_head) return;

  uint64_t key = edge_key[i];
  int j = i + 1;
  while (j < edge_count && edge_key[j] == key) ++j;

  if (j - i >= 2) {
    int f0 = edge_face[i];
    int f1 = edge_face[i + 1];
    if ((unsigned)f0 < (unsigned)face_count &&
        (unsigned)f1 < (unsigned)face_count && f0 != f1) {
      TrySetNeighbor(&face_adjacency[3 * f0], f1);
      TrySetNeighbor(&face_adjacency[3 * f1], f0);
    }
  }
}

// ---------------------------
// (2) Label propagation (early-exit flag)
// ---------------------------
__global__ void LabelPropagationKernel(const int* face_adjacency,
                                       int face_count, const int* label_in,
                                       int* label_out, int* d_changed) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;

  int old = label_in[f];
  int m = old;

  int n0 = face_adjacency[3 * f + 0];
  int n1 = face_adjacency[3 * f + 1];
  int n2 = face_adjacency[3 * f + 2];

  if (n0 >= 0) m = min(m, label_in[n0]);
  if (n1 >= 0) m = min(m, label_in[n1]);
  if (n2 >= 0) m = min(m, label_in[n2]);

  // pointer jumping (same as m = min(m, label[m]) in the OpenMP version)
  m = min(m, label_in[m]);

  label_out[f] = m;
  if (m != old) atomicOr(d_changed, 1);
}

// ---------------------------
// (3) Component size per face: histogram of the labels (a label is the index
//     of the component's smallest face, so the histogram has one bin per face)
// ---------------------------
__global__ void CountLabelsKernel(const int* label, int face_count,
                                  int* component_size) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  atomicAdd(&component_size[label[f]], 1);
}

__global__ void GatherComponentSizesKernel(const int* label,
                                           const int* component_size,
                                           int face_count, int* face_cc_size) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  face_cc_size[f] = component_size[label[f]];
}

// ---------------------------
// (4) Face keep flag, face compaction, vertex mark, vertex compaction, remap
// ---------------------------
__global__ void MarkKeptFacesKernel(const int* face_cc_size, int face_count,
                                    int min_faces, int* face_keep) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  face_keep[f] = (face_cc_size[f] >= min_faces) ? 1 : 0;
}

__global__ void CompactFacesAndNormalsKernel(
    const int* faces_in, const float3* face_normals_in, int face_count,
    const int* face_keep,
    const int* face_scan_excl,  // exclusive scan of keep
    int* faces_out,
    float3* face_normals_out)  // nullable: if nullptr, normals are skipped
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  if (!face_keep[f]) return;

  int out_f = face_scan_excl[f];

  faces_out[3 * out_f + 0] = faces_in[3 * f + 0];
  faces_out[3 * out_f + 1] = faces_in[3 * f + 1];
  faces_out[3 * out_f + 2] = faces_in[3 * f + 2];

  if (face_normals_out && face_normals_in) {
    face_normals_out[out_f] = face_normals_in[f];
  }
}

__global__ void MarkUsedVerticesKernel(const int* faces, int face_count,
                                       int vertex_count,
                                       int* v_used)  // int[V], 0/1
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;

  int i0 = faces[3 * f + 0];
  int i1 = faces[3 * f + 1];
  int i2 = faces[3 * f + 2];

  // Reject out-of-range indices (also serves as a debug check)
  if ((unsigned)i0 < (unsigned)vertex_count) atomicExch(&v_used[i0], 1);
  if ((unsigned)i1 < (unsigned)vertex_count) atomicExch(&v_used[i1], 1);
  if ((unsigned)i2 < (unsigned)vertex_count) atomicExch(&v_used[i2], 1);
}

// Build new_id from v_used -> v_scan_excl(int)
__global__ void BuildVertexRemapKernel(
    const int* v_used, const int* v_scan_excl, int vertex_count,
    int* v_new_id)  // -1 for unused, else new index
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertex_count) return;
  v_new_id[i] = v_used[i] ? v_scan_excl[i] : -1;
}

__global__ void GatherVerticesKernel(const float3* v_in, int vertex_count,
                                     const int* v_used, const int* v_scan_excl,
                                     float3* v_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertex_count) return;
  if (!v_used[i]) return;
  int ni = v_scan_excl[i];
  v_out[ni] = v_in[i];
}

__global__ void RemapFaceVerticesKernel(int* faces, int remaining_face_count,
                                        const int* v_new_id) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= remaining_face_count) return;
  faces[3 * f + 0] = v_new_id[faces[3 * f + 0]];
  faces[3 * f + 1] = v_new_id[faces[3 * f + 1]];
  faces[3 * f + 2] = v_new_id[faces[3 * f + 2]];
}

// CUB radix sort of (edge_key, edge_face) pairs into the *_out buffers.
void SortEdgesByKeyCub(const uint64_t* keys_in, uint64_t* keys_out,
                       const int* vals_in, int* vals_out, int edge_count,
                       grtmesh::DeviceBuffer<uint8_t>& temp_storage,
                       cudaStream_t stream) {
  void* temp_ptr = nullptr;
  size_t temp_bytes = 0;

  // 1) required temp size
  GRTMESH_CHECK_CUDA_ERRORS(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes, keys_in, keys_out, vals_in, vals_out, edge_count,
      /*begin_bit=*/0, /*end_bit=*/64, stream));

  // 2) allocate/reuse temp storage
  if (!temp_storage.Reserve(temp_bytes)) {
    GRTMESH_CHECK_CUDA_ERRORS(cudaErrorMemoryAllocation);
  }
  temp_ptr = temp_storage.data();

  // 3) sort
  GRTMESH_CHECK_CUDA_ERRORS(cub::DeviceRadixSort::SortPairs(
      temp_ptr, temp_bytes, keys_in, keys_out, vals_in, vals_out, edge_count,
      /*begin_bit=*/0, /*end_bit=*/64, stream));
}

template <typename T>
thrust::device_ptr<T> Ptr(grtmesh::DeviceBuffer<T>& buffer) {
  return thrust::device_pointer_cast(buffer.data());
}

}  // namespace

namespace grtmesh {

bool ConnectedComponentsScratch::Reserve(int num_faces, int num_vertices) {
  const size_t f = static_cast<size_t>(num_faces);
  const size_t v = static_cast<size_t>(num_vertices);
  const size_t e = 3 * f;
  return edge_key.Reserve(e) && edge_key_sorted.Reserve(e) &&
         edge_face.Reserve(e) && edge_face_sorted.Reserve(e) &&
         face_adjacency.Reserve(e) && label.Reserve(f) &&
         label_next.Reserve(f) && component_size.Reserve(f) &&
         face_cc_size.Reserve(f) && face_keep.Reserve(f) &&
         face_scan.Reserve(f) && vertex_used.Reserve(v) &&
         vertex_scan.Reserve(v) && vertex_new_id.Reserve(v) &&
         changed.Reserve(1) && faces_out.Reserve(e) &&
         face_normals_out.Reserve(f) && vertices_out.Reserve(v);
}

void RemoveSmallConnectedComponents(const MeshDevice& in, int max_iterations,
                                    int min_faces,
                                    int early_exit_check_interval,
                                    MeshDevice& out,
                                    ConnectedComponentsScratch& scratch,
                                    cudaStream_t stream) {
  const int face_count = in.num_faces;
  const int vertex_count = in.num_vertices;
  if (face_count < 1 || vertex_count < 3) {
    out = MeshDevice{};
    return;
  }

  if (!scratch.Reserve(face_count, vertex_count)) {
    GRTMESH_CHECK_CUDA_ERRORS(cudaErrorMemoryAllocation);
  }
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  const int threads = 256;
  const int blocks_f = (face_count + threads - 1) / threads;

  // ---- build neighbors ----
  const int edge_count = 3 * face_count;

  BuildEdgesFromFacesKernel<<<blocks_f, threads, 0, stream>>>(
      in.d_faces, face_count, scratch.edge_key.data(),
      scratch.edge_face.data());
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  SortEdgesByKeyCub(scratch.edge_key.data(), scratch.edge_key_sorted.data(),
                    scratch.edge_face.data(), scratch.edge_face_sorted.data(),
                    edge_count, scratch.sort_temp, stream);

  int* d_face_adjacency = scratch.face_adjacency.data();
  FillIntKernel<<<(3 * face_count + threads - 1) / threads, threads, 0,
                  stream>>>(d_face_adjacency, 3 * face_count, -1);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  BuildAdjacencyFromSortedEdgesKernel<<<(edge_count + threads - 1) / threads,
                                        threads, 0, stream>>>(
      scratch.edge_key_sorted.data(), scratch.edge_face_sorted.data(),
      edge_count, d_face_adjacency, face_count);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  // ---- label propagation with early exit ----
  int* d_label = scratch.label.data();
  int* d_next = scratch.label_next.data();
  thrust::sequence(thrust::cuda::par.on(stream), Ptr(scratch.label),
                   Ptr(scratch.label) + face_count);

  int* d_changed = scratch.changed.data();

  // Label propagation reaches a fixed point long before a generous K in
  // practice; checking d_changed periodically costs one 4-byte sync but
  // saves the remaining no-op iterations. 0 or negative used to mean "no
  // early exit"; results at the fixed point are identical either way.
  if (early_exit_check_interval <= 0) {
    early_exit_check_interval = 8;
  }

  for (int it = 0; it < max_iterations; ++it) {
    GRTMESH_CHECK_CUDA_ERRORS(
        cudaMemsetAsync(d_changed, 0, sizeof(int), stream));

    LabelPropagationKernel<<<blocks_f, threads, 0, stream>>>(
        d_face_adjacency, face_count, d_label, d_next, d_changed);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

    std::swap(d_label, d_next);

    if (early_exit_check_interval > 0 && it > 0 &&
        it % early_exit_check_interval == 0) {
      int h_changed = 0;
      GRTMESH_CHECK_CUDA_ERRORS(cudaMemcpyAsync(
          &h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream));
      GRTMESH_CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));  // early exit
      if (!h_changed) {
        break;
      }
    }
  }
  // d_label now holds the final labels.

  // ---- face_cc_size[f] = number of faces with the same label ----
  int* d_component_size = scratch.component_size.data();
  int* d_face_cc_size = scratch.face_cc_size.data();
  FillIntKernel<<<blocks_f, threads, 0, stream>>>(d_component_size, face_count,
                                                  0);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  CountLabelsKernel<<<blocks_f, threads, 0, stream>>>(d_label, face_count,
                                                      d_component_size);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  GatherComponentSizesKernel<<<blocks_f, threads, 0, stream>>>(
      d_label, d_component_size, face_count, d_face_cc_size);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  //// ---- face_keep and face compaction ----
  int* d_face_keep = scratch.face_keep.data();
  MarkKeptFacesKernel<<<blocks_f, threads, 0, stream>>>(
      d_face_cc_size, face_count, min_faces, d_face_keep);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  int* d_face_scan = scratch.face_scan.data();
  thrust::exclusive_scan(thrust::cuda::par.on(stream), Ptr(scratch.face_keep),
                         Ptr(scratch.face_keep) + face_count,
                         Ptr(scratch.face_scan), 0);

  int kept_faces = 0;
  int last_keep = 0, last_scan = 0;
  {
    GRTMESH_CHECK_CUDA_ERRORS(
        cudaMemcpyAsync(&last_keep, d_face_keep + (face_count - 1), sizeof(int),
                        cudaMemcpyDeviceToHost, stream));
    GRTMESH_CHECK_CUDA_ERRORS(
        cudaMemcpyAsync(&last_scan, d_face_scan + (face_count - 1), sizeof(int),
                        cudaMemcpyDeviceToHost, stream));
  }

  int* d_faces2 = scratch.faces_out.data();
  float3* d_face_normals2 = scratch.face_normals_out.data();

  CompactFacesAndNormalsKernel<<<blocks_f, threads, 0, stream>>>(
      in.d_faces, in.d_face_normals, face_count, d_face_keep, d_face_scan,
      d_faces2, d_face_normals2);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  GRTMESH_CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
  kept_faces = last_scan + last_keep;

  const int blocks_kept =
      0 < kept_faces ? (kept_faces + threads - 1) / threads : 1;

  int* d_v_used = scratch.vertex_used.data();
  thrust::fill(thrust::cuda::par.on(stream), Ptr(scratch.vertex_used),
               Ptr(scratch.vertex_used) + vertex_count, 0);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  if (0 < kept_faces) {
    MarkUsedVerticesKernel<<<blocks_kept, threads, 0, stream>>>(
        d_faces2, kept_faces, vertex_count, d_v_used);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  }

  int* d_v_scan = scratch.vertex_scan.data();
  thrust::exclusive_scan(thrust::cuda::par.on(stream), Ptr(scratch.vertex_used),
                         Ptr(scratch.vertex_used) + vertex_count,
                         Ptr(scratch.vertex_scan), 0);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  int kept_vertices = 0;
  int last_used = 0, last_vscan = 0;
  {
    GRTMESH_CHECK_CUDA_ERRORS(
        cudaMemcpyAsync(&last_used, d_v_used + (vertex_count - 1), sizeof(int),
                        cudaMemcpyDeviceToHost, stream));
    GRTMESH_CHECK_CUDA_ERRORS(
        cudaMemcpyAsync(&last_vscan, d_v_scan + (vertex_count - 1), sizeof(int),
                        cudaMemcpyDeviceToHost, stream));
  }

  float3* d_vertices2 = scratch.vertices_out.data();

  GatherVerticesKernel<<<(vertex_count + threads - 1) / threads, threads, 0,
                         stream>>>(in.d_vertices, vertex_count, d_v_used,
                                   d_v_scan, d_vertices2);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  // build v_new_id and remap faces
  int* d_v_new_id = scratch.vertex_new_id.data();
  BuildVertexRemapKernel<<<(vertex_count + threads - 1) / threads, threads, 0,
                           stream>>>(d_v_used, d_v_scan, vertex_count,
                                     d_v_new_id);
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  if (0 < kept_faces) {
    RemapFaceVerticesKernel<<<blocks_kept, threads, 0, stream>>>(
        d_faces2, kept_faces, d_v_new_id);
  }
  GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());

  GRTMESH_CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
  kept_vertices = last_vscan + last_used;

  // ---- output ----
  out.d_vertices = d_vertices2;
  out.d_faces = d_faces2;
  out.d_face_normals = d_face_normals2;
  out.num_vertices = kept_vertices;
  out.num_faces = kept_faces;
}

void BuildFaceAdjacency(const int* d_faces, int num_faces, uint64_t* d_edge_key,
                        int* d_edge_face, int* d_face_adjacency) {
  const int face_count = num_faces;
  const int edge_count = 3 * face_count;
  const int threads = 256;

  // 1) Edge list
  {
    int blocks = (face_count + threads - 1) / threads;
    BuildEdgesFromFacesKernel<<<blocks, threads>>>(d_faces, face_count,
                                                   d_edge_key, d_edge_face);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  }

  // 2) Init neighbors to -1
  {
    int blocks = (3 * face_count + threads - 1) / threads;
    FillIntKernel<<<blocks, threads>>>(d_face_adjacency, 3 * face_count, -1);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  }

  // 3) Sort by edge_key
  {
    auto keys = thrust::device_pointer_cast(d_edge_key);
    auto faces = thrust::device_pointer_cast(d_edge_face);
    thrust::sort_by_key(keys, keys + edge_count, faces);
  }

  // 4) Build adjacency from sorted edges
  {
    int blocks = (edge_count + threads - 1) / threads;
    BuildAdjacencyFromSortedEdgesKernel<<<blocks, threads>>>(
        d_edge_key, d_edge_face, edge_count, d_face_adjacency, face_count);
    GRTMESH_CHECK_CUDA_ERRORS(cudaGetLastError());
  }

  GRTMESH_CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}

}  // namespace grtmesh
