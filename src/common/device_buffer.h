// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Grow-only RAII wrappers for device and pinned host memory.

#ifndef GRTMESH_SRC_COMMON_DEVICE_BUFFER_H_
#define GRTMESH_SRC_COMMON_DEVICE_BUFFER_H_

#include <cuda_runtime.h>

#include <cstddef>

namespace grtmesh {

// Device allocation that only grows. Reserve() keeps the current allocation
// when the new one fails (and reports the failure), so callers can fall back;
// ResizeDiscard() frees first, for buffers that are large and never reused
// across the growth. Contents are not preserved by either.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool Reserve(size_t count) {
    if (count <= capacity_) {
      return true;
    }
    T* replacement = nullptr;
    if (cudaMalloc(&replacement, count * sizeof(T)) != cudaSuccess) {
      cudaGetLastError();  // acknowledge the recoverable allocation failure
      return false;
    }
    Release();
    ptr_ = replacement;
    capacity_ = count;
    return true;
  }

  bool ResizeDiscard(size_t count) {
    if (count <= capacity_) {
      return true;
    }
    Release();
    if (cudaMalloc(&ptr_, count * sizeof(T)) != cudaSuccess) {
      cudaGetLastError();
      ptr_ = nullptr;
      return false;
    }
    capacity_ = count;
    return true;
  }

  void Release() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    capacity_ = 0;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t capacity() const { return capacity_; }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  T* ptr_ = nullptr;
  size_t capacity_ = 0;
};

// Page-locked host allocation with the same grow-only contract.
template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  ~PinnedHostBuffer() { Release(); }
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  bool ResizeDiscard(size_t count) {
    if (count <= capacity_) {
      return true;
    }
    Release();
    if (cudaMallocHost(&ptr_, count * sizeof(T)) != cudaSuccess) {
      cudaGetLastError();
      ptr_ = nullptr;
      return false;
    }
    capacity_ = count;
    return true;
  }

  void Release() {
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
    }
    capacity_ = 0;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t capacity() const { return capacity_; }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  T* ptr_ = nullptr;
  size_t capacity_ = 0;
};

}  // namespace grtmesh

#endif  // GRTMESH_SRC_COMMON_DEVICE_BUFFER_H_
