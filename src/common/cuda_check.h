// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Minimal fail-fast CUDA runtime error checking.
//
// Usable from both .cu and .cc translation units (host code only) and safe to
// include more than once.

#ifndef GRTMESH_SRC_COMMON_CUDA_CHECK_H_
#define GRTMESH_SRC_COMMON_CUDA_CHECK_H_

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace grtmesh::detail {

// Prints "CUDA error at <file>:<line> code=<n>(<name>) "<expr>"" to stderr and
// exits the process when `result` is not cudaSuccess.
inline void Check(cudaError_t result, char const* const func,
                  const char* const file, int const line) {
  if (result) {
    std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file,
                 line, static_cast<int>(result), cudaGetErrorName(result),
                 func);
    std::exit(EXIT_FAILURE);
  }
}

// Reports the sticky error from cudaGetLastError() (e.g. a failed kernel
// launch) and exits the process when one is pending.
inline void GetLastError(const char* error_message, const char* file,
                         const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    std::fprintf(stderr,
                 "%s(%i) : getLastCudaError() CUDA error :"
                 " %s : (%d) %s.\n",
                 file, line, error_message, static_cast<int>(err),
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace grtmesh::detail

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define GRTMESH_CHECK_CUDA_ERRORS(val) \
  ::grtmesh::detail::Check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define GRTMESH_GET_LAST_CUDA_ERROR(msg) \
  ::grtmesh::detail::GetLastError(msg, __FILE__, __LINE__)

#endif  // GRTMESH_SRC_COMMON_CUDA_CHECK_H_
