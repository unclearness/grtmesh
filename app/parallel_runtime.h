// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_APP_PARALLEL_RUNTIME_H_
#define GRTMESH_APP_PARALLEL_RUNTIME_H_

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grtmesh::app {
inline void SetHostThreadCount(int count) {
#ifdef _OPENMP
  omp_set_num_threads(count);
#else
  (void)count;  // Host loops execute serially when OpenMP is disabled.
#endif
}
}  // namespace grtmesh::app

#endif  // GRTMESH_APP_PARALLEL_RUNTIME_H_
