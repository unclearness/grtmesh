// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_SRC_COMMON_STOPWATCH_H_
#define GRTMESH_SRC_COMMON_STOPWATCH_H_

#include <chrono>

namespace grtmesh {

// Monotonic wall-clock stopwatch in milliseconds.
class Stopwatch {
 public:
  Stopwatch() : start_(std::chrono::steady_clock::now()) {}
  void Restart() { start_ = std::chrono::steady_clock::now(); }
  double ElapsedMs() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace grtmesh

#endif  // GRTMESH_SRC_COMMON_STOPWATCH_H_
