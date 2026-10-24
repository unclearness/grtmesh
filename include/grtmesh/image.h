// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_INCLUDE_GRTMESH_IMAGE_H_
#define GRTMESH_INCLUDE_GRTMESH_IMAGE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

namespace grtmesh {

// Dense row-major image with C interleaved channels of T. Copies of an Image
// share the pixel storage; Clone() makes an independent copy. The storage can
// come from anywhere (e.g. pinned host memory) through Wrap().
template <typename T, int C>
struct Image {
  static constexpr int kChannels = C;

  uint32_t width = 0;
  uint32_t height = 0;
  std::shared_ptr<T[]> pixels;  // C * width * height elements

  // Zero-filled image.
  static Image Create(uint32_t w, uint32_t h) {
    Image img;
    img.width = w;
    img.height = h;
    img.pixels = std::shared_ptr<T[]>(new T[static_cast<size_t>(w) * h * C]());
    return img;
  }

  // Image over existing storage holding at least C * w * h elements.
  static Image Wrap(std::shared_ptr<T[]> storage, uint32_t w, uint32_t h) {
    Image img;
    img.width = w;
    img.height = h;
    img.pixels = std::move(storage);
    return img;
  }

  bool empty() const { return !pixels; }
  size_t size() const { return static_cast<size_t>(width) * height * C; }
  size_t size_bytes() const { return size() * sizeof(T); }
  T* data() { return pixels.get(); }
  const T* data() const { return pixels.get(); }
  T& at(uint32_t x, uint32_t y, int c) {
    return pixels[(static_cast<size_t>(y) * width + x) * C + c];
  }
  const T& at(uint32_t x, uint32_t y, int c) const {
    return pixels[(static_cast<size_t>(y) * width + x) * C + c];
  }

  Image Clone() const {
    Image img = Create(width, height);
    if (pixels) {
      std::memcpy(img.pixels.get(), pixels.get(), size_bytes());
    }
    return img;
  }
};

using ColorImage = Image<uint8_t, 3>;
using DepthImage = Image<float, 1>;

}  // namespace grtmesh

#endif  // GRTMESH_INCLUDE_GRTMESH_IMAGE_H_
