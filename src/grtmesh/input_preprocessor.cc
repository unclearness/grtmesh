// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "grtmesh/input_preprocessor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/stopwatch.h"
#include "grtmesh/input_preprocessor_kernels.cuh"

namespace grtmesh {
namespace {

namespace kernels = grtmesh::preprocess_kernels;

bool Check(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    std::cerr << "InputPreprocessor: " << what << ": "
              << cudaGetErrorString(error) << std::endl;
    return false;
  }
  return true;
}

// Device allocation freed with cudaFree.
template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  bool Allocate(size_t count) {
    if (ptr_ != nullptr) return true;
    return Check(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count),
                 "cudaMalloc");
  }
  bool Upload(const void* host, size_t bytes) {
    return Check(cudaMemcpy(ptr_, host, bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy");
  }
  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
};

// Pinned host allocation owned through a shared_ptr (grtmesh::Image storage).
template <typename T>
std::shared_ptr<T[]> AllocatePinned(size_t count) {
  T* ptr = nullptr;
  if (!Check(cudaHostAlloc(reinterpret_cast<void**>(&ptr), sizeof(T) * count,
                           cudaHostAllocDefault),
             "cudaHostAlloc")) {
    return nullptr;
  }
  return std::shared_ptr<T[]>(ptr, [](T* p) { cudaFreeHost(p); });
}

bool IsPinned(const void* ptr) {
  cudaPointerAttributes attributes;
  if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeHost;
}

// Gives `image` pinned storage of width x height unless it already has it.
template <typename ImageT>
bool EnsurePinned(ImageT& image, uint32_t width, uint32_t height) {
  if (!image.empty() && image.width == width && image.height == height &&
      IsPinned(image.data())) {
    return true;
  }
  using T = std::remove_pointer_t<decltype(image.data())>;
  std::shared_ptr<T[]> storage = AllocatePinned<T>(static_cast<size_t>(width) *
                                                   height * ImageT::kChannels);
  if (!storage) return false;
  image = ImageT::Wrap(std::move(storage), width, height);
  return true;
}

// Size only, no pixels (the pixels are on the device).
template <typename ImageT>
void SetSize(ImageT& image, uint32_t width, uint32_t height) {
  image = ImageT();
  image.width = width;
  image.height = height;
}

// Fixed-point maps of one calibration on the device (empty without lens
// distortion).
struct DeviceMaps {
  DeviceArray<short2> map1;
  DeviceArray<uint16_t> map2;
  bool valid = false;

  bool Build(const CameraCalibration& calibration) {
    if (!HasDistortion(calibration)) return true;
    std::vector<int16_t> xy;
    std::vector<uint16_t> frac;
    BuildUndistortMaps(calibration, xy, frac);
    const size_t n = frac.size();
    valid = map1.Allocate(n) && map2.Allocate(n) &&
            map1.Upload(xy.data(), sizeof(short2) * n) &&
            map2.Upload(frac.data(), sizeof(uint16_t) * n);
    return valid;
  }
  kernels::RemapMaps Get() const {
    return valid ? kernels::RemapMaps{map1.get(), map2.get()}
                 : kernels::RemapMaps{};
  }
};

bool IsRegistered(const SensorFrame& camera) {
  return camera.depth_calibration == camera.color_calibration &&
         camera.depth_to_color.matrix() == Eigen::Matrix4f::Identity();
}

}  // namespace

struct InputPreprocessor::DeviceFrames::Buffers {
  DeviceArray<uint8_t> color;
  DeviceArray<float> depth;
  DeviceArray<float> depth_in_color;  // unused when the depth is registered
  bool registered = false;
};

InputPreprocessor::DeviceFrames::DeviceFrames() = default;
InputPreprocessor::DeviceFrames::~DeviceFrames() = default;

DeviceRgbdBatch InputPreprocessor::DeviceFrames::batch() const {
  DeviceRgbdBatch batch;
  if (buffers_) {
    batch.color = buffers_->color.get();
    batch.depth = buffers_->depth.get();
    batch.depth_in_color = buffers_->registered
                               ? buffers_->depth.get()
                               : buffers_->depth_in_color.get();
  }
  return batch;
}

class InputPreprocessor::Impl {
 public:
  struct Camera {
    CameraCalibration color_calibration, depth_calibration;
    kernels::DepthToColorParams params{};
    DeviceMaps color_maps, depth_maps;
    DeviceArray<int> zbuffer;
  };

  ~Impl() {
    for (cudaEvent_t& e : events_) {
      if (e != nullptr) cudaEventDestroy(e);
    }
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  bool Init(const std::vector<SensorFrame>& cameras) {
    cameras_.clear();
    if (cameras.empty()) return false;
    const SensorFrame& first = cameras[0];
    cw_ = static_cast<int>(first.color_calibration.width);
    ch_ = static_cast<int>(first.color_calibration.height);
    dw_ = static_cast<int>(first.depth_calibration.width);
    dh_ = static_cast<int>(first.depth_calibration.height);
    registered_ = IsRegistered(first);
    const int n = static_cast<int>(cameras.size());
    if (stream_ == nullptr &&
        !Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreate")) {
      return false;
    }
    for (cudaEvent_t& e : events_) {
      if (e == nullptr && !Check(cudaEventCreate(&e), "cudaEventCreate")) {
        return false;
      }
    }

    // Same z-buffer size and resize offsets as the CPU reference +
    // cv::resize INTER_NEAREST (x_src = floor(x * src / dst), clamped).
    const float r = kDepthInColorScale;
    zh_ = static_cast<int>(ch_ * r);
    zw_ = static_cast<int>(cw_ * r);
    std::vector<int> xo(cw_), yo(ch_);
    const double ifx = 1. / (static_cast<double>(cw_) / zw_);
    const double ify = 1. / (static_cast<double>(ch_) / zh_);
    for (int x = 0; x < cw_; ++x) {
      xo[x] = std::min(static_cast<int>(std::floor(x * ifx)), zw_ - 1);
    }
    for (int y = 0; y < ch_; ++y) {
      yo[y] = std::min(static_cast<int>(std::floor(y * ify)), zh_ - 1);
    }
    if (!registered_ &&
        !(x_offsets_.Allocate(cw_) && y_offsets_.Allocate(ch_) &&
          x_offsets_.Upload(xo.data(), sizeof(int) * cw_) &&
          y_offsets_.Upload(yo.data(), sizeof(int) * ch_))) {
      return false;
    }

    for (const SensorFrame& source : cameras) {
      auto camera = std::make_unique<Camera>();
      Camera& c = *camera;
      c.color_calibration = source.color_calibration;
      c.depth_calibration = source.depth_calibration;
      if (static_cast<int>(c.color_calibration.width) != cw_ ||
          static_cast<int>(c.color_calibration.height) != ch_ ||
          static_cast<int>(c.depth_calibration.width) != dw_ ||
          static_cast<int>(c.depth_calibration.height) != dh_ ||
          IsRegistered(source) != registered_) {
        std::cerr << "InputPreprocessor: all cameras must share the image "
                     "sizes and the depth registration"
                  << std::endl;
        return false;
      }
      if (!c.color_maps.Build(c.color_calibration) ||
          !c.depth_maps.Build(c.depth_calibration)) {
        return false;
      }
      if (!registered_) {
        if (!c.zbuffer.Allocate(static_cast<size_t>(zw_) * zh_)) return false;
        const CameraIntrinsics& di = c.depth_calibration.intrinsics;
        const CameraIntrinsics& ci = c.color_calibration.intrinsics;
        c.params.depth_fx = di.fx;
        c.params.depth_fy = di.fy;
        c.params.depth_cx = di.cx;
        c.params.depth_cy = di.cy;
        c.params.color_fx = ci.fx;
        c.params.color_fy = ci.fy;
        c.params.color_cx = ci.cx;
        c.params.color_cy = ci.cy;
        c.params.scale = r;
        const Eigen::Matrix4f m = source.depth_to_color.matrix();
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) c.params.rotation[3 * i + j] = m(i, j);
          c.params.translation[i] = m(i, 3);
        }
      }
      cameras_.push_back(std::move(camera));
    }

    color_staging_ = AllocatePinned<uint8_t>(ColorBytes() * n);
    depth_staging_ = AllocatePinned<float>(DepthPixels() * n);
    in_color_ = std::make_unique<DeviceArray<uint8_t>>();
    in_depth_ = std::make_unique<DeviceArray<float>>();
    return color_staging_ && depth_staging_ &&
           in_color_->Allocate(ColorBytes() * n) &&
           in_depth_->Allocate(DepthPixels() * n);
  }

  // Upload once, then all kernels; leaves the work enqueued on stream_ and
  // events_[0..2] recorded around the upload and the kernels.
  bool Enqueue(const std::vector<SensorFrame>& sensor,
               std::vector<RgbdFramePtr>& frames, DeviceFrames& device,
               const Options& options) {
    const int n = static_cast<int>(cameras_.size());
    if (static_cast<int>(sensor.size()) != n) {
      std::cerr << "InputPreprocessor: expected " << n << " cameras, got "
                << sensor.size() << std::endl;
      return false;
    }
    for (const SensorFrame& s : sensor) {
      if (s.color == nullptr || s.depth == nullptr) {
        std::cerr << "InputPreprocessor: missing sensor image (or its size "
                     "differs from the calibration)"
                  << std::endl;
        return false;
      }
    }
    if (frames.size() != sensor.size()) {
      frames.resize(sensor.size());
      for (auto& frame : frames) frame = std::make_shared<RgbdFrame>();
    }
    if (!device.buffers_) {
      device.buffers_ = std::make_unique<DeviceFrames::Buffers>();
      device.buffers_->registered = registered_;
      if (!device.buffers_->color.Allocate(ColorBytes() * n) ||
          !device.buffers_->depth.Allocate(DepthPixels() * n) ||
          (!registered_ &&
           !device.buffers_->depth_in_color.Allocate(ColorPixels() * n))) {
        return false;
      }
    }
    DeviceFrames::Buffers& out = *device.buffers_;

    // Sensor images -> pinned staging (host copy), then one upload of each
    // batch.
    Stopwatch staging;
    auto stage = [&](int k) {
      std::memcpy(color_staging_.get() + ColorBytes() * k, sensor[k].color,
                  ColorBytes());
      std::memcpy(depth_staging_.get() + DepthPixels() * k, sensor[k].depth,
                  sizeof(float) * DepthPixels());
    };
    if (options.staging_threads > 0) {
#pragma omp parallel for num_threads(options.staging_threads)
      for (int k = 0; k < n; ++k) stage(k);
    } else {
#pragma omp parallel for
      for (int k = 0; k < n; ++k) stage(k);
    }
    timing_.staging_ms = staging.ElapsedMs();

    cudaEventRecord(events_[0], stream_);
    cudaMemcpyAsync(in_color_->get(), color_staging_.get(), ColorBytes() * n,
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(in_depth_->get(), depth_staging_.get(),
                    sizeof(float) * DepthPixels() * n, cudaMemcpyHostToDevice,
                    stream_);
    cudaEventRecord(events_[1], stream_);
    for (int k = 0; k < n; ++k) {
      const Camera& c = *cameras_[k];
      float* depth_k = out.depth.get() + DepthPixels() * k;
      kernels::RemapColor(in_color_->get() + ColorBytes() * k,
                          out.color.get() + ColorBytes() * k, cw_, ch_,
                          c.color_maps.Get(), stream_);
      kernels::RemapDepth(in_depth_->get() + DepthPixels() * k, depth_k, dw_,
                          dh_, c.depth_maps.Get(), options.depth_far_threshold,
                          stream_);
      if (!registered_) {
        kernels::ProjectDepthToColor(
            depth_k, dw_, dh_, c.params, c.zbuffer.get(), zw_, zh_,
            x_offsets_.get(), y_offsets_.get(),
            out.depth_in_color.get() + ColorPixels() * k, cw_, ch_, stream_);
      }
    }
    cudaEventRecord(events_[2], stream_);

    for (int k = 0; k < n; ++k) {
      const SensorFrame& in = sensor[k];
      RgbdFrame& frame = *frames[k];
      frame.color_intrinsics = in.color_calibration.intrinsics;
      frame.color_to_world = in.color_to_world;
      frame.depth_intrinsics = in.depth_calibration.intrinsics;
      frame.depth_to_world = in.depth_to_world;
    }
    return Check(cudaGetLastError(), "kernel launch");
  }

  bool RunDevice(const std::vector<SensorFrame>& sensor,
                 std::vector<RgbdFramePtr>& frames, DeviceFrames& device,
                 const Options& options) {
    Stopwatch total;
    if (!Enqueue(sensor, frames, device, options) ||
        !Check(cudaEventSynchronize(events_[2]), "cudaEventSynchronize")) {
      return false;
    }
    for (const RgbdFramePtr& frame : frames) {
      SetSize(frame->color, cw_, ch_);
      SetSize(frame->depth, dw_, dh_);
      if (registered_) {
        SetSize(frame->depth_in_color, dw_, dh_);
      } else {
        SetSize(frame->depth_in_color, cw_, ch_);
      }
    }
    ReadEvents(false);
    timing_.total_ms = total.ElapsedMs();
    return true;
  }

  bool RunHost(const std::vector<SensorFrame>& sensor,
               std::vector<RgbdFramePtr>& frames, const Options& options) {
    Stopwatch total;
    if (!Enqueue(sensor, frames, host_slot_, options)) return false;
    const DeviceRgbdBatch batch = host_slot_.batch();
    const size_t dic_pixels = registered_ ? DepthPixels() : ColorPixels();
    for (size_t k = 0; k < frames.size(); ++k) {
      RgbdFrame& frame = *frames[k];
      if (!EnsurePinned(frame.color, cw_, ch_) ||
          !EnsurePinned(frame.depth, dw_, dh_) ||
          !(registered_ ? EnsurePinned(frame.depth_in_color, dw_, dh_)
                        : EnsurePinned(frame.depth_in_color, cw_, ch_))) {
        return false;
      }
      cudaMemcpyAsync(frame.color.data(), batch.color + ColorBytes() * k,
                      ColorBytes(), cudaMemcpyDeviceToHost, stream_);
      cudaMemcpyAsync(frame.depth.data(), batch.depth + DepthPixels() * k,
                      sizeof(float) * DepthPixels(), cudaMemcpyDeviceToHost,
                      stream_);
      cudaMemcpyAsync(
          frame.depth_in_color.data(), batch.depth_in_color + dic_pixels * k,
          sizeof(float) * dic_pixels, cudaMemcpyDeviceToHost, stream_);
    }
    cudaEventRecord(events_[3], stream_);
    if (!Check(cudaEventSynchronize(events_[3]), "cudaEventSynchronize")) {
      return false;
    }
    ReadEvents(true);
    timing_.total_ms = total.ElapsedMs();
    return true;
  }

  const Timing& timing() const { return timing_; }

 private:
  size_t ColorPixels() const { return static_cast<size_t>(cw_) * ch_; }
  size_t ColorBytes() const { return 3 * ColorPixels(); }
  size_t DepthPixels() const { return static_cast<size_t>(dw_) * dh_; }

  void ReadEvents(bool with_download) {
    float upload = 0.f, kernel = 0.f, download = 0.f;
    cudaEventElapsedTime(&upload, events_[0], events_[1]);
    cudaEventElapsedTime(&kernel, events_[1], events_[2]);
    if (with_download) cudaEventElapsedTime(&download, events_[2], events_[3]);
    timing_.upload_ms = upload;
    timing_.kernel_ms = kernel;
    timing_.download_ms = download;
  }

  int cw_ = 0, ch_ = 0, dw_ = 0, dh_ = 0, zw_ = 0, zh_ = 0;
  // Depth registered to the color camera: the depth is the depth in color.
  bool registered_ = false;
  std::vector<std::unique_ptr<Camera>> cameras_;
  DeviceArray<int> x_offsets_, y_offsets_;
  std::shared_ptr<uint8_t[]> color_staging_;
  std::shared_ptr<float[]> depth_staging_;
  std::unique_ptr<DeviceArray<uint8_t>> in_color_;
  std::unique_ptr<DeviceArray<float>> in_depth_;
  DeviceFrames host_slot_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t events_[4] = {};
  Timing timing_;
};

InputPreprocessor::InputPreprocessor() : impl_(std::make_unique<Impl>()) {}

InputPreprocessor::~InputPreprocessor() = default;

bool InputPreprocessor::Init(const std::vector<SensorFrame>& cameras) {
  return impl_->Init(cameras);
}

bool InputPreprocessor::Run(const std::vector<SensorFrame>& sensor,
                            std::vector<RgbdFramePtr>& frames,
                            DeviceFrames& device) {
  return impl_->RunDevice(sensor, frames, device, options);
}

bool InputPreprocessor::Run(const std::vector<SensorFrame>& sensor,
                            std::vector<RgbdFramePtr>& frames) {
  return impl_->RunHost(sensor, frames, options);
}

const InputPreprocessor::Timing& InputPreprocessor::last_timing() const {
  return impl_->timing();
}

}  // namespace grtmesh
