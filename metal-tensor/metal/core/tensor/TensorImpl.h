#pragma once

#include <array>
#include <cstddef>
#ifdef __APPLE__
#include <os/lock.h>
#else
#include <mutex>
#endif

#include "Storage.h"

namespace orchard::core::tensor {

struct TensorImpl {
  Storage *storage{nullptr};
  std::array<std::int64_t, 8> shape{};
  std::array<std::int64_t, 8> strides{};
  DType dtype{DType::f32};
  Device device{Device::cpu};
  std::int64_t offset{0};
#ifdef __APPLE__
  os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
#else
  std::mutex lock;
#endif
  void *grad_fn{nullptr};
  void *grad_ctx{nullptr};

  TensorImpl() = default;
  TensorImpl(const TensorImpl &other)
      : storage(other.storage), shape(other.shape), strides(other.strides),
        dtype(other.dtype), device(other.device), offset(other.offset),
        grad_fn(other.grad_fn), grad_ctx(other.grad_ctx) {}
};

} // namespace orchard::core::tensor
