#pragma once

#include <array>
#include <cstddef>
#include <os/lock.h>

#include "Storage.h"

namespace orchard::core::tensor {

struct TensorImpl {
  Storage *storage{nullptr};
  std::array<std::int64_t, 8> shape{};
  std::array<std::int64_t, 8> strides{};
  DType dtype{DType::f32};
  Device device{Device::cpu};
  std::int64_t offset{0};
  os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
  void *grad_fn{nullptr};
  void *grad_ctx{nullptr};
};

} // namespace orchard::core::tensor
