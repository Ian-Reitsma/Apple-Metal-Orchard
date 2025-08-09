#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>

#include "DType.h"

namespace orchard::core::tensor {

struct Storage {
  void* data{nullptr};
  std::size_t nbytes{0};
  Device device{Device::cpu};
  std::atomic<std::size_t> refcount{1};

  static Storage* create(std::size_t bytes, Device dev) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, bytes) != 0) {
      return nullptr;
    }
    return new Storage{ptr, bytes, dev};
  }

  void retain() { refcount.fetch_add(1, std::memory_order_relaxed); }
  void release() {
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      free(data);
      delete this;
    }
  }
};

} // namespace orchard::core::tensor
