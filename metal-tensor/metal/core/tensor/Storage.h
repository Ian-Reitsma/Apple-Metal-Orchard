#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <string>

#include <uuid/uuid.h>

#include "DType.h"
#include "runtime/Allocator.h"

namespace orchard::core::tensor {

struct Storage {
  void *data{nullptr};
  std::size_t nbytes{0};
  Device device{Device::cpu};
  std::atomic<std::size_t> refcount{1};
  runtime::Allocator *allocator{nullptr};
  std::string label;

  static Storage *create(std::size_t bytes, Device dev) {
    static runtime::CpuAllocator cpu_alloc;
    static runtime::MetalAllocator metal_alloc;

    runtime::Allocator *alloc = &cpu_alloc;
    if (dev == Device::mps) {
      alloc = &metal_alloc;
    }

    uuid_t id;
    uuid_generate(id);
    char uuid_str[37];
    uuid_unparse(id, uuid_str);

    void *ptr = alloc->allocate(bytes, uuid_str);
    if (!ptr)
      return nullptr;

    Storage *st = new Storage;
    st->data = ptr;
    st->nbytes = bytes;
    st->device = dev;
    st->allocator = alloc;
    st->label = uuid_str;
    return st;
  }

  void retain() { refcount.fetch_add(1, std::memory_order_relaxed); }
  void release() {
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (allocator)
        allocator->deallocate(data);
      delete this;
    }
  }
};

} // namespace orchard::core::tensor
