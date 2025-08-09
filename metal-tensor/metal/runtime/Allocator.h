#pragma once

#include <cstddef>
#include <cstdlib>

namespace orchard::runtime {

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual void* allocate(std::size_t bytes) = 0;
  virtual void deallocate(void* ptr) = 0;
};

class CpuAllocator : public Allocator {
public:
  void* allocate(std::size_t bytes) override {
    void* p = nullptr;
    posix_memalign(&p, 64, bytes);
    return p;
  }
  void deallocate(void* ptr) override { free(ptr); }
};

} // namespace orchard::runtime
