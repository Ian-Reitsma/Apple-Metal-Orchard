// CPU-only runtime implementation used when Metal APIs are unavailable.
#include "runtime/CpuContext.h"
#include "runtime/MetalContext.h"

#include <cstring>
#include <string>
#include <unordered_map>

namespace orchard::runtime {

using ContextFactory = void *(*)();

namespace {

// Simple registry mapping device names to context factories.
std::unordered_map<std::string, ContextFactory> &registry() {
  static std::unordered_map<std::string, ContextFactory> instance;
  return instance;
}

} // namespace

void register_device(const std::string &name, ContextFactory factory) {
  registry()[name] = factory;
}

void *get_device(const std::string &name) {
  auto it = registry().find(name);
  if (it == registry().end()) {
    return nullptr;
  }
  return it->second();
}

void register_runtime_devices() {
  register_device("metal", []() -> void * { return &metal_context(); });
  register_device("cpu", []() -> void * { return &cpu_context(); });
}

void metal_copy_buffers(void *dstBuf, void *srcBuf, std::size_t bytes) {
  std::memcpy(dstBuf, srcBuf, bytes);
}

void metal_copy_cpu_to_metal(void *dstBuf, const void *src, std::size_t bytes) {
  std::memcpy(dstBuf, src, bytes);
}

void metal_copy_metal_to_cpu(void *dst, void *srcBuf, std::size_t bytes) {
  std::memcpy(dst, srcBuf, bytes);
}

} // namespace orchard::runtime

int runtime_stub() { return 0; }
