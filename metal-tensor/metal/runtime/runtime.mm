#include "runtime/Runtime.h"
#include "runtime/CpuContext.h"
#include "runtime/MetalContext.h"

#include <stdexcept>
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

#ifdef __APPLE__
void metal_copy_buffers(void *dstBuf, const void *srcBuf, std::size_t bytes) {
  MetalContext &ctx = metal_context();
  id<MTLCommandQueue> queue = nil;
  id<MTLCommandBuffer> cmd = nil;
  id<MTLBlitCommandEncoder> blit = ctx.acquire_blit_encoder(queue, cmd);
  if (!blit)
    throw std::runtime_error("Metal device unavailable");
  id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstBuf;
  id<MTLBuffer> src = (__bridge id<MTLBuffer>)(const_cast<void *>(srcBuf));
  [blit copyFromBuffer:src
           sourceOffset:0
               toBuffer:dst
      destinationOffset:0
                   size:bytes];
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_copy_cpu_to_metal(void *dstBuf, const void *src, std::size_t bytes) {
  MetalContext &ctx = metal_context();
  if (!ctx.device())
    throw std::runtime_error("Metal device unavailable");
  id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstBuf;
  id<MTLBuffer> tmp =
      [ctx.device() newBufferWithBytes:src
                                length:bytes
                               options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue = nil;
  id<MTLCommandBuffer> cmd = nil;
  id<MTLBlitCommandEncoder> blit = ctx.acquire_blit_encoder(queue, cmd);
  if (!blit) {
    [tmp release];
    throw std::runtime_error("Metal device unavailable");
  }
  [blit copyFromBuffer:tmp
           sourceOffset:0
               toBuffer:dst
      destinationOffset:0
                   size:bytes];
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
  [tmp release];
}

void metal_copy_metal_to_cpu(void *dst, const void *srcBuf, std::size_t bytes) {
  MetalContext &ctx = metal_context();
  if (!ctx.device())
    throw std::runtime_error("Metal device unavailable");
  id<MTLBuffer> src = (__bridge id<MTLBuffer>)(const_cast<void *>(srcBuf));
  id<MTLBuffer> tmp =
      [ctx.device() newBufferWithBytesNoCopy:dst
                                      length:bytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
  id<MTLCommandQueue> queue = nil;
  id<MTLCommandBuffer> cmd = nil;
  id<MTLBlitCommandEncoder> blit = ctx.acquire_blit_encoder(queue, cmd);
  if (!blit) {
    [tmp release];
    throw std::runtime_error("Metal device unavailable");
  }
  [blit copyFromBuffer:src
           sourceOffset:0
               toBuffer:tmp
      destinationOffset:0
                   size:bytes];
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
  [tmp release];
}
#endif

} // namespace orchard::runtime

int runtime_stub() { return 0; }
