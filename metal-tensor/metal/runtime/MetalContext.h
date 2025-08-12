// MetalContext.h
// -------------------------------------------------------------
// Thin wrapper around the system default Metal device. Each thread
// receives its own context instance which keeps a pool of command
// queues for reuse.

#pragma once

#include <vector>

#ifdef __APPLE__
#include <Metal/Metal.h>
using MTLCommandQueueRef = id<MTLCommandQueue>;
using MTLCommandBufferRef = id<MTLCommandBuffer>;
using MTLBlitCommandEncoderRef = id<MTLBlitCommandEncoder>;
#else
using MTLCommandQueueRef = void *;
using MTLCommandBufferRef = void *;
using MTLBlitCommandEncoderRef = void *;
#endif

namespace orchard::runtime {

class MetalContext {
public:
  MetalContext();

#ifdef __APPLE__
  /// Returns the underlying MTLDevice.
  id<MTLDevice> device() const { return device_; }
#endif

  /// Acquire a command queue for the current thread.
  MTLCommandQueueRef acquire_command_queue();

  /// Return a command queue to the thread‑local pool.
  void return_command_queue(MTLCommandQueueRef queue);

  /// Acquire a blit command encoder along with its backing queue and
  /// command buffer. The caller is responsible for ending encoding,
  /// committing the command buffer and returning the queue.
  MTLBlitCommandEncoderRef acquire_blit_encoder(MTLCommandQueueRef &queue,
                                                MTLCommandBufferRef &cmdBuf);

private:
#ifdef __APPLE__
  id<MTLDevice> device_ = nil;
  std::vector<MTLCommandQueueRef> queue_pool_;
#endif
};

/// Obtain the Metal context associated with the calling thread.
MetalContext &metal_context();

} // namespace orchard::runtime

// Inline implementations
inline orchard::runtime::MetalContext::MetalContext() {
#ifdef __APPLE__
  device_ = MTLCreateSystemDefaultDevice();
#endif
}

inline MTLCommandQueueRef
orchard::runtime::MetalContext::acquire_command_queue() {
#ifdef __APPLE__
  if (!queue_pool_.empty()) {
    id<MTLCommandQueue> queue = queue_pool_.back();
    queue_pool_.pop_back();
    return queue;
  }
  return [device_ newCommandQueue];
#else
  return nullptr;
#endif
}

inline void
orchard::runtime::MetalContext::return_command_queue(MTLCommandQueueRef queue) {
#ifdef __APPLE__
  if (queue)
    queue_pool_.push_back(queue);
#else
  (void)queue;
#endif
}

inline MTLBlitCommandEncoderRef
orchard::runtime::MetalContext::acquire_blit_encoder(
    MTLCommandQueueRef &queue, MTLCommandBufferRef &cmdBuf) {
#ifdef __APPLE__
  queue = acquire_command_queue();
  cmdBuf = [queue commandBuffer];
  return [cmdBuf blitCommandEncoder];
#else
  (void)queue;
  (void)cmdBuf;
  return nullptr;
#endif
}

inline orchard::runtime::MetalContext &orchard::runtime::metal_context() {
  thread_local MetalContext ctx;
  return ctx;
}
