// MetalContext.h
// -------------------------------------------------------------
// Thin wrapper around the system default Metal device. Each thread
// receives its own context instance which keeps a pool of command
// queues for reuse.

#pragma once

#include <vector>

#if defined(__APPLE__) && defined(__OBJC__)
#include <Metal/Metal.h>
using MTLDeviceRef = id<MTLDevice>;
using MTLCommandQueueRef = id<MTLCommandQueue>;
using MTLCommandBufferRef = id<MTLCommandBuffer>;
using MTLBlitCommandEncoderRef = id<MTLBlitCommandEncoder>;
#else
using MTLDeviceRef = void *;
using MTLCommandQueueRef = void *;
using MTLCommandBufferRef = void *;
using MTLBlitCommandEncoderRef = void *;
#endif

namespace orchard::runtime {

class MetalContext {
public:
  MetalContext();

#if defined(__APPLE__) && defined(__OBJC__)
  /// Returns the underlying MTLDevice.
  MTLDeviceRef device() const { return has_device_ ? device_ : nil; }
  bool has_device() const { return has_device_; }
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
#if defined(__APPLE__) && defined(__OBJC__)
  MTLDeviceRef device_ = nil;
  bool has_device_ = false;
  std::vector<MTLCommandQueueRef> queue_pool_;
#else
  [[maybe_unused]] MTLDeviceRef device_ = nullptr;
  [[maybe_unused]] bool has_device_ = false;
  [[maybe_unused]] std::vector<MTLCommandQueueRef> queue_pool_;
#endif
};

/// Obtain the Metal context associated with the calling thread.
MetalContext &metal_context();

} // namespace orchard::runtime

// Inline implementations
inline orchard::runtime::MetalContext::MetalContext() {
#if defined(__APPLE__) && defined(__OBJC__)
  device_ = MTLCreateSystemDefaultDevice();
  has_device_ = device_ != nil;
#endif
}

inline MTLCommandQueueRef
orchard::runtime::MetalContext::acquire_command_queue() {
#if defined(__APPLE__) && defined(__OBJC__)
  if (!has_device_)
    return nil;
  if (!queue_pool_.empty()) {
    id<MTLCommandQueue> queue = queue_pool_.back();
    queue_pool_.pop_back();
    return queue;
  }
  return has_device_ ? [device_ newCommandQueue] : nil;
#else
  return nullptr;
#endif
}

inline void
orchard::runtime::MetalContext::return_command_queue(MTLCommandQueueRef queue) {
#if defined(__APPLE__) && defined(__OBJC__)
  if (has_device_ && queue)
    queue_pool_.push_back(queue);
#else
  (void)queue;
#endif
}

inline MTLBlitCommandEncoderRef
orchard::runtime::MetalContext::acquire_blit_encoder(
    MTLCommandQueueRef &queue, MTLCommandBufferRef &cmdBuf) {
#if defined(__APPLE__) && defined(__OBJC__)
  if (!has_device_) {
    queue = nil;
    cmdBuf = nil;
    return nil;
  }
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
