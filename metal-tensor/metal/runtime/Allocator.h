#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>

#ifdef __APPLE__
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#endif
#endif

namespace orchard::runtime {

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual void *allocate(std::size_t bytes, const char *label) = 0;
  virtual void deallocate(void *ptr) = 0;
};

class CpuAllocator : public Allocator {
public:
  void *allocate(std::size_t bytes, const char * /*label*/) override {
    void *p = nullptr;
    posix_memalign(&p, 64, bytes);
    return p;
  }
  void deallocate(void *ptr) override { free(ptr); }
};

class MetalAllocator : public Allocator {
public:
  MetalAllocator();
  void *allocate(std::size_t bytes, const char *label) override;
  void deallocate(void *ptr) override;

private:
#ifdef __OBJC__
  id<MTLDevice> device_{nil};
#endif
};

inline MetalAllocator::MetalAllocator() {
#ifdef __OBJC__
  device_ = MTLCreateSystemDefaultDevice();
#endif
}

inline void *MetalAllocator::allocate(std::size_t bytes, const char *label) {
#ifdef __OBJC__
  id<MTLBuffer> buffer = nil;
  if (bytes > (16 << 20)) {
    const void *keys[] = {(const void *)kIOSurfaceWidth,
                          (const void *)kIOSurfaceHeight,
                          (const void *)kIOSurfaceBytesPerElement};
    int width = bytes;
    int height = 1;
    int bpe = 1;
    const void *values[] = {
        CFNumberCreate(nullptr, kCFNumberSInt32Type, &width),
        CFNumberCreate(nullptr, kCFNumberSInt32Type, &height),
        CFNumberCreate(nullptr, kCFNumberSInt32Type, &bpe),
    };
    CFDictionaryRef dict = CFDictionaryCreate(nullptr, keys, values, 3,
                                              &kCFTypeDictionaryKeyCallBacks,
                                              &kCFTypeDictionaryValueCallBacks);
    IOSurfaceRef surface = IOSurfaceCreate(dict);
    for (int i = 0; i < 3; ++i)
      CFRelease(values[i]);
    CFRelease(dict);
    buffer = [device_ newBufferWithIOSurface:surface
                                     options:MTLResourceStorageModeShared
                                      offset:0
                                      length:bytes];
    [buffer setPurgeableState:MTLPurgeableStateKeepCurrent];
    CFRelease(surface);
  } else {
    buffer = [device_ newBufferWithLength:bytes
                                  options:MTLResourceStorageModeShared];
  }
  buffer.label = [[NSString alloc] initWithUTF8String:label];
  return (__bridge_retained void *)buffer;
#else
  (void)bytes;
  (void)label;
  return nullptr;
#endif
}

inline void MetalAllocator::deallocate(void *ptr) {
#ifdef __OBJC__
  id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)ptr;
  buffer = nil;
#else
  (void)ptr;
#endif
}

} // namespace orchard::runtime

