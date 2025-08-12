#include "MetalKernels.h"
#include "CpuContext.h"
#include "MetalContext.h"

#include <array>
#include <cstring>
#include <fstream>
#include <string>

#ifdef __APPLE__
#include <Foundation/Foundation.h>
#endif

namespace orchard::runtime {

#ifdef __APPLE__
void metal_add(const float *a, const float *b, float *c,
               const std::int64_t *shape, const std::int64_t *astrides,
               const std::int64_t *bstrides, std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/add.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"add_arrays"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)b offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
  id<MTLBuffer> shapeBuf =
      [ctx.device() newBufferWithBytes:shape
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> aBuf =
      [ctx.device() newBufferWithBytes:astrides
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> bBuf =
      [ctx.device() newBufferWithBytes:bstrides
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  [enc setBuffer:shapeBuf offset:0 atIndex:3];
  [enc setBuffer:aBuf offset:0 atIndex:4];
  [enc setBuffer:bBuf offset:0 atIndex:5];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  [shapeBuf release];
  [aBuf release];
  [bBuf release];
  ctx.return_command_queue(queue);
}

void metal_mul(const float *a, const float *b, float *c,
               const std::int64_t *shape, const std::int64_t *astrides,
               const std::int64_t *bstrides, std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/mul.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_arrays"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)b offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
  id<MTLBuffer> shapeBuf =
      [ctx.device() newBufferWithBytes:shape
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> aBuf =
      [ctx.device() newBufferWithBytes:astrides
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> bBuf =
      [ctx.device() newBufferWithBytes:bstrides
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  [enc setBuffer:shapeBuf offset:0 atIndex:3];
  [enc setBuffer:aBuf offset:0 atIndex:4];
  [enc setBuffer:bBuf offset:0 atIndex:5];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  [shapeBuf release];
  [aBuf release];
  [bBuf release];
  ctx.return_command_queue(queue);
}
void metal_div(const float *a, const float *b, float *c,
               const std::int64_t *shape, const std::int64_t *astrides,
               const std::int64_t *bstrides, std::size_t n, bool safe) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/div.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"div_arrays"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)b offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
  id<MTLBuffer> shapeBuf =
      [ctx.device() newBufferWithBytes:shape
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> aBuf =
      [ctx.device() newBufferWithBytes:astrides
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> bBuf =
      [ctx.device() newBufferWithBytes:bstrides
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  [enc setBuffer:shapeBuf offset:0 atIndex:3];
  [enc setBuffer:aBuf offset:0 atIndex:4];
  [enc setBuffer:bBuf offset:0 atIndex:5];
  int s = safe ? 1 : 0;
  [enc setBytes:&s length:sizeof(int) atIndex:6];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  [shapeBuf release];
  [aBuf release];
  [bBuf release];
  ctx.return_command_queue(queue);
}

void metal_div_scalar(const float *a, float scalar, float *out, std::size_t n,
                      bool safe) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/div.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"div_scalar"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBytes:&scalar length:sizeof(float) atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
  int s = safe ? 1 : 0;
  [enc setBytes:&s length:sizeof(int) atIndex:3];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_mul_backward_a(const float *g, const float *b, float *ga,
                          std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/mul_backward.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_backward_a"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)g offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)b offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)ga offset:0 atIndex:2];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_mul_backward_b(const float *g, const float *a, float *gb,
                          std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/mul_backward.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_backward_b"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)g offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)gb offset:0 atIndex:2];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}
void metal_div_backward_a(const float *g, const float *b, float *ga,
                          std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/div.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"div_backward_a"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)g offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)b offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)ga offset:0 atIndex:2];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_div_backward_b(const float *g, const float *a, const float *b,
                          float *gb, std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/div.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"div_backward_b"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)g offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)b offset:0 atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)gb offset:0 atIndex:3];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_div_backward_a(const float *g, const float *b, float *ga,
                          std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    ga[i] = g[i] / b[i];
}

void metal_div_backward_b(const float *g, const float *a, const float *b,
                          float *gb, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    gb[i] = -g[i] * a[i] / (b[i] * b[i]);
}

void metal_matmul(const float *a, const float *b, float *c, std::size_t m,
                  std::size_t n, std::size_t k) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/matmul.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_kernel"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)b offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
  uint32_t mm = static_cast<uint32_t>(m);
  uint32_t nn = static_cast<uint32_t>(n);
  uint32_t kk = static_cast<uint32_t>(k);
  [enc setBytes:&mm length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&nn length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&kk length:sizeof(uint32_t) atIndex:5];
  MTLSize grid = MTLSizeMake(m * n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_reduce_sum(const float *a, float *out, std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/reduce_sum.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"reduce_sum"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
  uint32_t nn = static_cast<uint32_t>(n);
  [enc setBytes:&nn length:sizeof(uint32_t) atIndex:2];
  MTLSize grid = MTLSizeMake(1, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_mean(const float *a, float *out, std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/mean.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"mean"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
  uint32_t nn = static_cast<uint32_t>(n);
  [enc setBytes:&nn length:sizeof(uint32_t) atIndex:2];
  MTLSize grid = MTLSizeMake(1, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

// Parameters follow (m, n, k)
void metal_matmul_backward_a(const float *g, const float *b, float *ga,
                             std::size_t m, std::size_t n, std::size_t k) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/matmul_backward.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_backward_a"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)g offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)b offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)ga offset:0 atIndex:2];
  uint32_t mm = static_cast<uint32_t>(m);
  uint32_t nn = static_cast<uint32_t>(n);
  uint32_t kk = static_cast<uint32_t>(k);
  [enc setBytes:&mm length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&nn length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&kk length:sizeof(uint32_t) atIndex:5];
  MTLSize grid = MTLSizeMake(m * k, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

// Parameters follow (m, n, k)
void metal_matmul_backward_b(const float *g, const float *a, float *gb,
                             std::size_t m, std::size_t n, std::size_t k) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/matmul_backward.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_backward_b"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)g offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)gb offset:0 atIndex:2];
  uint32_t mm = static_cast<uint32_t>(m);
  uint32_t nn = static_cast<uint32_t>(n);
  uint32_t kk = static_cast<uint32_t>(k);
  [enc setBytes:&mm length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&nn length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&kk length:sizeof(uint32_t) atIndex:5];
  MTLSize grid = MTLSizeMake(k * n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_transpose_backward(const float *g, float *out, std::size_t m,
                              std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/transpose_backward.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"transpose_backward"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)g offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
  uint32_t mm = static_cast<uint32_t>(m);
  uint32_t nn = static_cast<uint32_t>(n);
  [enc setBytes:&mm length:sizeof(uint32_t) atIndex:2];
  [enc setBytes:&nn length:sizeof(uint32_t) atIndex:3];
  MTLSize grid = MTLSizeMake(m * n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_fill(float *out, float value, std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/fill.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"fill_value"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:0];
  [enc setBytes:&value length:sizeof(float) atIndex:1];
  uint32_t nn = static_cast<uint32_t>(n);
  [enc setBytes:&nn length:sizeof(uint32_t) atIndex:2];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_reduce_sum_axis(const float *a, float *out,
                           const std::int64_t *shape,
                           const std::int64_t *strides, std::uint32_t axis_len,
                           std::uint32_t axis, std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/reduce_sum_axis.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"reduce_sum_axis"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
  id<MTLBuffer> shapeBuf =
      [ctx.device() newBufferWithBytes:shape
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> strideBuf =
      [ctx.device() newBufferWithBytes:strides
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  [enc setBuffer:shapeBuf offset:0 atIndex:2];
  [enc setBuffer:strideBuf offset:0 atIndex:3];
  [enc setBytes:&axis_len length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&axis length:sizeof(uint32_t) atIndex:5];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  [shapeBuf release];
  [strideBuf release];
  ctx.return_command_queue(queue);
}

void metal_mean_axis(const float *a, float *out, const std::int64_t *shape,
                     const std::int64_t *strides, std::uint32_t axis_len,
                     std::uint32_t axis, std::size_t n) {
  static id<MTLComputePipelineState> pipeline = nil;
  MetalContext &ctx = metal_context();
  if (!pipeline) {
    std::ifstream ifs("metal/kernels/mean_axis.metal");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    NSString *nsSrc = [[NSString alloc] initWithBytes:src.data()
                                               length:src.size()
                                             encoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id<MTLLibrary> lib = [ctx.device() newLibraryWithSource:nsSrc
                                                    options:nil
                                                      error:&err];
    [nsSrc release];
    id<MTLFunction> fn = [lib newFunctionWithName:@"mean_axis"];
    pipeline = [ctx.device() newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
  }
  id<MTLCommandQueue> queue = ctx.acquire_command_queue();
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:(__bridge id<MTLBuffer>)(const void *)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
  id<MTLBuffer> shapeBuf =
      [ctx.device() newBufferWithBytes:shape
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> strideBuf =
      [ctx.device() newBufferWithBytes:strides
                                length:sizeof(std::int64_t) * 8
                               options:MTLResourceStorageModeShared];
  [enc setBuffer:shapeBuf offset:0 atIndex:2];
  [enc setBuffer:strideBuf offset:0 atIndex:3];
  [enc setBytes:&axis_len length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&axis length:sizeof(uint32_t) atIndex:5];
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  [shapeBuf release];
  [strideBuf release];
  ctx.return_command_queue(queue);
}
#else
void metal_add(const float *a, const float *b, float *c,
               const std::int64_t *shape, const std::int64_t *astrides,
               const std::int64_t *bstrides, std::size_t n) {
  std::array<std::int64_t, 8> shp;
  std::array<std::int64_t, 8> as;
  std::array<std::int64_t, 8> bs;
  std::memcpy(shp.data(), shape, sizeof(std::int64_t) * 8);
  std::memcpy(as.data(), astrides, sizeof(std::int64_t) * 8);
  std::memcpy(bs.data(), bstrides, sizeof(std::int64_t) * 8);
  std::array<std::int64_t, 8> idx{};
  std::int64_t ao = 0;
  std::int64_t bo = 0;
  for (std::size_t i = 0; i < n; ++i) {
    c[i] = a[ao] + b[bo];
    for (int d = 7; d >= 0; --d) {
      idx[d]++;
      ao += as[d];
      bo += bs[d];
      if (idx[d] < shp[d])
        break;
      idx[d] = 0;
      ao -= as[d] * shp[d];
      bo -= bs[d] * shp[d];
    }
  }
}

void metal_div_scalar(const float *a, float scalar, float *out, std::size_t n,
                      bool safe) {
  if (safe && scalar == 0.0f) {
    for (std::size_t i = 0; i < n; ++i)
      out[i] = 0.0f;
  } else {
    for (std::size_t i = 0; i < n; ++i)
      out[i] = a[i] / scalar;
  }
}

void metal_mul(const float *a, const float *b, float *c,
               const std::int64_t *shape, const std::int64_t *astrides,
               const std::int64_t *bstrides, std::size_t n) {
  std::array<std::int64_t, 8> shp;
  std::array<std::int64_t, 8> as;
  std::array<std::int64_t, 8> bs;
  std::memcpy(shp.data(), shape, sizeof(std::int64_t) * 8);
  std::memcpy(as.data(), astrides, sizeof(std::int64_t) * 8);
  std::memcpy(bs.data(), bstrides, sizeof(std::int64_t) * 8);
  std::array<std::int64_t, 8> idx{};
  std::int64_t ao = 0;
  std::int64_t bo = 0;
  for (std::size_t i = 0; i < n; ++i) {
    c[i] = a[ao] * b[bo];
    for (int d = 7; d >= 0; --d) {
      idx[d]++;
      ao += as[d];
      bo += bs[d];
      if (idx[d] < shp[d])
        break;
      idx[d] = 0;
      ao -= as[d] * shp[d];
      bo -= bs[d] * shp[d];
    }
  }
}

void metal_div(const float *a, const float *b, float *c,
               const std::int64_t *shape, const std::int64_t *astrides,
               const std::int64_t *bstrides, std::size_t n, bool safe) {
  std::array<std::int64_t, 8> shp;
  std::array<std::int64_t, 8> as;
  std::array<std::int64_t, 8> bs;
  std::memcpy(shp.data(), shape, sizeof(std::int64_t) * 8);
  std::memcpy(as.data(), astrides, sizeof(std::int64_t) * 8);
  std::memcpy(bs.data(), bstrides, sizeof(std::int64_t) * 8);
  std::array<std::int64_t, 8> idx{};
  std::int64_t ao = 0;
  std::int64_t bo = 0;
  for (std::size_t i = 0; i < n; ++i) {
    float bv = b[bo];
    c[i] = (safe && bv == 0.0f) ? 0.0f : a[ao] / bv;
    for (int d = 7; d >= 0; --d) {
      idx[d]++;
      ao += as[d];
      bo += bs[d];
      if (idx[d] < shp[d])
        break;
      idx[d] = 0;
      ao -= as[d] * shp[d];
      bo -= bs[d] * shp[d];
    }
  }
}

void metal_mul_backward_a(const float *g, const float *b, float *ga,
                          std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    ga[i] = g[i] * b[i];
}

void metal_mul_backward_b(const float *g, const float *a, float *gb,
                          std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    gb[i] = g[i] * a[i];
}

void metal_div_backward_a(const float *g, const float *b, float *ga,
                          std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    ga[i] = g[i] / b[i];
}

void metal_div_backward_b(const float *g, const float *a, const float *b,
                          float *gb, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    gb[i] = -g[i] * a[i] / (b[i] * b[i]);
}

void metal_matmul(const float *a, const float *b, float *c, std::size_t m,
                  std::size_t n, std::size_t k) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float s = 0.0f;
      for (std::size_t p = 0; p < k; ++p)
        s += a[i * k + p] * b[p * n + j];
      c[i * n + j] = s;
    }
  }
}

void metal_reduce_sum(const float *a, float *out, std::size_t n) {
  float s = 0.0f;
  for (std::size_t i = 0; i < n; ++i)
    s += a[i];
  out[0] = s;
}

void metal_mean(const float *a, float *out, std::size_t n) {
  float s = 0.0f;
  for (std::size_t i = 0; i < n; ++i)
    s += a[i];
  out[0] = s / static_cast<float>(n);
}

void metal_reduce_sum_axis(const float *a, float *out,
                           const std::int64_t *shape,
                           const std::int64_t *strides, std::uint32_t axis_len,
                           std::uint32_t axis, std::size_t n) {
  std::array<std::int64_t, 8> shp;
  std::array<std::int64_t, 8> st;
  std::memcpy(shp.data(), shape, sizeof(std::int64_t) * 8);
  std::memcpy(st.data(), strides, sizeof(std::int64_t) * 8);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t idx = i;
    long base = 0;
    for (int d = 7; d >= 0; --d) {
      long s = shp[d];
      long coord = idx % s;
      idx /= s;
      base += coord * st[d];
    }
    float s = 0.0f;
    long pos = base;
    for (std::uint32_t j = 0; j < axis_len; ++j) {
      s += a[pos];
      pos += st[axis];
    }
    out[i] = s;
  }
}

void metal_mean_axis(const float *a, float *out, const std::int64_t *shape,
                     const std::int64_t *strides, std::uint32_t axis_len,
                     std::uint32_t axis, std::size_t n) {
  std::array<std::int64_t, 8> shp;
  std::array<std::int64_t, 8> st;
  std::memcpy(shp.data(), shape, sizeof(std::int64_t) * 8);
  std::memcpy(st.data(), strides, sizeof(std::int64_t) * 8);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t idx = i;
    long base = 0;
    for (int d = 7; d >= 0; --d) {
      long s = shp[d];
      long coord = idx % s;
      idx /= s;
      base += coord * st[d];
    }
    float s = 0.0f;
    long pos = base;
    for (std::uint32_t j = 0; j < axis_len; ++j) {
      s += a[pos];
      pos += st[axis];
    }
    out[i] = s / static_cast<float>(axis_len);
  }
}

// Parameters follow (m, n, k)
void metal_matmul_backward_a(const float *g, const float *b, float *ga,
                             std::size_t m, std::size_t n, std::size_t k) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      float s = 0.0f;
      for (std::size_t p = 0; p < n; ++p)
        s += g[i * n + p] * b[j * n + p];
      ga[i * k + j] = s;
    }
  }
}

// Parameters follow (m, n, k)
void metal_matmul_backward_b(const float *g, const float *a, float *gb,
                             std::size_t m, std::size_t n, std::size_t k) {
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float s = 0.0f;
      for (std::size_t p = 0; p < m; ++p)
        s += a[p * k + i] * g[p * n + j];
      gb[i * n + j] = s;
    }
  }
}

void metal_transpose_backward(const float *g, float *out, std::size_t m,
                              std::size_t n) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      out[i * n + j] = g[j * m + i];
    }
  }
}

void metal_fill(float *out, float value, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    out[i] = value;
}
#endif

} // namespace orchard::runtime
