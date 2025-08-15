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

#endif

} // namespace orchard::runtime
