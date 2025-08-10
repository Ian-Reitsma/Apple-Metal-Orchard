#include "MetalKernels.h"
#include "CpuContext.h"
#include "MetalContext.h"

#include <fstream>
#include <string>

#ifdef __APPLE__
#include <Foundation/Foundation.h>
#endif

namespace orchard::runtime {

#ifdef __APPLE__
void metal_add(const float *a, const float *b, float *c, std::size_t n) {
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
  MTLSize grid = MTLSizeMake(n, 1, 1);
  MTLSize thread = MTLSizeMake(1, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:thread];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ctx.return_command_queue(queue);
}

void metal_mul(const float *a, const float *b, float *c, std::size_t n) {
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
#else
void metal_add(const float *a, const float *b, float *c, std::size_t n) {
  cpu_context().add(a, b, c, n);
}

void metal_mul(const float *a, const float *b, float *c, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    c[i] = a[i] * b[i];
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
#endif

} // namespace orchard::runtime
