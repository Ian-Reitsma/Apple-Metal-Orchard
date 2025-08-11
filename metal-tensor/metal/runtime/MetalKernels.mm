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

void metal_fill(float *out, float value, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    out[i] = value;
}
#endif

} // namespace orchard::runtime
