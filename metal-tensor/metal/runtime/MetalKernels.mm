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
#else
void metal_add(const float *a, const float *b, float *c, std::size_t n) {
  cpu_context().add(a, b, c, n);
}
#endif

} // namespace orchard::runtime
