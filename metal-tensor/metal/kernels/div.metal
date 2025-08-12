#include <metal_stdlib>
using namespace metal;

kernel void div_arrays(const device float *a [[buffer(0)]],
                       const device float *b [[buffer(1)]],
                       device float *c [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
  c[gid] = a[gid] / b[gid];
}

kernel void div_backward_a(const device float *grad [[buffer(0)]],
                           const device float *b [[buffer(1)]],
                           device float *out [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  out[gid] = grad[gid] / b[gid];
}

kernel void div_backward_b(const device float *grad [[buffer(0)]],
                           const device float *a [[buffer(1)]],
                           const device float *b [[buffer(2)]],
                           device float *out [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  out[gid] = -grad[gid] * a[gid] / (b[gid] * b[gid]);
}
