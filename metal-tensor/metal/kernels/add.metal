#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(const device float *a [[buffer(0)]],
                       const device float *b [[buffer(1)]],
                       device float *c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
  c[id] = a[id] + b[id];
}
