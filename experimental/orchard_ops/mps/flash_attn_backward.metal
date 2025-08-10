#include <metal_stdlib>
using namespace metal;

kernel void flash_attn_bwd(const device float* grad_out [[buffer(0)]],
                           const device float* mask [[buffer(1)]],
                           device float* grad_q [[buffer(2)]],
                           device float* grad_k [[buffer(3)]],
                           device float* grad_v [[buffer(4)]],
                           constant uint& n [[buffer(5)]],
                           constant float& scale [[buffer(6)]],
                           constant float& dropout_p [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float g = grad_out[gid] * mask[gid] / (1.0 - dropout_p);
    grad_q[gid] = g * scale;
    grad_k[gid] = 0.0;
    grad_v[gid] = 0.0;
}
