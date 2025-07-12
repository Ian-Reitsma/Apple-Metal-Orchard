#import <ATen/ATen.h>
#import <torch/extension.h>
#import "flash_attn.h"

at::Tensor orchard_flash_attn_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float scale,
    bool causal) {
  // stub: call stock MPS attention until kernel is wired
  return at::native::scaled_dot_product_attention(q, k, v, {}, 0.0, causal, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("orchard_flash_attn_fwd", &orchard_flash_attn_fwd);
}
