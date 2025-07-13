#import <ATen/ATen.h>
#include <torch/script.h>
#include <atomic>
#include <fstream>

static std::atomic<int> flashattn_call_count(0);

at::Tensor orchard_flash_attn_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double scale,
    bool causal)
{
  flashattn_call_count++;
  if (flashattn_call_count <= 1000 || flashattn_call_count % 1000 == 0) {
    std::ofstream log("/tmp/flashattn_kernel_calls.log", std::ios_base::app);
    log << "[flashattn.mm] orchard_flash_attn_fwd called, count=" << flashattn_call_count << std::endl;
  }
  return at::native::scaled_dot_product_attention(q, k, v, {}, 0.0, causal, static_cast<float>(scale));
}

TORCH_LIBRARY(flash_attn_mps, m) {
    m.def("_flash_attn_fwd(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor", orchard_flash_attn_fwd);
}
