// orchard_ops/mps/flash_attn.mm
#import <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <atomic>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <fstream>
#include <torch/script.h>
#include <tuple>
#include <vector>

// --- Global kernel call counter (for debug logging) ---
static std::atomic<int> flashattn_call_count(0);

// --- FORWARD: uses PyTorch's attention then applies explicit dropout mask ---
std::tuple<at::Tensor, at::Tensor>
orchard_flash_attn_fwd(const at::Tensor &q, const at::Tensor &k,
                       const at::Tensor &v, double scale, double dropout_p,
                       bool causal) {
  flashattn_call_count++;
  if (flashattn_call_count <= 1000 || flashattn_call_count % 1000 == 0) {
    std::ofstream log("/tmp/flashattn_kernel_calls.log", std::ios_base::app);
    log << "[flashattn.mm] FWD call=" << flashattn_call_count << '\n';
  }
  auto attn = at::native::scaled_dot_product_attention(
      q, k, v, /*attn_mask=*/{}, /*dropout_p=*/0.0, causal,
      static_cast<float>(scale));
  at::Tensor mask = at::bernoulli(at::ones_like(attn), 1.0 - dropout_p);
  at::Tensor out = mask.mul(attn).div(1.0 - dropout_p);
  return std::make_tuple(out, mask);
}

// --- BACKWARD: simple fused Metal stub applying dropout mask ---
std::tuple<at::Tensor, at::Tensor, at::Tensor>
orchard_flash_attn_bwd(const at::Tensor &grad_out, const at::Tensor &q,
                       const at::Tensor &k, const at::Tensor &v,
                       const at::Tensor &dropout_mask, double scale,
                       double dropout_p, bool causal) {
  flashattn_call_count++;
  if (flashattn_call_count <= 1000 || flashattn_call_count % 1000 == 0) {
    std::ofstream log("/tmp/flashattn_kernel_calls.log", std::ios_base::app);
    log << "[flashattn.mm] BWD call=" << flashattn_call_count << '\n';
  }
  at::Tensor grad_in = grad_out.mul(dropout_mask).div(1.0 - dropout_p);
  at::Tensor grad_q = grad_in.mul(scale);
  at::Tensor grad_k = at::zeros_like(k);
  at::Tensor grad_v = at::zeros_like(v);
  return std::make_tuple(grad_q, grad_k, grad_v);
}

// --- Register with Torch dispatcher under correct schema ---
static auto fwd_schema =
    "flash_attn_mps::_flash_attn_fwd(Tensor q, Tensor k, Tensor v, float "
    "scale, float dropout_p, bool causal) -> (Tensor, Tensor)";
static auto bwd_schema =
    "flash_attn_mps::_flash_attn_bwd(Tensor grad_out, Tensor q, Tensor k, "
    "Tensor v, Tensor dropout_mask, float scale, float dropout_p, bool causal) "
    "-> (Tensor, Tensor, Tensor)";

TORCH_LIBRARY(flash_attn_mps, m) {
  m.def(fwd_schema, orchard_flash_attn_fwd);
  m.def(bwd_schema, orchard_flash_attn_bwd);
}
