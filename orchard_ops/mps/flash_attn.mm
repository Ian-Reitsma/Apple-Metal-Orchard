// orchard_ops/mps/flash_attn.mm
#import <ATen/ATen.h>
#include <torch/script.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <vector>
#include <tuple>
#include <atomic>
#include <fstream>

// --- Global kernel call counter (for debug logging) ---
static std::atomic<int> flashattn_call_count(0);

// --- FORWARD: calls PyTorch's native scaled_dot_product_attention ---
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
        log << "[flashattn.mm] FWD call=" << flashattn_call_count << '\n';
    }
    // Always call PyTorch's fused forward (best baseline)
    return at::native::scaled_dot_product_attention(
        q, k, v, /*attn_mask=*/{}, /*dropout_p=*/0.0, causal, static_cast<float>(scale));
}

// --- BACKWARD: calls PyTorch's dispatcher for backward kernel (C++ path!) ---
std::tuple<at::Tensor, at::Tensor, at::Tensor> orchard_flash_attn_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double scale,
    bool causal)
{
    flashattn_call_count++;
    if (flashattn_call_count <= 1000 || flashattn_call_count % 1000 == 0) {
        std::ofstream log("/tmp/flashattn_kernel_calls.log", std::ios_base::app);
        log << "[flashattn.mm] BWD call=" << flashattn_call_count << '\n';
    }
    std::vector<c10::IValue> stack = {
        grad_out, q, k, v,
        c10::IValue(),   // attn_mask=None
        c10::IValue(),   // attn_bias=None
        0.0,             // dropout_p
        causal,          // is_causal
        static_cast<float>(scale) // scale
    };
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::scaled_dot_product_attention_backward", "");
    op.callBoxed(&stack);
    auto result_tuple = stack.back().toTuple();
    at::Tensor grad_q = result_tuple->elements()[0].toTensor();
    at::Tensor grad_k = result_tuple->elements()[1].toTensor();
    at::Tensor grad_v = result_tuple->elements()[2].toTensor();
    return std::make_tuple(grad_q, grad_k, grad_v);
}

// --- Register with Torch dispatcher under correct schema ---
static auto fwd_schema =
    "flash_attn_mps::_flash_attn_fwd(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor";
static auto bwd_schema =
    "flash_attn_mps::_flash_attn_bwd(Tensor grad_out, Tensor q, Tensor k, Tensor v, float scale, bool causal) -> (Tensor, Tensor, Tensor)";

TORCH_LIBRARY(flash_attn_mps, m) {
    m.def(fwd_schema, orchard_flash_attn_fwd);
    m.def(bwd_schema, orchard_flash_attn_bwd);
}
