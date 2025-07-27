# orchard_ops/flash_attn_function.py

"""
Autograd wrapper for Metal FlashAttention in Orchard.
- Handles forward and backward with strict Metal op checks.
- Fails clearly if kernels or PyTorch are missing.
- Optional debug path can fall back to PyTorch CPU/MPS backward (for CI/dev only).
"""

import sys
import os

try:
    import torch
    from torch.autograd import Function
except Exception:  # pragma: no cover
    torch = None
    class Function: pass

_DEBUG = bool(int(os.environ.get("ORCHARD_DEBUG_FLASHATN", "0"))) if "os" in sys.modules else False

def _fail(msg: str) -> None:
    raise RuntimeError(f"[orchard][flash_attn_function] {msg}")

def _metal_kernel_available():
    return (
        torch is not None and
        hasattr(torch.ops, "flash_attn_mps") and
        hasattr(torch.ops.flash_attn_mps, "_flash_attn_fwd") and
        hasattr(torch.ops.flash_attn_mps, "_flash_attn_bwd")
    )

class FlashAttnFunction(Function):
    """
    Autograd function for Metal FlashAttention (forward and backward).
    If Metal backward is not available, devs can enable PyTorch fallback via _DEBUG.
    """

    @staticmethod
    def forward(ctx, q, k, v, scale: float, causal: bool):
        if torch is None:
            _fail("PyTorch not available (did you install torch?)")
        if not hasattr(torch.ops, "flash_attn_mps") or not hasattr(torch.ops.flash_attn_mps, "_flash_attn_fwd"):
            _fail("flash_attn_mps kernel not loaded (did you call enable_flash.main()?)")
        # Only allow 'mps' device for the Metal path
        if any(t.device.type != 'mps' for t in (q, k, v)):
            _fail(f"Input tensors must be on 'mps' device, got {[t.device for t in (q, k, v)]}")
        ctx.scale = scale
        ctx.causal = causal
        ctx.save_for_backward(q, k, v)
        if _DEBUG:
            print(f"[orchard][FlashAttnFunction.forward] Shapes: q={q.shape}, scale={scale}, causal={causal}")
        return torch.ops.flash_attn_mps._flash_attn_fwd(q, k, v, float(scale), causal)

    @staticmethod
    def backward(ctx, grad_out):
        if torch is None:
            _fail("PyTorch not available")
        q, k, v = ctx.saved_tensors
        metal_kernel_ok = (
            hasattr(torch.ops, "flash_attn_mps") and
            hasattr(torch.ops.flash_attn_mps, "_flash_attn_bwd")
        )
        if metal_kernel_ok:
            grad_q, grad_k, grad_v = torch.ops.flash_attn_mps._flash_attn_bwd(
                grad_out, q, k, v, float(ctx.scale), ctx.causal
            )
        elif _DEBUG:
            print("[orchard][FlashAttnFunction.backward] Metal kernel unavailable, attempting PyTorch fallback.", file=sys.stderr)
            # DEV/CI fallback: use PyTorch autograd for backward if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                with torch.enable_grad():
                    q_, k_, v_ = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q_, k_, v_, dropout_p=0.0, is_causal=ctx.causal
                    )
                    grads = torch.autograd.grad(out, (q_, k_, v_), grad_out)
                    grad_q, grad_k, grad_v = grads
            else:
                _fail("Neither Metal nor PyTorch fallback for FlashAttention backward is available.")
        else:
            _fail("Metal FlashAttention backward kernel unavailable (no fallback; set ORCHARD_DEBUG_FLASHATN=1 for PyTorch dev mode).")
        return grad_q, grad_k, grad_v, None, None


def flash_attn(q, k, v, scale, causal=False):
    if torch is None:
        _fail("PyTorch not available")
    return FlashAttnFunction.apply(q, k, v, scale, causal)
