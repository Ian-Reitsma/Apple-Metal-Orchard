# Project Status – July 2025

This document captures the current state of the Orchard effort and the path
forward. It should help new contributors understand what works today and what
remains open.

## FlashAttention Path (Experimental)
- Forward pass is integrated via a PyTorch C++ extension and monkey‑patch. The
  custom Metal kernel runs for all GPT‑2 attention layers when
  `USE_FLASH_ATTN=2` is set.
- Supports multi‑head, batching, causal masking and both BF16 and FP32.
- Performance matches the baseline within 1e‑4 and delivers noticeable
  speedups only at long sequences (4K–16K tokens). At shorter contexts (e.g.
  512 tokens) throughput is similar to the stock MPS implementation.
- Backward pass is not yet fused; training falls back to the CPU/PyTorch
  implementation.
- Known limitations: `head_dim` must be a multiple of 8 and dropout is
  unimplemented.

## Tensor v0 Path
- `metal-tensor/` contains header scaffolding for a new Metal-first tensor
  library. The design targets intrusive ref‑counted storage, rank‑8 shapes and
  zero-copy CPU↔GPU transfers.
- Implementation is incomplete and currently does not build in this Linux
  environment. Development requires macOS with Xcode 15+ and the Metal 4 SDK.

## Next Steps
1. Fuse the backward FlashAttention kernels and add dropout support.
2. Flesh out the Tensor v0 headers into a working library with allocator,
   runtime contexts and tests.
3. Benchmark FlashAttention at long sequence lengths and document the speedups
   once the kernel is fused.
4. Expand documentation and CI so the project can be built and tested on Apple
   Silicon without relying on PyTorch.

## Getting Involved
- See the top-level `README.md` and `AGENTS.md` for build instructions and
  contributor guidelines.
- Legacy PyTorch code and benchmark scripts live under `experimental/`.
- Updates to this file should accompany significant project milestones.
