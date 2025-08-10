# Project Status – July 2025

This document captures the current state of the Orchard effort and the path
forward. It helps new contributors understand what works today and what remains
open.

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
- `metal-tensor/` now provides the initial Tensor v0 implementation:
  - intrusive ref‑counted storage with zero‑copy `Tensor::fromData`
  - CPU↔Metal copies through `Tensor::to` and runtime blit helpers
  - allocation profiling and `dump_live_tensors()` for debug tracing
  - tests for contiguity, CPU adds, command‑queue pooling, and profiling logs
- Implementation still requires macOS with Xcode 15+ and the Metal 4 SDK and
  does not build in this Linux environment.

## Next Steps
1. Fuse the backward FlashAttention kernels and add dropout support.
2. Extend the Tensor v0 operator set and begin autograd scaffolding.
3. Replace remaining CPU fallbacks with optimised Metal kernels.
4. Stand up macOS CI running the full CMake and CTest flow.
5. Benchmark FlashAttention and core tensor ops and publish performance data.

## Getting Involved
- See the top-level `README.md` and `AGENTS.md` for build instructions and
  contributor guidelines.
- Legacy PyTorch code and benchmark scripts live under `experimental/`.
- Update this file when significant project milestones land.
