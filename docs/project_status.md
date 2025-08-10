# Project Status – August 2025

This document captures the current state of the Orchard effort and the path forward. It helps new contributors understand what works today and what remains open.

## FlashAttention Path (Experimental)
- Forward pass is integrated via a PyTorch C++ extension and monkey-patch. The custom Metal kernel runs for all GPT-2 attention layers when the environment variable `USE_FLASH_ATTN` is set to 2.
- Supports multi-head attention, batching, causal masking, and both BF16 and FP32 data types.
- Performance matches the baseline within a tolerance of 1e-4 and delivers noticeable speedups only at long sequences between four and sixteen thousand tokens. At shorter contexts such as five hundred and twelve tokens the throughput resembles the stock MPS implementation.
- Backward pass is not yet fused; training falls back to the CPU or PyTorch implementation.
- Known limitations: the head dimension must be a multiple of eight and dropout remains unimplemented.

## Tensor v0 Path
- `metal-tensor/` provides the initial Tensor v0 implementation:
  - intrusive ref-counted storage with zero-copy Tensor::fromData for wrapping external memory
  - CPU and Metal transfers through Tensor::to with runtime helpers for blit operations
  - allocation profiling and dump_live_tensors for debug tracing
  - tests for contiguity, CPU adds, command-queue pooling, and profiling logs
  - seeded autograd and validated gradients for elementwise add across CPU and Metal
- macOS continuous integration caches builds, treats warnings as errors, and runs the test suite on every pull request.
- Implementation requires macOS with Xcode 15+ and the Metal 4 SDK. The project does not build in this Linux environment, but agents must still attempt configuration and report failures.

## Next Steps
1. Fuse the backward FlashAttention kernels and add dropout support.
2. Extend the Tensor v0 operator set and expand autograd coverage.
3. Replace remaining CPU fallbacks with optimised Metal kernels.
4. Keep macOS CI green and broaden the matrix as needed.
5. Benchmark FlashAttention and core tensor ops and publish performance data.

## Milestones
1. FlashAttention backward kernels with dropout support unlocking end-to-end training benchmarks.
2. Tensor v0 reaching feature parity with the PyTorch bridge and enabling its removal.
3. Complete replacement of CPU fallbacks with Metal kernels and expanded autograd for training loops.
4. Public 0.1 release accompanied by documentation, benchmarks, and CI coverage across supported macOS versions.

## Getting Involved
- See the top-level `README.md` and `AGENTS.md` for build instructions and contributor guidelines.
- Legacy PyTorch code and benchmark scripts live under `experimental/`.
- Update this file when significant project milestones land.
- When editing, preserve the chronological order and expand notes so future contributors understand the current capabilities and limitations of each path.
