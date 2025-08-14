# Project Status – August 2025

This document captures the current state of the Orchard effort and the path forward. It helps new contributors understand what works today and what remains open.

## FlashAttention Path (Experimental)
- Forward pass is integrated via a PyTorch C++ extension and monkey-patch. The custom Metal kernel runs for all GPT-2 attention layers when the environment variable `USE_FLASH_ATTN` is set to 2.
- Supports multi-head attention, batching, causal masking, and both BF16 and FP32 data types.
- Performance matches the baseline within a tolerance of 1e-4 and delivers noticeable speedups only at long sequences between four and sixteen thousand tokens. At shorter contexts such as five hundred and twelve tokens the throughput resembles the stock MPS implementation.
- Backward pass now employs a fused Metal kernel that applies the dropout mask, rescales gradients, and computes outputs for query, key, and value tensors.
- Known limitations: the head dimension must be a multiple of eight and dropout probabilities must lie within `[0,1)`.

## Tensor v0 Path
- `metal-tensor/` provides the initial Tensor v0 implementation:
  - intrusive ref-counted storage with zero-copy Tensor::fromData for wrapping external memory
  - CPU and Metal transfers through Tensor::to with runtime helpers for blit operations
    - allocation profiling and dump_live_tensors for debug tracing
      - tests for contiguity, CPU adds, command-queue pooling, reductions, elementwise division, filling, detachment, and profiling logs that can be toggled at runtime via `tensor_profile_reset`
      - each storage creation logs a single `alloc` entry and the final release logs a matching `free` entry to track allocator symmetry under load
    - helper Tensor::is_alias_of with tests mutating detached views and clone-before-detach paths to verify storage aliasing
    - seeded autograd and validated gradients for elementwise add, divide, matmul, mean, reductions, and view transforms across CPU and Metal
    - division gradients dispatch by device allowing CPU-only builds without Metal
    - the Metal allocator falls back to CPU memory on non-Apple hosts while still logging profiling events
      - autograd nodes snapshot pre-mutation values to avoid recursive gradient application with tests covering chained and repeated in-place scalar divisions and CPU add and mul backward paths
      - vector and matrix broadcast tests now align shapes to prevent prior crashes
      - safe division masks zeros with broadcast-aware strides so CPU and Metal results match
      - sum and mean shift stride metadata when dimensions drop to keep source offsets correct
      - profiling reads `ORCHARD_TENSOR_PROFILE` on each query and pairs every `alloc` with a matching `free`
    - macOS continuous integration caches builds, treats warnings as errors, and runs the test suite on every pull request.
- Implementation requires macOS with Xcode 15+ and the Metal 4 SDK. The project does not build in this Linux environment, but agents must still attempt configuration and report failures.
- A minimal copy of GoogleTest resides under `third_party/googletest` so tests compile without downloads; upstream tests and samples were dropped to reduce repository size.

## Next Steps
1. Harden the fused FlashAttention backward kernels and benchmark training loops that depend on dropout.
2. Extend the Tensor v0 operator set and expand autograd coverage beyond division, mean, and transpose.
3. Replace remaining CPU fallbacks with optimised Metal kernels.
4. Fix CMake configuration on non-Apple platforms so CPU-only builds and tests succeed without Objective-C++.
5. Keep macOS CI green and broaden the matrix as needed.
 6. Broaden the test matrix with additional stress cases and CPU-only execution paths.
7. Benchmark FlashAttention and core tensor ops and publish performance data.

## Milestones
1. FlashAttention backward kernels with dropout support unlocking end-to-end training benchmarks.
2. Tensor v0 reaching feature parity with the PyTorch bridge and enabling its removal.
3. Complete replacement of CPU fallbacks with Metal kernels and expanded autograd for training loops including division and detachment semantics.
4. Public 0.1 release accompanied by documentation, benchmarks, and CI coverage across supported macOS versions.

## Getting Involved
- See the top-level `README.md` and `AGENTS.md` for build instructions and contributor guidelines.
- Legacy PyTorch code and benchmark scripts live under `experimental/`.
- Update this file when significant project milestones land.
- When editing, preserve the chronological order and expand notes so future contributors understand the current capabilities and limitations of each path.
