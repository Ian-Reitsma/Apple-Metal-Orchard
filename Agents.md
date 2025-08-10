# AGENTS.md – Contributor Guide

This repository builds a standalone, Metal‑first tensor stack. All code lives
in this tree; there are no submodules.

## Current Status
- `metal-tensor/` exposes working pieces of **Tensor v0**:
  - intrusive ref‑counted `Storage` with zero‑copy wrapping via `Tensor::fromData`
  - `Tensor::to` performs CPU↔Metal transfers and remains zero‑copy on matching
    devices
  - allocation profiling hooks log `alloc`, `free`, and `live` events to
    `/tmp/orchard_tensor_profile.log`
  - initial autograd scaffolding with `Tensor::requires_grad`, gradient storage,
    and a `Node` graph driving `backward`
  - Metal kernels beginning with elementwise add, with CPU fallbacks when Metal
    is unavailable
- `experimental/` retains the legacy PyTorch path with a forward FlashAttention
  kernel for regression comparison.

## Recent Work
- Centralised profiling helpers in `common/Profiling.h` and added
  `dump_live_tensors()` for live allocation inspection
- Hardened `view` and `slice` semantics and tracked offsets for debugging
- Introduced the zero‑copy `Tensor::fromData` factory with optional deleter
- Wired CPU↔Metal copy paths and command‑queue pooling tests
- Broadened unit tests for contiguity, CPU vector addition, queue reuse, and
  profiling log creation
- Added stress tests for non‑contiguous CPU→Metal→CPU transfers and profiling log
  contents
- Seeded autograd and validated gradients for elementwise add across CPU and
  Metal
- Stood up macOS CI that caches builds, treats warnings as errors, and runs the
  test suite

## Next Steps
1. Expand the differentiable operator set and autograd coverage
2. Replace remaining CPU fallbacks with optimised Metal kernels
3. Broaden device‑transfer and profiling tests
4. Benchmark kernel performance and publish results

## Layout
- `metal-tensor/` – core tensor and runtime libraries
- `experimental/` – legacy PyTorch-based code kept for reference
- `docs/` – project documentation

## Building
1. Install Xcode 15+ and the Metal 4 SDK
2. Configure:
   ```bash
   cmake -S . -B build
   ```
3. Build:
   ```bash
   cmake --build build
   ```
4. Pass `-DORCHARD_BUILD_EXPERIMENTAL=ON` to include the legacy PyTorch bridge

## Testing
Run the test suite with:
```bash
cmake --build build --target test
```
Always attempt to configure and run tests before submitting a PR, even when the
Metal toolchain is unavailable.

## Guidelines
- Use C++20 and clang-format for all C++/ObjC++ code
- Avoid committing binaries larger than 5 MB
- Run `cmake` and the test suite before every PR

Happy hacking!
