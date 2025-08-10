# AGENTS.md – Contributor Guide

This repository is transitioning to a standalone Metal-based tensor stack.
All code now lives directly in this repository; **there are no submodules**.

## Current Status
- `metal-tensor/` currently contains header scaffolding for **Tensor v0**; the
  implementation is incomplete and requires Apple’s toolchain (Xcode 15+ and
  the Metal SDK) to build.
- `experimental/` holds the older PyTorch path. It includes a working forward
  FlashAttention kernel that is invoked through a monkey‑patch. The backward
  pass and fused dropout are not yet implemented and performance gains only
  appear at long sequence lengths (≥4K tokens).
- The repository is pivoting toward a **full Metal-native forward and backward
  pass** for tensor operations. The existing PyTorch bridge remains only for
  regression tests while we rebuild the stack from scratch to outperform
  PyTorch on Apple Silicon.
- See `docs/project_status.md` for a snapshot of ongoing work and next steps.

## Layout
- `metal-tensor/` – core tensor and runtime libraries
- `experimental/` – legacy PyTorch-based code kept for reference
- `docs/` – project documentation

## Building
```bash
cmake -S . -B build
cmake --build build
```
Pass `-DORCHARD_BUILD_EXPERIMENTAL=ON` to build the legacy PyTorch bridge.

## Testing
Tests will reside in `metal-tensor/tests` and can be executed with
`cmake --build build --target test`.

## Guidelines
- Use C++20 and clang-format for all C++/ObjC++ code.
- Avoid committing large binaries (>5 MB).
- Run `cmake` and the test suite before submitting a PR.

Happy hacking!
