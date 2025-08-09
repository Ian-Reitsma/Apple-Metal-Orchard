# Experimental Code

This directory contains legacy PyTorch-based code and other artifacts kept for
reference while the new Metal tensor stack is developed. Nothing here is
required for building the core libraries.

Contents may include:
- `orchard_ops` – PyTorch extension modules
- `benchmarks`, `scripts`, `tests` – tooling and test suites relying on PyTorch
- `kernel_lib` – prebuilt FlashAttention kernels

The FlashAttention path exposes a forward-only Metal kernel that can be enabled
with `USE_FLASH_ATTN=2`. It matches PyTorch numerically but currently offers
speedups only at long sequence lengths and falls back to CPU for the backward
pass. PyTorch and its dependencies are required to run any of these examples or
tests.

These files are provided for historical context and may be removed once the
standalone tensor stack reaches feature parity.
