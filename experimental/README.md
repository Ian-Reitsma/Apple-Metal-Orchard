# Experimental Code

This directory houses legacy PyTorch-based code and other artifacts kept for
reference while the new Metal tensor stack is developed. None of these files
are required to build or test the core libraries.

Contents include:
- `orchard_ops` – PyTorch extension modules
- `benchmarks`, `scripts`, `tests` – tooling and test suites that rely on
  PyTorch
- `kernel_lib` – prebuilt FlashAttention kernels

The FlashAttention path exposes a forward-only Metal kernel that can be enabled
with `USE_FLASH_ATTN=2`. It matches PyTorch numerically but currently offers
speedups only at long sequence lengths and falls back to CPU for the backward
pass. PyTorch and its dependencies must be installed to run these examples or
tests.

These files are preserved for historical context and may be removed once the
standalone Metal stack reaches feature parity.
