# Tests

This directory collects unit and integration tests for the experimental
PyTorch-based path. Tests rely on PyTorch and its dependencies and do not cover
the new Metal tensor stack.

## Running
Execute the tests with `pytest` from this directory. Ensure PyTorch and all required Python packages are installed.

## Next Steps
Expand or retire these tests as the Metal-native stack reaches parity and the experimental path is removed. Recent additions verify gradients for FlashAttention with dropout enabled, providing coverage while the fused Metal kernels mature.
