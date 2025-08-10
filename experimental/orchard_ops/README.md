# orchard_ops

This directory hosts all Python and C++ extension modules used by the legacy
PyTorch integration. Operators defined here are loaded as part of the
experimental pipeline and are **not** required for the standalone Metal tensor
stack.

## Contents
- custom kernels built as PyTorch extensions
- Python glue code for integrating those kernels

## Next Steps
These files remain for reference while the Metal stack matures. Remove or update
them once equivalent Metal-native kernels exist in `metal-tensor`.
