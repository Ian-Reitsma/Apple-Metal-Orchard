# metal-tensor

`metal-tensor` contains the Tensor v0 library and runtime. It provides:

- intrusive ref‑counted `Storage`
- `Tensor::empty`, `Tensor::view`, `Tensor::slice`, and zero‑copy
  `Tensor::fromData`
- CPU↔Metal transfers via `Tensor::to`
- allocation profiling and `dump_live_tensors()` for debug tracing
- a starter autograd engine with `Tensor::requires_grad`, gradient tensors, and
  a `Node`/`Edge` graph powering `backward`
- Metal compute kernels beginning with vector add, automatically selected when
  tensors live on an MPS device

## Building
1. Ensure Xcode 15+, the Metal 4 SDK, and command line tools are installed
2. From the repository root run:
   ```bash
   cmake -S . -B build -G Ninja
   cmake --build build
   ```

## Testing
Invoke the tests with:
```bash
cmake --build build --target test
```
The suite covers contiguity, CPU arithmetic, command‑queue pooling, profiling
log creation, non‑contiguous CPU→Metal→CPU transfers, and autograd gradients for
elementwise add.

## Next Steps
- Broaden autograd coverage and add more differentiable operators
- Implement Metal kernels for remaining CPU paths
- Grow the test suite to cover additional device transfers and upcoming ops
