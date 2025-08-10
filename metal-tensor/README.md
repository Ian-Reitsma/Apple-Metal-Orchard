# metal-tensor

`metal-tensor` contains the Tensor v0 library and runtime. It provides:

- intrusive ref‑counted `Storage`
- `Tensor::empty`, `Tensor::view`, `Tensor::slice`, and zero‑copy
  `Tensor::fromData`
- CPU↔Metal transfers via `Tensor::to`
- allocation profiling and `dump_live_tensors()` for debug tracing

## Building
1. Ensure Xcode 15+ and the Metal 4 SDK are installed
2. From the repository root run:
   ```bash
   cmake -S . -B build
   cmake --build build
   ```

## Testing
Invoke the tests with:
```bash
cmake --build build --target test
```
The test suite covers contiguity, CPU arithmetic, command‑queue pooling, and
profiling log creation.

## Next Steps
- Expand the operator set and autograd scaffolding
- Implement Metal kernels for remaining CPU paths
- Grow the test suite to cover device transfers and new operators
