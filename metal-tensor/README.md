# metal-tensor

`metal-tensor` contains the Tensor v0 library and runtime. It provides:

- intrusive ref-counted Storage objects
- Tensor::empty, Tensor::view, Tensor::slice, and zero-copy Tensor::fromData
- CPU and Metal transfers via Tensor::to
- allocation profiling and dump_live_tensors for debug tracing
- a starter autograd engine with Tensor::requires_grad, gradient tensors, and a Node and Edge graph powering backward
- Metal compute kernels beginning with vector add and matmul, automatically selected when tensors live on an mps device

## Current Status
- CPU and Metal backends allocate tensors with intrusive storage and share views without copying.
- Host and device transfers through Tensor::to round-trip data between CPU and mps devices.
- Tests validate contiguity, profiling logs, command-queue pooling, multi-device transfers, alignment on non-contiguous views, and gradient propagation for elementwise add, matmul, reductions, and view transforms.
- Only vector addition and matmul have dedicated Metal kernels; other operations still execute on the CPU.

## Directory map
- `metal/` holds the implementation source
  - `common/` groups utilities such as Profiling.h and debug helpers including dump_live_tensors
  - `core/` defines Tensor, Storage, Device, and factory functions like Tensor::empty, view, slice, and fromData
  - `kernels/` contains Metal shader entry points
  - `runtime/` manages MTLDevice selection and command queue pooling referenced by Tensor::to during transfers
- `tests/` bundles unit tests centred around tensor_tests.cpp that validate contiguity, CPU arithmetic, Metal dispatch, profiling, and autograd
- `docs/` houses design specifications such as design_spec_tensor_v0.md

## Building
1. Ensure Xcode 15+, the Metal 4 SDK, and command line tools are installed.
2. From the repository root run cmake -S . -B build -G Ninja followed by cmake --build build to configure and build the library.
3. Linux hosts cannot compile the project but should still attempt these commands and include the failure output in pull requests.

## Testing
Invoke the tests with cmake --build build --target test. The suite covers contiguity, CPU arithmetic, command-queue pooling, multi-device CPU↔Metal↔CPU transfers, mixed CPU→Metal→CPU→Metal sequences, multi-threaded large tensor moves, profiling log validation with matching alloc/free counts, silent behaviour when ORCHARD_TENSOR_PROFILE is unset, alignment and zero-copy checks, and autograd gradients for elementwise add.

## Milestones
1. Autograd support for a base operator set including matmul and reductions.
2. Optimized Metal kernels for all core operations with parity to CPU fallbacks.
3. Stable 0.1 release enabling external projects to consume `liborchard_core.a` and `liborchard_metal.a`.

## Next Steps
- Broaden autograd coverage and add more differentiable operators.
- Implement Metal kernels for remaining CPU paths and retire redundant code.
- Grow the test suite to cover additional device transfers and upcoming ops.
