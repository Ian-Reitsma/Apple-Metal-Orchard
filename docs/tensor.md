# Tensor v0 Overview

`metal-tensor` provides a minimal tensor API backed by Apple Metal. It is the
foundation for a fully Metal-native forward and backward pass that aims to
surpass PyTorch on Apple Silicon. This document outlines the features currently
implemented and how to use them.

- rank-8 shapes and strides
- intrusive reference counting
- zero-copy CPU↔GPU transfers
- allocation profiling with live tensor dumps

## Toolchain

Building requires Apple's Xcode command line tools and the Metal SDK.

1. Install the tools with `xcode-select --install`.
2. Confirm availability: `xcode-select -p` should print a path.
3. Verify the SDK: `xcrun --sdk macosx --show-sdk-path`.
4. Configure and build the project:
   ```bash
   cmake -S . -B build
   cmake --build build
   ```
   The build scripts automatically align `CMAKE_OSX_SYSROOT` and
   `CMAKE_OSX_DEPLOYMENT_TARGET` with the detected SDK.

## Zero-Copy Construction

Wrap existing host data without copying using `Tensor::fromData`. The pointer
must be 64-byte aligned and may carry an optional deleter:

```cpp
#include <metal/core/tensor/Tensor.h>

float buffer[16] __attribute__((aligned(64))) = {0};
Tensor t = Tensor::fromData(buffer, {16,1,1,1,1,1,1,1}, DType::f32, Device::cpu);
```

## Slice and View Semantics

`view` now checks that the requested shape covers the same number of elements as
the original tensor. `slice` records the starting offset in bytes so chained
views maintain correct addressing.

## Device Transfers

`Tensor::to` moves data between devices. When source and destination devices
match, the call returns a view with shared storage. CPU↔CPU copies use
`memcpy`; CPU↔Metal copies employ a transient `MTLBlitCommandEncoder` obtained
from `MetalContext`:

```cpp
Tensor cpu = Tensor::empty({2,3}, DType::f32, Device::cpu);
Tensor gpu = cpu.to(Device::mps);        // copy to Metal
Tensor back = gpu.to(Device::cpu);       // copy back to CPU
```

## Allocation Profiling

Set `ORCHARD_TENSOR_PROFILE=1` to log tensor storage allocations and frees to
`/tmp/orchard_tensor_profile.log`. The log records `alloc`, `free`, and `live`
events with storage labels and sizes. Call `dump_live_tensors()` at any point to
append all currently live allocations to the log:

```cpp
#include <metal/core/tensor/Debug.h>

dump_live_tensors();
```

## Next Steps
- Expand the operator set and add autograd scaffolding
- Implement optimised Metal kernels for core operations
- Grow the test suite to cover new functionality
