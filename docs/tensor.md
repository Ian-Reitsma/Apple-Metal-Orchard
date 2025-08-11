# Tensor v0 Overview

`metal-tensor` provides a minimal tensor API backed by Apple Metal. It is the foundation for a fully Metal-native forward and backward pass that aims to surpass PyTorch on Apple Silicon. This document outlines the features currently implemented and how to use them.

- rank-8 shapes and strides that describe up to eight dimensions with explicit byte steps
- intrusive reference counting so storage lifetimes are deterministic and thread safe
- zero-copy CPU and GPU transfers that share underlying storage when devices match
- allocation profiling with live tensor dumps for leak analysis
- matmul, sum, mean, and view gradients extending the autograd graph beyond elementwise add
- Metal kernels drive forward and backward passes for matmul and whole-tensor reductions; view gradients reshape without computation

## Toolchain

Building requires Apple's Xcode command line tools and the Metal SDK.

1. Install the tools with xcode-select --install.
2. Confirm availability by running xcode-select -p and verifying a path is printed.
3. Verify the SDK with xcrun --sdk macosx --show-sdk-path.
4. Configure and build the project with cmake -S . -B build followed by cmake --build build. The scripts automatically align CMAKE_OSX_SYSROOT and CMAKE_OSX_DEPLOYMENT_TARGET with the detected SDK.
5. Agents working on Linux must still attempt these commands and record the failure output in pull requests.

## Zero-Copy Construction

Wrap existing host data without copying using Tensor::fromData. The pointer must be sixty-four byte aligned and may carry an optional deleter. Include metal/core/tensor/Tensor.h, define a buffer such as float buffer[16] aligned to sixty-four bytes, and call Tensor::fromData on that buffer with the desired shape, data type, and device.

## Slice and View Semantics

view now checks that the requested shape covers the same number of elements as the original tensor. slice records the starting offset in bytes so chained views maintain correct addressing.

## Device Transfers

Tensor::to moves data between devices. When source and destination devices match, the call returns a view with shared storage. CPU to CPU copies use memcpy while CPU to Metal copies employ a transient MTLBlitCommandEncoder obtained from MetalContext. For example, a CPU tensor created with Tensor::empty can be sent to the mps device via to(Device::mps) and then returned to the CPU.

## Allocation Profiling

Set ORCHARD_TENSOR_PROFILE to one to log tensor storage allocations and frees to /tmp/orchard_tensor_profile.log. The log records alloc, free, and live events with storage labels and sizes. Call dump_live_tensors at any point to append all currently live allocations to the log. Include metal/core/tensor/Debug.h and invoke dump_live_tensors.

## Next Steps
- Expand the operator set and autograd coverage
- Implement optimised Metal kernels for core operations
- Grow the test suite to cover new functionality
