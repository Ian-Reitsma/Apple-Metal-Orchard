# Tensor v0 Overview

`metal-tensor` provides a minimal tensor API backed by Apple Metal. It is the
foundation for a fully Metal-native forward and backward pass that aims to
surpass PyTorch on Apple Silicon. This document will grow with the
implementation and currently outlines the planned features:

- rank-8 shapes and strides
- intrusive reference counting
- zero-copy transfers between CPU and GPU

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

Example usage:

```cpp
#include <metal/core/tensor/Tensor.h>

using namespace orchard::core::tensor;

int main() {
    Tensor t = Tensor::empty({2,3}, DType::f32, Device::mps);
    return 0;
}
```

More details will be added as the API stabilises.
