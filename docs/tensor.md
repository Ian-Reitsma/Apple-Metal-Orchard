# Tensor v0 Overview

`metal-tensor` provides a minimal tensor API backed by Apple Metal. This
document will grow with the implementation and currently outlines the planned
features:

- rank-8 shapes and strides
- intrusive reference counting
- zero-copy transfers between CPU and GPU

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
