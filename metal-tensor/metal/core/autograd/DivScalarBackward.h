#pragma once

#include "../tensor/Tensor.h"
#include "Node.h"

namespace orchard::core::autograd {

struct DivScalarBackward : Node {
  tensor::Tensor a;
  tensor::Tensor *pa;
  float scalar;
  bool safe{false};
  DivScalarBackward(const tensor::Tensor &aa, float s, bool sf);
  void apply(tensor::Tensor &g) override;
};

} // namespace orchard::core::autograd
