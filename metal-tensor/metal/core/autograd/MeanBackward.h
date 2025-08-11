#pragma once

#include "../tensor/Tensor.h"
#include "Node.h"

namespace orchard::core::autograd {

struct MeanBackward : Node {
  tensor::Tensor a;
  explicit MeanBackward(const tensor::Tensor &aa);
  void apply(tensor::Tensor &g) override;
};

} // namespace orchard::core::autograd
