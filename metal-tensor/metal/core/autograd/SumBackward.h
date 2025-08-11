#pragma once

#include "../tensor/Tensor.h"
#include "Node.h"

namespace orchard::core::autograd {

struct SumBackward : Node {
  tensor::Tensor a;
  explicit SumBackward(const tensor::Tensor &aa);
  void apply(tensor::Tensor &g) override;
};

} // namespace orchard::core::autograd
