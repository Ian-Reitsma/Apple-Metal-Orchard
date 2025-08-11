#pragma once

#include "../tensor/Tensor.h"
#include "Node.h"

namespace orchard::core::autograd {

struct ViewBackward : Node {
  tensor::Tensor base;
  explicit ViewBackward(const tensor::Tensor &b);
  void apply(tensor::Tensor &g) override;
};

} // namespace orchard::core::autograd
