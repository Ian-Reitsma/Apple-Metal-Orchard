#pragma once

#include "../tensor/Tensor.h"
#include "Node.h"

namespace orchard::core::autograd {

struct MatmulBackward : Node {
  tensor::Tensor a;
  tensor::Tensor b;
  MatmulBackward(const tensor::Tensor &aa, const tensor::Tensor &bb);
  void apply(tensor::Tensor &g) override;
};

} // namespace orchard::core::autograd
