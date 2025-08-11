#include "ViewBackward.h"

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

ViewBackward::ViewBackward(const Tensor &b) : base(b) {}

void ViewBackward::apply(Tensor &g) {
  Tensor reshaped = g.view(base.shape());
  accumulate(base, reshaped);
  if (base.grad_fn())
    base.grad_fn()->apply(base.grad());
}

} // namespace orchard::core::autograd
