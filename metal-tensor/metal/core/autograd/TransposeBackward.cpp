#include "TransposeBackward.h"

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

TransposeBackward::TransposeBackward(const Tensor &b, int d0, int d1)
    : base(b), pbase(const_cast<Tensor *>(&b)), dim0(d0), dim1(d1) {}

void TransposeBackward::apply(Tensor &g) {
  Tensor routed = g.to(pbase->device());
  accumulate(*pbase, routed);
  if (pbase->grad_fn() && pbase->grad_fn().get() != this)
    pbase->grad_fn()->apply(routed);
}

} // namespace orchard::core::autograd
