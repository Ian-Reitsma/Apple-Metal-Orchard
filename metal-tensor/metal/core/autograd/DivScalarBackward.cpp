#include "DivScalarBackward.h"

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

DivScalarBackward::DivScalarBackward(tensor::Tensor before,
                                     tensor::Tensor &after, float s, bool sf)
    : a(std::move(before)), pa(&after), scalar(s), safe(sf) {}

void DivScalarBackward::apply(Tensor &g) {
  Tensor gg = g.to(Device::cpu);
  Tensor ga = Tensor::empty(gg.shape(), gg.dtype(), Device::cpu);
  auto *gp = static_cast<const float *>(gg.data_ptr());
  auto *gap = static_cast<float *>(ga.data_ptr());
  std::size_t n = gg.numel();
  if (safe && scalar == 0.0f) {
    for (std::size_t i = 0; i < n; ++i)
      gap[i] = 0.0f;
  } else {
    for (std::size_t i = 0; i < n; ++i)
      gap[i] = gp[i] / scalar;
  }
  Tensor ga_t = ga.to(pa->device());
  if (a.grad_fn()) {
    a.grad_fn()->apply(ga_t);
  } else {
    accumulate(*pa, ga_t);
  }
}

} // namespace orchard::core::autograd
