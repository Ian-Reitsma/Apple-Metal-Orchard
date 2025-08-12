#include "DivScalarBackward.h"

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

DivScalarBackward::DivScalarBackward(const Tensor &aa, float s, bool sf)
    : a(aa), scalar(s), safe(sf) {}

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
  accumulate(a, ga.to(a.device()));
  if (a.grad_fn())
    a.grad_fn()->apply(a.grad());
}

} // namespace orchard::core::autograd
