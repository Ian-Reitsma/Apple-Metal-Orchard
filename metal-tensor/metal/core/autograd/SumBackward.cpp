#include "SumBackward.h"
#include "../../runtime/MetalKernels.h"

#include <cstddef>

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

SumBackward::SumBackward(const Tensor &aa) : a(aa) {}

void SumBackward::apply(Tensor &g) {
  Tensor grad = Tensor::empty(a.shape(), DType::f32, g.device());
  Tensor g_cpu = g.to(Device::cpu);
  float v = *static_cast<float *>(g_cpu.data_ptr());
  if (g.device() == Device::mps) {
    runtime::metal_fill(static_cast<float *>(grad.data_ptr()), v, a.numel());
  } else {
    auto *ptr = static_cast<float *>(grad.data_ptr());
    for (std::size_t i = 0; i < a.numel(); ++i)
      ptr[i] = v;
  }
  accumulate(a, grad.to(a.device()));
  if (a.grad_fn())
    a.grad_fn()->apply(a.grad());
}

} // namespace orchard::core::autograd
