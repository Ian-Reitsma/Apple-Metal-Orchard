#include "MatmulBackward.h"
#include "../../runtime/MetalKernels.h"

#include <cstddef>

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

MatmulBackward::MatmulBackward(const Tensor &aa, const Tensor &bb)
    : a(aa), b(bb) {}

void MatmulBackward::apply(Tensor &g) {
  auto m = a.shape()[0];
  auto n = b.shape()[1];
  auto k = a.shape()[1];
  Device dev = g.device();
  Tensor ga = Tensor::empty(a.shape(), DType::f32, dev);
  Tensor gb = Tensor::empty(b.shape(), DType::f32, dev);
  Tensor aa = a.to(dev);
  Tensor bb = b.to(dev);
  if (dev == Device::mps) {
    runtime::metal_matmul_backward_a(static_cast<const float *>(g.data_ptr()),
                                     static_cast<const float *>(bb.data_ptr()),
                                     static_cast<float *>(ga.data_ptr()), m, n,
                                     k);
    runtime::metal_matmul_backward_b(static_cast<const float *>(g.data_ptr()),
                                     static_cast<const float *>(aa.data_ptr()),
                                     static_cast<float *>(gb.data_ptr()), m, n,
                                     k);
  } else {
    const auto *gp = static_cast<const float *>(g.data_ptr());
    const auto *bp = static_cast<const float *>(bb.data_ptr());
    const auto *ap = static_cast<const float *>(aa.data_ptr());
    auto *gap = static_cast<float *>(ga.data_ptr());
    auto *gbp = static_cast<float *>(gb.data_ptr());
    for (std::size_t i = 0; i < static_cast<std::size_t>(m); ++i) {
      for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
        float s = 0.0f;
        for (std::size_t p = 0; p < static_cast<std::size_t>(n); ++p)
          s += gp[i * n + p] * bp[j * n + p];
        gap[i * k + j] = s;
      }
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(k); ++i) {
      for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
        float s = 0.0f;
        for (std::size_t p = 0; p < static_cast<std::size_t>(m); ++p)
          s += ap[p * k + i] * gp[p * n + j];
        gbp[i * n + j] = s;
      }
    }
  }
  accumulate(a, ga.to(a.device()));
  accumulate(b, gb.to(b.device()));
  if (a.grad_fn())
    a.grad_fn()->apply(a.grad());
  if (b.grad_fn())
    b.grad_fn()->apply(b.grad());
}

} // namespace orchard::core::autograd
