#include "TransposeBackward.h"
#include "../../runtime/MetalKernels.h"

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

TransposeBackward::TransposeBackward(const Tensor &b, int d0, int d1)
    : base(b), pbase(const_cast<Tensor *>(&b)), dim0(d0), dim1(d1) {}

void TransposeBackward::apply(Tensor &g) {
  Tensor gg = g.to(pbase->device());
  Tensor out;
  if (gg.device() == Device::cpu) {
    std::size_t m = static_cast<std::size_t>(base.shape()[dim0]);
    std::size_t n = static_cast<std::size_t>(base.shape()[dim1]);
    out = Tensor::empty(base.shape(), base.dtype(), Device::cpu);
    auto *gp = static_cast<const float *>(gg.data_ptr());
    auto *op = static_cast<float *>(out.data_ptr());
    for (std::size_t k = 0; k < m * n; ++k)
      op[k] = gp[k];
  } else {
    out = Tensor::empty(base.shape(), base.dtype(), pbase->device());
    std::size_t m = static_cast<std::size_t>(base.shape()[dim0]);
    std::size_t n = static_cast<std::size_t>(base.shape()[dim1]);
    runtime::metal_transpose_backward(static_cast<const float *>(gg.data_ptr()),
                                      static_cast<float *>(out.data_ptr()), m,
                                      n);
  }
  if (pbase->grad_fn() && pbase->grad_fn().get() != this) {
    pbase->grad_fn()->apply(out);
  } else {
    accumulate(*pbase, out);
  }
}

} // namespace orchard::core::autograd
