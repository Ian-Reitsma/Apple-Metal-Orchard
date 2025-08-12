#include "DivBackward.h"
#include "../../runtime/MetalKernels.h"

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

DivBackward::DivBackward(const Tensor &aa, const Tensor &bb) : a(aa), b(bb) {}

void DivBackward::apply(Tensor &g) {
  Device dev = g.device();
  Tensor ga = Tensor::empty(a.shape(), DType::f32, dev);
  Tensor gb = Tensor::empty(b.shape(), DType::f32, dev);
  Tensor gg = g.to(dev);
  Tensor aa = a.to(dev);
  Tensor bb = b.to(dev);
  std::size_t n = a.numel();
  runtime::metal_div_backward_a(static_cast<const float *>(gg.data_ptr()),
                                static_cast<const float *>(bb.data_ptr()),
                                static_cast<float *>(ga.data_ptr()), n);
  runtime::metal_div_backward_b(static_cast<const float *>(gg.data_ptr()),
                                static_cast<const float *>(aa.data_ptr()),
                                static_cast<const float *>(bb.data_ptr()),
                                static_cast<float *>(gb.data_ptr()), n);
  accumulate(a, ga.to(a.device()));
  accumulate(b, gb.to(b.device()));
  if (a.grad_fn())
    a.grad_fn()->apply(a.grad());
  if (b.grad_fn())
    b.grad_fn()->apply(b.grad());
}

} // namespace orchard::core::autograd
