#include "TransposeBackward.h"
#include "../../runtime/MetalKernels.h"

using namespace orchard::core::tensor;

namespace orchard::core::autograd {

TransposeBackward::TransposeBackward(const Tensor &b, int d0, int d1)
    : base(b), dim0(d0), dim1(d1) {}

void TransposeBackward::apply(Tensor &g) {
  int m = static_cast<int>(base.shape()[dim0]);
  int n = static_cast<int>(base.shape()[dim1]);
  Device dev = g.device();
  Tensor out = Tensor::empty(base.shape(), DType::f32, dev);
  Tensor gg = g.to(dev);
  runtime::metal_transpose_backward(static_cast<const float *>(gg.data_ptr()),
                                    static_cast<float *>(out.data_ptr()), m, n);
  accumulate(base, out.to(base.device()));
  if (base.grad_fn())
    base.grad_fn()->apply(base.grad());
}

} // namespace orchard::core::autograd
