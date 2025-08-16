#include "core/autograd/Node.h"
#include "core/tensor/Tensor.h"
#include "runtime/MetalContext.h"
#include <array>
#include <cstdlib>
#include <gtest/gtest.h>

using namespace orchard::core::tensor;
using namespace orchard::core::autograd;

struct AccNode : Node {
  using Node::accumulate;
  void apply(Tensor &) override {}
};

TEST(FallbackTest, AccumulateFallsBackToCpu) {
  std::array<std::int64_t, 8> shape{2, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::mps);
  t.set_requires_grad(true);
  Tensor g = Tensor::empty(shape, DType::f32, Device::mps);
  auto *gptr = static_cast<float *>(g.data_ptr());
  gptr[0] = 1.0f;
  gptr[1] = 1.0f;
  AccNode::accumulate(t, g);
  auto *grad = static_cast<float *>(t.grad().data_ptr());
  EXPECT_FLOAT_EQ(grad[0], 1.0f);
  EXPECT_FLOAT_EQ(grad[1], 1.0f);
}
#ifdef __APPLE__
TEST(FallbackTest, CopyBuffersThrowsWithoutDevice) {
  std::array<std::int64_t, 8> shape{1, 1, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(shape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(shape, DType::f32, Device::cpu);
  EXPECT_THROW(
      {
        orchard::runtime::metal_copy_buffers(a.data_ptr(), b.data_ptr(),
                                             sizeof(float));
      },
      std::runtime_error);
}
#endif

TEST(FallbackTest, AddFallsBackWhenKernelMissing) {
  const char *orig = std::getenv("ORCHARD_KERNEL_DIR");
  setenv("ORCHARD_KERNEL_DIR", "/tmp/orchard_missing", 1);
  std::array<std::int64_t, 8> shape{1, 1, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(shape, DType::f32, Device::mps);
  Tensor b = Tensor::empty(shape, DType::f32, Device::mps);
  a.fill(1.0f);
  b.fill(1.0f);
  Tensor out;
  EXPECT_NO_THROW({ out = a.add(b); });
  auto *ptr = static_cast<float *>(out.data_ptr());
  EXPECT_FLOAT_EQ(ptr[0], 2.0f);
  if (orig)
    setenv("ORCHARD_KERNEL_DIR", orig, 1);
  else
    unsetenv("ORCHARD_KERNEL_DIR");
}
