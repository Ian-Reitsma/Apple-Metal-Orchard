#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>

#include "core/tensor/Tensor.h"
#include "runtime/Allocator.h"

using namespace orchard::core::tensor;

TEST(TensorTest, ToCpuZeroCopy) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  Tensor cpu = t.to(Device::cpu);
  EXPECT_EQ(t.data_ptr(), cpu.data_ptr());
}

TEST(TensorTest, ViewSliceMutation) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *base = static_cast<float *>(t.data_ptr());
  for (int i = 0; i < 4; ++i)
    base[i] = static_cast<float>(i);

  std::array<std::int64_t, 8> newShape{2, 2, 1, 1, 1, 1, 1, 1};
  Tensor v = t.view(newShape);
  auto *vptr = static_cast<float *>(v.data_ptr());
  vptr[1] = 42.0f;
  EXPECT_EQ(base[1], 42.0f);

  Tensor s = t.slice(0, 0, 2);
  auto *sptr = static_cast<float *>(s.data_ptr());
  sptr[1] = 99.0f;
  EXPECT_EQ(base[1], 99.0f);
}

TEST(TensorTest, CloneDistinctStorage) {
  std::array<std::int64_t, 8> shape{2, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *base = static_cast<float *>(t.data_ptr());
  base[0] = 1.0f;
  base[1] = 2.0f;
  Tensor c = t.clone();
  EXPECT_NE(c.data_ptr(), t.data_ptr());
  auto *cptr = static_cast<float *>(c.data_ptr());
  EXPECT_EQ(cptr[0], base[0]);
  base[0] = 3.0f;
  EXPECT_NE(cptr[0], base[0]);
}

TEST(TensorTest, FromDataCopiesInput) {
  std::array<std::int64_t, 8> shape{2, 1, 1, 1, 1, 1, 1, 1};
  float src[2] = {1.0f, 2.0f};
  Tensor t = Tensor::fromData(src, shape, DType::f32, Device::cpu);
  auto *ptr = static_cast<float *>(t.data_ptr());
  EXPECT_EQ(ptr[0], 1.0f);
  src[0] = 3.0f;
  EXPECT_NE(ptr[0], src[0]);
}

TEST(TensorTest, DataPtrAlignment) {
  std::array<std::int64_t, 8> shape{1, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(t.data_ptr());
  EXPECT_EQ(addr % 64, 0u);
}

TEST(AllocatorTest, ArenaStress) {
  orchard::runtime::CpuAllocator alloc;
  for (int i = 0; i < 100000; ++i) {
    void *p = alloc.allocate(64, "stress");
    alloc.deallocate(p);
  }
  SUCCEED();
}
