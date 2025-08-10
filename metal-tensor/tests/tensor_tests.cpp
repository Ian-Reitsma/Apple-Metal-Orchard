#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "core/tensor/Debug.h"
#include "core/tensor/Tensor.h"
#include "runtime/Allocator.h"
#include "runtime/CpuContext.h"
#include "runtime/MetalContext.h"

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

TEST(TensorTest, ViewInvalidShape) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  std::array<std::int64_t, 8> badShape{3, 2, 1, 1, 1, 1, 1, 1};
  Tensor v = t.view(badShape);
  EXPECT_EQ(v.data_ptr(), nullptr);
}

TEST(TensorTest, SliceOffsetStart) {
  std::array<std::int64_t, 8> shape{5, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *base = static_cast<float *>(t.data_ptr());
  for (int i = 0; i < 5; ++i)
    base[i] = static_cast<float>(i);
  Tensor s = t.slice(0, 2, 5);
  auto *sptr = static_cast<float *>(s.data_ptr());
  EXPECT_EQ(sptr[0], base[2]);
  EXPECT_EQ(s.offset(), 2);
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

TEST(TensorTest, FromDataZeroCopyAndDeleter) {
  std::array<std::int64_t, 8> shape{2, 1, 1, 1, 1, 1, 1, 1};
  void *raw = nullptr;
  posix_memalign(&raw, 64, 2 * sizeof(float));
  auto *src = static_cast<float *>(raw);
  src[0] = 1.0f;
  src[1] = 2.0f;
  bool freed = false;
  {
    Tensor t =
        Tensor::fromData(src, shape, DType::f32, Device::cpu, [&](void *p) {
          free(p);
          freed = true;
        });
    auto *ptr = static_cast<float *>(t.data_ptr());
    EXPECT_EQ(ptr, src);
    src[0] = 3.0f;
    EXPECT_EQ(ptr[0], 3.0f);
  }
  EXPECT_TRUE(freed);
}

TEST(TensorTest, DataPtrAlignment) {
  std::array<std::int64_t, 8> shape{1, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(t.data_ptr());
  EXPECT_EQ(addr % 64, 0u);
}

#ifdef __APPLE__
TEST(TensorTest, CpuMetalRoundtrip) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor cpu = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *ptr = static_cast<float *>(cpu.data_ptr());
  for (int i = 0; i < 4; ++i)
    ptr[i] = static_cast<float>(i + 1);
  Tensor metal = cpu.to(Device::mps);
  Tensor back = metal.to(Device::cpu);
  auto *bptr = static_cast<float *>(back.data_ptr());
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(bptr[i], ptr[i]);
}
#endif

TEST(AllocatorTest, ArenaStress) {
  orchard::runtime::CpuAllocator alloc;
  for (int i = 0; i < 100000; ++i) {
    void *p = alloc.allocate(64, "stress");
    alloc.deallocate(p);
  }
  SUCCEED();
}

TEST(TensorTest, ContiguousPreservesData) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *base = static_cast<float *>(t.data_ptr());
  for (int i = 0; i < 4; ++i)
    base[i] = static_cast<float>(i);
  Tensor s = t.slice(0, 0, 4, 2);
  EXPECT_FALSE(s.is_contiguous());
  Tensor c = s.contiguous();
  auto *ptr = static_cast<float *>(c.data_ptr());
  EXPECT_FLOAT_EQ(ptr[0], 0.0f);
  EXPECT_FLOAT_EQ(ptr[1], 2.0f);
}

TEST(RuntimeTest, CpuContextAdd) {
  float a[3] = {1.0f, 2.0f, 3.0f};
  float b[3] = {4.0f, 5.0f, 6.0f};
  float c[3] = {0.0f, 0.0f, 0.0f};
  orchard::runtime::cpu_context().add(a, b, c, 3);
  EXPECT_FLOAT_EQ(c[0], 5.0f);
  EXPECT_FLOAT_EQ(c[1], 7.0f);
  EXPECT_FLOAT_EQ(c[2], 9.0f);
}

TEST(RuntimeTest, MetalContextQueuePooling) {
  auto &ctx = orchard::runtime::metal_context();
  auto q1 = ctx.acquire_command_queue();
  ctx.return_command_queue(q1);
  auto q2 = ctx.acquire_command_queue();
  EXPECT_EQ(q1, q2);
  ctx.return_command_queue(q2);
}

TEST(TensorTest, ProfilingLogCreation) {
  std::remove("/tmp/orchard_tensor_profile.log");
  setenv("ORCHARD_TENSOR_PROFILE", "1", 1);
  std::array<std::int64_t, 8> shape{1, 1, 1, 1, 1, 1, 1, 1};
  {
    Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
    (void)t;
  }
  dump_live_tensors();
  std::ifstream ifs("/tmp/orchard_tensor_profile.log");
  EXPECT_TRUE(ifs.good());
  unsetenv("ORCHARD_TENSOR_PROFILE");
}
