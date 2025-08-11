#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

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

TEST(TensorAutogradTest, AddBackward) {
  std::array<std::int64_t, 8> shape{3, 1, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(shape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(a.data_ptr());
  auto *bp = static_cast<float *>(b.data_ptr());
  for (int i = 0; i < 3; ++i) {
    ap[i] = static_cast<float>(i);
    bp[i] = static_cast<float>(i * 2);
  }
  a.set_requires_grad(true);
  b.set_requires_grad(true);
  Tensor c = a.add(b);
  c.backward();
  auto *ag = static_cast<float *>(a.grad().data_ptr());
  auto *bg = static_cast<float *>(b.grad().data_ptr());
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(ag[i], 1.0f);
    EXPECT_FLOAT_EQ(bg[i], 1.0f);
  }
}

TEST(TensorAutogradTest, MulBackward) {
  std::array<std::int64_t, 8> shape{3, 1, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(shape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(a.data_ptr());
  auto *bp = static_cast<float *>(b.data_ptr());
  for (int i = 0; i < 3; ++i) {
    ap[i] = static_cast<float>(i + 1);
    bp[i] = static_cast<float>(i + 2);
  }
  a.set_requires_grad(true);
  b.set_requires_grad(true);
  Tensor c = a.mul(b);
  c.backward();
  auto *ag = static_cast<float *>(a.grad().data_ptr());
  auto *bg = static_cast<float *>(b.grad().data_ptr());
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(ag[i], bp[i]);
    EXPECT_FLOAT_EQ(bg[i], ap[i]);
  }
}

TEST(TensorAutogradTest, MatmulBackward) {
  std::array<std::int64_t, 8> aShape{2, 3, 1, 1, 1, 1, 1, 1};
  std::array<std::int64_t, 8> bShape{3, 2, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(aShape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(bShape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(a.data_ptr());
  auto *bp = static_cast<float *>(b.data_ptr());
  for (int i = 0; i < 6; ++i)
    ap[i] = static_cast<float>(i + 1);
  for (int i = 0; i < 6; ++i)
    bp[i] = static_cast<float>(i + 1);
  a.set_requires_grad(true);
  b.set_requires_grad(true);
  Tensor c = a.matmul(b);
  c.backward();
  EXPECT_NE(a.grad().data_ptr(), nullptr);
  EXPECT_NE(b.grad().data_ptr(), nullptr);
}

TEST(TensorAutogradTest, SumMeanBackward) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *p = static_cast<float *>(t.data_ptr());
  for (int i = 0; i < 4; ++i)
    p[i] = static_cast<float>(i + 1);
  t.set_requires_grad(true);
  Tensor s = t.sum();
  s.backward();
  auto *sg = static_cast<float *>(t.grad().data_ptr());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(sg[i], 1.0f);
  Tensor t2 = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *p2 = static_cast<float *>(t2.data_ptr());
  for (int i = 0; i < 4; ++i)
    p2[i] = static_cast<float>(i + 1);
  t2.set_requires_grad(true);
  Tensor m = t2.mean();
  m.backward();
  auto *mg = static_cast<float *>(t2.grad().data_ptr());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(mg[i], 0.25f);
}

TEST(TensorAutogradTest, ViewBackward) {
  std::array<std::int64_t, 8> shape{2, 2, 1, 1, 1, 1, 1, 1};
  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *p = static_cast<float *>(t.data_ptr());
  for (int i = 0; i < 4; ++i)
    p[i] = 1.0f;
  t.set_requires_grad(true);
  std::array<std::int64_t, 8> newShape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor v = t.view(newShape);
  Tensor s = v.sum();
  s.backward();
  auto *g = static_cast<float *>(t.grad().data_ptr());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(g[i], 1.0f);
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

TEST(TensorTest, CpuMetalRoundtripNonContiguousLarge) {
  const std::int64_t N = 10000;
  std::array<std::int64_t, 8> shape{N, 1, 1, 1, 1, 1, 1, 1};
  Tensor cpu = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *base = static_cast<float *>(cpu.data_ptr());
  for (std::int64_t i = 0; i < N; ++i)
    base[i] = static_cast<float>(i);
  Tensor slice = cpu.slice(0, 0, N, 2);
  EXPECT_FALSE(slice.is_contiguous());
  Tensor metal = slice.to(Device::mps);
  Tensor back = metal.to(Device::cpu);
  auto *bptr = static_cast<float *>(back.data_ptr());
  for (std::int64_t i = 0; i < N / 2; ++i)
    EXPECT_FLOAT_EQ(bptr[i], static_cast<float>(i * 2));
}

TEST(TensorTest, AddMetalMatchesCpu) {
  std::array<std::int64_t, 8> shape{3, 1, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(shape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(a.data_ptr());
  auto *bp = static_cast<float *>(b.data_ptr());
  for (int i = 0; i < 3; ++i) {
    ap[i] = static_cast<float>(i);
    bp[i] = static_cast<float>(i * 2);
  }
  Tensor cpu = a.add(b);
  Tensor ma = a.to(Device::mps);
  Tensor mb = b.to(Device::mps);
  Tensor mc = ma.add(mb).to(Device::cpu);
  auto *cp = static_cast<float *>(cpu.data_ptr());
  auto *mp = static_cast<float *>(mc.data_ptr());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(cp[i], mp[i]);
}

TEST(TensorTest, MulMetalMatchesCpu) {
  std::array<std::int64_t, 8> shape{3, 1, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(shape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(a.data_ptr());
  auto *bp = static_cast<float *>(b.data_ptr());
  for (int i = 0; i < 3; ++i) {
    ap[i] = static_cast<float>(i + 1);
    bp[i] = static_cast<float>(i + 2);
  }
  Tensor cpu = a.mul(b);
  Tensor ma = a.to(Device::mps);
  Tensor mb = b.to(Device::mps);
  Tensor mc = ma.mul(mb).to(Device::cpu);
  auto *cp = static_cast<float *>(cpu.data_ptr());
  auto *mp = static_cast<float *>(mc.data_ptr());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(cp[i], mp[i]);
}

TEST(TensorTest, MatmulMetalMatchesCpu) {
  std::array<std::int64_t, 8> aShape{2, 3, 1, 1, 1, 1, 1, 1};
  std::array<std::int64_t, 8> bShape{3, 2, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(aShape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(bShape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(a.data_ptr());
  auto *bp = static_cast<float *>(b.data_ptr());
  for (int i = 0; i < 6; ++i) {
    ap[i] = static_cast<float>(i + 1);
    bp[i] = static_cast<float>(i + 1);
  }
  Tensor cpu = a.matmul(b);
  Tensor ma = a.to(Device::mps);
  Tensor mb = b.to(Device::mps);
  Tensor mc = ma.matmul(mb).to(Device::cpu);
  auto *cp = static_cast<float *>(cpu.data_ptr());
  auto *mp = static_cast<float *>(mc.data_ptr());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(cp[i], mp[i]);
}

TEST(TensorTest, SumMetalMatchesCpu) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(a.data_ptr());
  for (int i = 0; i < 4; ++i)
    ap[i] = static_cast<float>(i + 1);
  Tensor cpu = a.sum();
  Tensor ma = a.to(Device::mps);
  Tensor mb = ma.sum().to(Device::cpu);
  auto *cp = static_cast<float *>(cpu.data_ptr());
  auto *mp = static_cast<float *>(mb.data_ptr());
  EXPECT_FLOAT_EQ(cp[0], mp[0]);
}

TEST(TensorTest, AutogradAddMetal) {
  std::array<std::int64_t, 8> shape{3, 1, 1, 1, 1, 1, 1, 1};
  Tensor a = Tensor::empty(shape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(a.data_ptr());
  auto *bp = static_cast<float *>(b.data_ptr());
  for (int i = 0; i < 3; ++i) {
    ap[i] = static_cast<float>(i);
    bp[i] = static_cast<float>(i * 2);
  }
  a.set_requires_grad(true);
  b.set_requires_grad(true);
  Tensor ma = a.to(Device::mps);
  Tensor mb = b.to(Device::mps);
  Tensor c = ma.add(mb);
  c.backward();
  Tensor ag = ma.grad().to(Device::cpu);
  Tensor bg = mb.grad().to(Device::cpu);
  auto *agp = static_cast<float *>(ag.data_ptr());
  auto *bgp = static_cast<float *>(bg.data_ptr());
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(agp[i], 1.0f);
    EXPECT_FLOAT_EQ(bgp[i], 1.0f);
  }
}

TEST(TensorTest, AutogradMatmulMetal) {
  std::array<std::int64_t, 8> aShape{2, 3, 1, 1, 1, 1, 1, 1};
  std::array<std::int64_t, 8> bShape{3, 2, 1, 1, 1, 1, 1, 1};
  Tensor ac = Tensor::empty(aShape, DType::f32, Device::cpu);
  Tensor bc = Tensor::empty(bShape, DType::f32, Device::cpu);
  auto *ap = static_cast<float *>(ac.data_ptr());
  auto *bp = static_cast<float *>(bc.data_ptr());
  for (int i = 0; i < 6; ++i) {
    ap[i] = static_cast<float>(i + 1);
    bp[i] = static_cast<float>(i + 1);
  }
  ac.set_requires_grad(true);
  bc.set_requires_grad(true);
  Tensor cc = ac.matmul(bc);
  cc.backward();
  Tensor ag_exp = ac.grad();
  Tensor bg_exp = bc.grad();

  Tensor a = Tensor::empty(aShape, DType::f32, Device::cpu);
  Tensor b = Tensor::empty(bShape, DType::f32, Device::cpu);
  std::memcpy(a.data_ptr(), ac.data_ptr(), 6 * sizeof(float));
  std::memcpy(b.data_ptr(), bc.data_ptr(), 6 * sizeof(float));
  a.set_requires_grad(true);
  b.set_requires_grad(true);
  Tensor ma = a.to(Device::mps);
  Tensor mb = b.to(Device::mps);
  Tensor c = ma.matmul(mb);
  c.backward();
  Tensor ag = ma.grad().to(Device::cpu);
  Tensor bg = mb.grad().to(Device::cpu);
  auto *agp = static_cast<float *>(ag.data_ptr());
  auto *bgp = static_cast<float *>(bg.data_ptr());
  auto *ag_exp_p = static_cast<float *>(ag_exp.data_ptr());
  auto *bg_exp_p = static_cast<float *>(bg_exp.data_ptr());
  for (int i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(agp[i], ag_exp_p[i]);
  for (int i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(bgp[i], bg_exp_p[i]);
}

TEST(TensorTest, AutogradSumMetal) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor tc = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *cp = static_cast<float *>(tc.data_ptr());
  for (int i = 0; i < 4; ++i)
    cp[i] = static_cast<float>(i + 1);
  tc.set_requires_grad(true);
  Tensor sc = tc.sum();
  sc.backward();
  Tensor g_exp = tc.grad();

  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  std::memcpy(t.data_ptr(), tc.data_ptr(), 4 * sizeof(float));
  t.set_requires_grad(true);
  Tensor m = t.to(Device::mps);
  Tensor s = m.sum();
  s.backward();
  Tensor g = m.grad().to(Device::cpu);
  auto *gp = static_cast<float *>(g.data_ptr());
  auto *exp_p = static_cast<float *>(g_exp.data_ptr());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(gp[i], exp_p[i]);
}

TEST(TensorTest, AutogradMeanMetal) {
  std::array<std::int64_t, 8> shape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor tc = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *cp = static_cast<float *>(tc.data_ptr());
  for (int i = 0; i < 4; ++i)
    cp[i] = static_cast<float>(i + 1);
  tc.set_requires_grad(true);
  Tensor sc = tc.mean();
  sc.backward();
  Tensor g_exp = tc.grad();

  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  std::memcpy(t.data_ptr(), tc.data_ptr(), 4 * sizeof(float));
  t.set_requires_grad(true);
  Tensor m = t.to(Device::mps);
  Tensor s = m.mean();
  s.backward();
  Tensor g = m.grad().to(Device::cpu);
  auto *gp = static_cast<float *>(g.data_ptr());
  auto *exp_p = static_cast<float *>(g_exp.data_ptr());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(gp[i], exp_p[i]);
}

TEST(TensorTest, AutogradViewMetal) {
  std::array<std::int64_t, 8> shape{2, 2, 1, 1, 1, 1, 1, 1};
  Tensor tc = Tensor::empty(shape, DType::f32, Device::cpu);
  auto *cp = static_cast<float *>(tc.data_ptr());
  for (int i = 0; i < 4; ++i)
    cp[i] = 1.0f;
  tc.set_requires_grad(true);
  std::array<std::int64_t, 8> newShape{4, 1, 1, 1, 1, 1, 1, 1};
  Tensor vc = tc.view(newShape);
  Tensor sc = vc.sum();
  sc.backward();
  Tensor g_exp = tc.grad();

  Tensor t = Tensor::empty(shape, DType::f32, Device::cpu);
  std::memcpy(t.data_ptr(), tc.data_ptr(), 4 * sizeof(float));
  t.set_requires_grad(true);
  Tensor m = t.to(Device::mps);
  Tensor v = m.view(newShape);
  Tensor s = v.sum();
  s.backward();
  Tensor g = m.grad().to(Device::cpu);
  auto *gp = static_cast<float *>(g.data_ptr());
  auto *exp_p = static_cast<float *>(g_exp.data_ptr());
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(gp[i], exp_p[i]);
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

TEST(TensorTest, ProfilingLogEntries) {
  std::remove("/tmp/orchard_tensor_profile.log");
  setenv("ORCHARD_TENSOR_PROFILE", "1", 1);
  std::array<std::int64_t, 8> shape{1, 1, 1, 1, 1, 1, 1, 1};
  {
    Tensor a = Tensor::empty(shape, DType::f32, Device::cpu);
    Tensor b = Tensor::empty(shape, DType::f32, Device::cpu);
    dump_live_tensors();
    (void)a;
    (void)b;
  }
  dump_live_tensors();
  unsetenv("ORCHARD_TENSOR_PROFILE");

  std::ifstream ifs("/tmp/orchard_tensor_profile.log");
  ASSERT_TRUE(ifs.good());
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  std::string contents = buffer.str();
  EXPECT_NE(contents.find("alloc"), std::string::npos);
  EXPECT_NE(contents.find("free"), std::string::npos);
  EXPECT_NE(contents.find("live"), std::string::npos);
}
