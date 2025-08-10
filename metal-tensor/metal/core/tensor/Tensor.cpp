#include "Tensor.h"

#include <cstring>
#include <cassert>
#include <sstream>

namespace orchard::core::tensor {

namespace {

int rank_of(const std::array<std::int64_t, 8> &shape) {
  int r = 0;
  for (auto s : shape) {
    if (s <= 0)
      break;
    ++r;
  }
  return r;
}

std::array<std::int64_t, 8>
contiguous_strides(const std::array<std::int64_t, 8> &shape) {
  std::array<std::int64_t, 8> strides{};
  int r = rank_of(shape);
  std::int64_t stride = 1;
  for (int i = r - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

std::int64_t numel(const std::array<std::int64_t, 8> &shape) {
  int r = rank_of(shape);
  std::int64_t n = 1;
  for (int i = 0; i < r; ++i)
    n *= shape[i];
  return n;
}

} // namespace

Tensor::Tensor(const Tensor &other) {
  if (other.impl_) {
    impl_ = new TensorImpl(*other.impl_);
    if (impl_->storage) {
      os_unfair_lock_lock(&other.impl_->lock);
      impl_->storage->retain();
      os_unfair_lock_unlock(&other.impl_->lock);
    }
  }
}

Tensor &Tensor::operator=(const Tensor &other) {
  if (this == &other)
    return *this;
  if (impl_) {
    if (impl_->storage) {
      os_unfair_lock_lock(&impl_->lock);
      impl_->storage->release();
      os_unfair_lock_unlock(&impl_->lock);
    }
    delete impl_;
  }
  impl_ = nullptr;
  if (other.impl_) {
    impl_ = new TensorImpl(*other.impl_);
    if (impl_->storage) {
      os_unfair_lock_lock(&other.impl_->lock);
      impl_->storage->retain();
      os_unfair_lock_unlock(&other.impl_->lock);
    }
  }
  return *this;
}

Tensor::Tensor(Tensor &&other) noexcept : impl_(other.impl_) {
  other.impl_ = nullptr;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    if (impl_ && impl_->storage) {
      os_unfair_lock_lock(&impl_->lock);
      impl_->storage->release();
      os_unfair_lock_unlock(&impl_->lock);
      delete impl_;
    }
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

Tensor::~Tensor() {
  if (impl_) {
    if (impl_->storage) {
      os_unfair_lock_lock(&impl_->lock);
      impl_->storage->release();
      os_unfair_lock_unlock(&impl_->lock);
    }
    delete impl_;
  }
}

Tensor Tensor::empty(const std::array<std::int64_t, 8> &shape, DType dtype,
                     Device dev) {
  int r = rank_of(shape);
  if (r > 8)
    return Tensor{};
  std::int64_t n = numel(shape);
  std::size_t bytes = n * dtype_size(dtype);
  Storage *storage = Storage::create(bytes, dev);
  if (!storage)
    return Tensor{};
  auto *impl = new TensorImpl{};
  impl->storage = storage;
  impl->shape = shape;
  impl->strides = contiguous_strides(shape);
  impl->dtype = dtype;
  impl->device = dev;
  impl->offset = 0;
  return Tensor(impl);
}

Tensor Tensor::zerosLike(const Tensor &other) {
  Tensor t = empty(other.shape(), other.dtype(), other.device());
  if (t.impl_ && t.impl_->storage) {
    std::memset(t.impl_->storage->data, 0, t.impl_->storage->nbytes);
  }
  return t;
}

Tensor Tensor::fromData(const void *data,
                        const std::array<std::int64_t, 8> &shape,
                        DType dtype, Device dev) {
  Tensor t = empty(shape, dtype, dev);
  if (t.impl_ && t.impl_->storage && data) {
    std::memcpy(t.impl_->storage->data, data, t.impl_->storage->nbytes);
  }
  return t;
}

Tensor Tensor::view(const std::array<std::int64_t, 8> &newShape) const {
  int r = rank_of(newShape);
  if (r > 8 || !impl_ || !impl_->storage)
    return Tensor{};
  if (numel(newShape) != numel(impl_->shape))
    return Tensor{};
  auto *impl = new TensorImpl{};
  os_unfair_lock_lock(&this->impl_->lock);
  this->impl_->storage->retain();
  os_unfair_lock_unlock(&this->impl_->lock);
  impl->storage = this->impl_->storage;
  impl->dtype = this->impl_->dtype;
  impl->device = this->impl_->device;
  impl->shape = newShape;
  impl->strides = contiguous_strides(newShape);
  impl->offset = this->impl_->offset;
  return Tensor(impl);
}

Tensor Tensor::slice(int dim, int start, int end, int step) const {
  if (!impl_ || !impl_->storage)
    return Tensor{};
  int r = rank_of(impl_->shape);
  if (dim < 0 || dim >= r || step <= 0)
    return Tensor{};
  if (start < 0 || end > impl_->shape[dim] || start >= end)
    return Tensor{};

  auto newShape = impl_->shape;
  auto newStrides = impl_->strides;
  std::int64_t len = (end - start + step - 1) / step;
  newShape[dim] = len;
  newStrides[dim] *= step;

  auto *impl = new TensorImpl{};
  os_unfair_lock_lock(&this->impl_->lock);
  this->impl_->storage->retain();
  os_unfair_lock_unlock(&this->impl_->lock);
  impl->storage = this->impl_->storage;
  impl->dtype = this->impl_->dtype;
  impl->device = this->impl_->device;
  impl->shape = newShape;
  impl->strides = newStrides;
  impl->offset = this->impl_->offset + start * this->impl_->strides[dim];
  return Tensor(impl);
}

Tensor Tensor::to(Device dev) const {
  if (!impl_ || !impl_->storage)
    return Tensor{};
  if (dev == impl_->device) {
    auto *impl = new TensorImpl{};
    os_unfair_lock_lock(&this->impl_->lock);
    this->impl_->storage->retain();
    os_unfair_lock_unlock(&this->impl_->lock);
    impl->storage = this->impl_->storage;
    impl->dtype = this->impl_->dtype;
    impl->device = this->impl_->device;
    impl->shape = this->impl_->shape;
    impl->strides = this->impl_->strides;
    impl->offset = this->impl_->offset;
    return Tensor(impl);
  }

  Tensor t = empty(impl_->shape, impl_->dtype, dev);
  if (t.impl_ && t.impl_->storage) {
    Tensor src = contiguous();
    if (src.impl_ && src.impl_->storage) {
      std::memcpy(t.impl_->storage->data, src.data_ptr(),
                  t.impl_->storage->nbytes);
    }
  }
  return t;
}

Tensor Tensor::contiguous() const {
  if (!impl_ || !impl_->storage)
    return Tensor{};
  if (is_contiguous()) {
    auto *impl = new TensorImpl{};
    os_unfair_lock_lock(&this->impl_->lock);
    this->impl_->storage->retain();
    os_unfair_lock_unlock(&this->impl_->lock);
    impl->storage = this->impl_->storage;
    impl->dtype = this->impl_->dtype;
    impl->device = this->impl_->device;
    impl->shape = this->impl_->shape;
    impl->strides = this->impl_->strides;
    impl->offset = this->impl_->offset;
    return Tensor(impl);
  }

  Tensor out = empty(impl_->shape, impl_->dtype, impl_->device);
  int r = rank_of(impl_->shape);
  std::array<std::int64_t, 8> dstStrides = contiguous_strides(impl_->shape);
  std::size_t esize = dtype_size(impl_->dtype);

  const char *src = static_cast<const char *>(impl_->storage->data);
  char *dst = static_cast<char *>(out.impl_->storage->data);
  std::array<std::int64_t, 8> idx{};
  for (std::int64_t i0 = 0; i0 < (r > 0 ? impl_->shape[0] : 1); ++i0) {
    idx[0] = i0;
    for (std::int64_t i1 = 0; i1 < (r > 1 ? impl_->shape[1] : 1); ++i1) {
      idx[1] = i1;
      for (std::int64_t i2 = 0; i2 < (r > 2 ? impl_->shape[2] : 1); ++i2) {
        idx[2] = i2;
        for (std::int64_t i3 = 0; i3 < (r > 3 ? impl_->shape[3] : 1); ++i3) {
          idx[3] = i3;
          for (std::int64_t i4 = 0; i4 < (r > 4 ? impl_->shape[4] : 1); ++i4) {
            idx[4] = i4;
            for (std::int64_t i5 = 0; i5 < (r > 5 ? impl_->shape[5] : 1);
                 ++i5) {
              idx[5] = i5;
              for (std::int64_t i6 = 0; i6 < (r > 6 ? impl_->shape[6] : 1);
                   ++i6) {
                idx[6] = i6;
                for (std::int64_t i7 = 0; i7 < (r > 7 ? impl_->shape[7] : 1);
                     ++i7) {
                  idx[7] = i7;
                  std::int64_t srcOff = impl_->offset;
                  std::int64_t dstOff = 0;
                  for (int d = 0; d < r; ++d) {
                    srcOff += idx[d] * impl_->strides[d];
                    dstOff += idx[d] * dstStrides[d];
                  }
                  std::memcpy(dst + dstOff * esize, src + srcOff * esize,
                              esize);
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

bool Tensor::is_contiguous() const {
  if (!impl_)
    return true;
  std::array<std::int64_t, 8> expected = contiguous_strides(impl_->shape);
  int r = rank_of(impl_->shape);
  for (int i = 0; i < r; ++i) {
    if (impl_->strides[i] != expected[i])
      return false;
  }
  return true;
}

Tensor Tensor::clone() const {
  if (!impl_ || !impl_->storage)
    return Tensor{};
  Tensor out = empty(impl_->shape, impl_->dtype, impl_->device);
  int r = rank_of(impl_->shape);
  std::array<std::int64_t, 8> dstStrides = contiguous_strides(impl_->shape);
  std::size_t esize = dtype_size(impl_->dtype);
  const char *src = static_cast<const char *>(impl_->storage->data);
  char *dst = static_cast<char *>(out.impl_->storage->data);
  std::array<std::int64_t, 8> idx{};
  for (std::int64_t i0 = 0; i0 < (r > 0 ? impl_->shape[0] : 1); ++i0) {
    idx[0] = i0;
    for (std::int64_t i1 = 0; i1 < (r > 1 ? impl_->shape[1] : 1); ++i1) {
      idx[1] = i1;
      for (std::int64_t i2 = 0; i2 < (r > 2 ? impl_->shape[2] : 1); ++i2) {
        idx[2] = i2;
        for (std::int64_t i3 = 0; i3 < (r > 3 ? impl_->shape[3] : 1); ++i3) {
          idx[3] = i3;
          for (std::int64_t i4 = 0; i4 < (r > 4 ? impl_->shape[4] : 1); ++i4) {
            idx[4] = i4;
            for (std::int64_t i5 = 0; i5 < (r > 5 ? impl_->shape[5] : 1);
                 ++i5) {
              idx[5] = i5;
              for (std::int64_t i6 = 0; i6 < (r > 6 ? impl_->shape[6] : 1);
                   ++i6) {
                idx[6] = i6;
                for (std::int64_t i7 = 0; i7 < (r > 7 ? impl_->shape[7] : 1);
                     ++i7) {
                  idx[7] = i7;
                  std::int64_t srcOff = impl_->offset;
                  std::int64_t dstOff = 0;
                  for (int d = 0; d < r; ++d) {
                    srcOff += idx[d] * impl_->strides[d];
                    dstOff += idx[d] * dstStrides[d];
                  }
                  std::memcpy(dst + dstOff * esize, src + srcOff * esize,
                              esize);
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

std::string Tensor::toString() const {
  if (!impl_)
    return "Tensor()";
  std::ostringstream oss;
  oss << "Tensor(dtype=" << static_cast<int>(impl_->dtype)
      << ", device=" << device_name(impl_->device) << ", shape=[";
  int r = rank_of(impl_->shape);
  for (int i = 0; i < r; ++i) {
    if (i)
      oss << ", ";
    oss << impl_->shape[i];
  }
  oss << "], strides=[";
  for (int i = 0; i < r; ++i) {
    if (i)
      oss << ", ";
    oss << impl_->strides[i];
  }
  oss << "])";
  return oss.str();
}

void Tensor::backward() const {
  assert(!"Tensor::backward() not implemented; autograd engine pending");
}

} // namespace orchard::core::tensor
