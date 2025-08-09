#pragma once

#include <array>
#include <string>

#include "TensorImpl.h"

namespace orchard::core::tensor {

class Tensor {
public:
  Tensor() = default;
  explicit Tensor(TensorImpl* impl) : impl_(impl) {}

  [[nodiscard]] static Tensor empty(const std::array<std::int64_t,8>& shape,
                                    DType dtype, Device dev);
  [[nodiscard]] static Tensor zerosLike(const Tensor& other);
  [[nodiscard]] Tensor view(const std::array<std::int64_t,8>& newShape) const;
  [[nodiscard]] Tensor slice(int dim, int start, int end, int step = 1) const;
  [[nodiscard]] Tensor to(Device dev) const;
  [[nodiscard]] Tensor contiguous() const;

  DType dtype() const { return impl_->dtype; }
  Device device() const { return impl_->device; }
  const std::array<std::int64_t,8>& shape() const { return impl_->shape; }
  const std::array<std::int64_t,8>& strides() const { return impl_->strides; }
  std::size_t nbytes() const { return impl_->storage ? impl_->storage->nbytes : 0; }
  bool is_contiguous() const;
  std::string toString() const;

private:
  TensorImpl* impl_{nullptr};
};

} // namespace orchard::core::tensor
