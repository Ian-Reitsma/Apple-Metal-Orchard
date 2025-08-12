#pragma once

#include <cstddef>

namespace orchard::runtime {
void metal_add(const float *a, const float *b, float *c, std::size_t n);
void metal_mul(const float *a, const float *b, float *c, std::size_t n);
void metal_matmul(const float *a, const float *b, float *c, std::size_t m,
                  std::size_t n, std::size_t k);
void metal_reduce_sum(const float *a, float *out, std::size_t n);
void metal_mul_backward_a(const float *g, const float *b, float *ga,
                          std::size_t n);
void metal_mul_backward_b(const float *g, const float *a, float *gb,
                          std::size_t n);
void metal_transpose_backward(const float *g, float *out, std::size_t m,
                              std::size_t n);
// Backward matmul kernels expect dimensions in (m, n, k) order
void metal_matmul_backward_a(const float *g, const float *b, float *ga,
                             std::size_t m, std::size_t n, std::size_t k);
void metal_matmul_backward_b(const float *g, const float *a, float *gb,
                             std::size_t m, std::size_t n, std::size_t k);
void metal_fill(float *out, float value, std::size_t n);
} // namespace orchard::runtime
