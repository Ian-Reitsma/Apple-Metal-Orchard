#pragma once

#include <cstddef>

namespace orchard::runtime {
void metal_add(const float *a, const float *b, float *c, std::size_t n);
void metal_mul(const float *a, const float *b, float *c, std::size_t n);
void metal_matmul(const float *a, const float *b, float *c, std::size_t m,
                  std::size_t n, std::size_t k);
void metal_reduce_sum(const float *a, float *out, std::size_t n);
} // namespace orchard::runtime
