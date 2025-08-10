#pragma once

#include <cstddef>

namespace orchard::runtime {
void metal_add(const float *a, const float *b, float *c, std::size_t n);
}
