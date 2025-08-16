#pragma once

#include <cstddef>

namespace orchard::runtime {

void metal_copy_buffers(void *dstBuf, const void *srcBuf, std::size_t bytes);
void metal_copy_cpu_to_metal(void *dstBuf, const void *src, std::size_t bytes);
void metal_copy_metal_to_cpu(void *dst, const void *srcBuf, std::size_t bytes);

} // namespace orchard::runtime
