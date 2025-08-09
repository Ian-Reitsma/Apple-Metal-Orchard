#pragma once

#ifdef __APPLE__
#include <Metal/Metal.h>
#endif

namespace orchard::runtime {

class MetalContext {
public:
  MetalContext() = default;
};

} // namespace orchard::runtime
