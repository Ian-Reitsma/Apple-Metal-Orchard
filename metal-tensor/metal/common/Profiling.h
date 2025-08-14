#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

namespace orchard {

inline bool tensor_profile_enabled() {
  return std::getenv("ORCHARD_TENSOR_PROFILE") != nullptr;
}

inline void tensor_profile_reset() {
  // Environment is queried on every call, leaving no cached state.
}

inline void tensor_profile_clear_log() {
  std::remove("/tmp/orchard_tensor_profile.log");
}

inline void tensor_profile_log(const std::string &msg) {
  if (!tensor_profile_enabled())
    return;
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  std::ofstream ofs("/tmp/orchard_tensor_profile.log", std::ios::app);
  ofs << msg << '\n';
}

} // namespace orchard
