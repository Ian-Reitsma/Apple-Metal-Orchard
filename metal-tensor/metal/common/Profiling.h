#pragma once

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

namespace orchard {
inline std::atomic<bool> &tensor_profile_cached() {
  static std::atomic<bool> cached{false};
  return cached;
}

inline std::atomic<bool> &tensor_profile_initialized() {
  static std::atomic<bool> init{false};
  return init;
}

inline bool tensor_profile_enabled() {
  auto &init = tensor_profile_initialized();
  auto &cached = tensor_profile_cached();
  if (!init.load(std::memory_order_relaxed)) {
    bool enabled = std::getenv("ORCHARD_TENSOR_PROFILE") != nullptr;
    cached.store(enabled, std::memory_order_relaxed);
    init.store(true, std::memory_order_relaxed);
  }
  return cached.load(std::memory_order_relaxed);
}

inline void tensor_profile_reset() {
  tensor_profile_initialized().store(false, std::memory_order_relaxed);
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
