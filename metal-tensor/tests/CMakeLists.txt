include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
FetchContent_MakeAvailable(googletest)

add_executable(metal_tensor_tests tensor_tests.cpp)

target_link_libraries(metal_tensor_tests PRIVATE gtest_main orchard_core orchard_metal)

target_compile_features(metal_tensor_tests PRIVATE cxx_std_20)

add_test(NAME metal_tensor_tests COMMAND metal_tensor_tests)

