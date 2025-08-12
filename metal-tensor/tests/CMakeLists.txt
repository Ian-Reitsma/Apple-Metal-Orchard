if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
  add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/googletest" googletest)
  set(GTEST_LIB gtest_main)
else()
  find_package(GTest REQUIRED)
  set(GTEST_LIB GTest::gtest_main)
endif()

add_executable(metal_tensor_tests tensor_tests.cpp multi_device_tests.cpp)

target_link_libraries(metal_tensor_tests PRIVATE ${GTEST_LIB} orchard_core orchard_metal)

target_compile_features(metal_tensor_tests PRIVATE cxx_std_20)

add_test(NAME metal_tensor_tests COMMAND metal_tensor_tests)

