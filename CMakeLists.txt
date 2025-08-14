cmake_minimum_required(VERSION 3.27)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  project(orchard LANGUAGES C CXX OBJCXX)
  find_package(Metal REQUIRED)
  link_libraries(Metal::Metal)
  set(CMAKE_OSX_ARCHITECTURES "arm64;arm64e")
  set(CMAKE_OBJCXX_STANDARD 20)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  set(CMAKE_OBJCXX_EXTENSIONS ON)
  add_compile_options($<$<COMPILE_LANGUAGE:OBJCXX>:-fno-objc-arc>)
  add_link_options(-ObjC++)
else()
  project(orchard LANGUAGES C CXX)
endif()

option(ORCHARD_BUILD_EXPERIMENTAL "Build experimental PyTorch components" OFF)
option(FETCHCONTENT_FULLY_DISCONNECTED "Disable FetchContent network access" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_compile_options(-std=gnu++20 -Wall -Wextra -pedantic)

add_subdirectory(metal-tensor)
add_subdirectory(benchmarks)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND ORCHARD_BUILD_EXPERIMENTAL)
  add_subdirectory(experimental/orchard_ops)
endif()
