cmake_minimum_required(VERSION 3.27)
project(orchard LANGUAGES C CXX OBJCXX)

option(ORCHARD_BUILD_EXPERIMENTAL "Build experimental PyTorch components" OFF)

set(CMAKE_OSX_ARCHITECTURES "arm64;arm64e")
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS ON)

add_compile_options(-std=gnu++20 -Wall -Wextra -pedantic)
add_compile_options($<$<COMPILE_LANGUAGE:OBJCXX>:-fno-objc-arc>)
add_link_options(-ObjC++)

find_library(METAL Metal REQUIRED)
find_library(METALKIT MetalKit REQUIRED)
find_library(FOUNDATION Foundation REQUIRED)

add_subdirectory(metal-tensor)

if(ORCHARD_BUILD_EXPERIMENTAL)
  add_subdirectory(experimental/orchard_ops)
endif()
