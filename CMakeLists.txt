cmake_minimum_required(VERSION 3.27)
project(orchard LANGUAGES C CXX OBJCXX)

# macOS arm-only build
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)

# ATen needs at least C++17; we’ll go straight to gnu++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS ON)

add_subdirectory(submodules/metal-tensor)
add_subdirectory(orchard_ops)