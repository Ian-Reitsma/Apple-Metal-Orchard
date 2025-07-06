cmake_minimum_required(VERSION 3.27)
project(orchard_ops LANGUAGES C CXX OBJCXX)

# macOS arm-only build
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)

# ATen needs at least C++17; we’ll go straight to 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

add_subdirectory(orchard_ops)
