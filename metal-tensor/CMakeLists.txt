add_library(orchard_core STATIC
  metal/core/core.cpp
  metal/core/autograd/Node.cpp
  metal/core/tensor/Tensor.cpp
  metal/core/tensor/Debug.cpp
)

target_include_directories(orchard_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/metal)

target_link_libraries(orchard_core PUBLIC uuid)

add_library(orchard_metal STATIC
  metal/runtime/runtime.mm
  metal/runtime/MetalKernels.mm
)

target_include_directories(orchard_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/metal)

target_link_libraries(orchard_metal PUBLIC orchard_core)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
