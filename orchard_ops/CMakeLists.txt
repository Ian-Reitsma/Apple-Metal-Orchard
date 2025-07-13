add_library(flash_attn SHARED
    mps/flash_attn.mm
)
find_package(Torch REQUIRED)
target_include_directories(flash_attn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mps
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(flash_attn PRIVATE ${TORCH_LIBRARIES})
set_target_properties(flash_attn PROPERTIES
    PREFIX "" OUTPUT_NAME "_flash_attn"
)

# enforce the same standard on the Obj-C++ translation unit
target_compile_features(flash_attn PRIVATE cxx_std_20)
set_property(TARGET flash_attn PROPERTY OBJCXX_STANDARD 20)
find_package(Python3 COMPONENTS Development REQUIRED)
message(STATUS "PYTHON EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "PYTHON INCLUDE DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "TORCH LIB DIRS: ${TORCH_LIBRARY_DIRS}")
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS
        ${CMAKE_SOURCE_DIR}/../orchard-env/lib/python3.9/site-packages/torch/lib
        ${CMAKE_SOURCE_DIR}/orchard-env/lib/python3.9/site-packages/torch/lib
        /Users/ianreitsma/projects/orchard/orchard-env/lib/python3.9/site-packages/torch/lib
    NO_DEFAULT_PATH
    REQUIRED
)
if (NOT TORCH_PYTHON_LIBRARY) 
    message(FATAL_ERROR "Could not find libtorch_python.dylib – needed for the pybind11 type_casters") 
endif() 
message(STATUS "Linking against: ${TORCH_PYTHON_LIBRARY}")
target_link_libraries(flash_attn PRIVATE ${TORCH_PYTHON_LIBRARY})
target_include_directories(flash_attn PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(flash_attn PRIVATE ${Python3_LIBRARIES})
