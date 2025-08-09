add_library(orchard_core STATIC
  metal/core/core.cpp
)

target_include_directories(orchard_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/metal)

add_library(orchard_metal STATIC
  metal/runtime/runtime.mm
)

target_include_directories(orchard_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/metal)

target_link_libraries(orchard_metal PUBLIC orchard_core ${METAL} ${METALKIT} ${FOUNDATION})
