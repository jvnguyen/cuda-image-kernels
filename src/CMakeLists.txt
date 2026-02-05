# ------------------------------------------------------------
# Source files for image kernels
# ------------------------------------------------------------

set(KERNEL_SOURCES
    kernel_handlers.cu
)

# Create a library from kernel handlers (shared by run_kernels and tests)
add_library(kernel_handlers_lib STATIC
    ${KERNEL_SOURCES}
)

target_include_directories(kernel_handlers_lib
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Find libpng
find_package(PNG REQUIRED)

# ------------------------------------------------------------
# Driver executable
# ------------------------------------------------------------
add_executable(run_kernels
    main.cpp
    png_utils.cpp
)

target_link_libraries(run_kernels
    PRIVATE kernel_handlers_lib PNG::PNG
)


