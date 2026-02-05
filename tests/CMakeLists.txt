enable_testing()

# Find Google Test
find_package(GTest REQUIRED)

# Include source directory for access to headers
include_directories(${CMAKE_SOURCE_DIR}/src)

# Create the test executable
add_executable(kernel_tests
    kernel_tests.cu
)

# Set CUDA as a language for this target
set_target_properties(kernel_tests PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link against the kernel handlers library and Google Test
target_link_libraries(kernel_tests
    PRIVATE
    kernel_handlers_lib
    ${CUDA_LIBRARIES}
    GTest::gtest_main
    GTest::gtest
)

# Add tests
add_test(NAME KernelTests COMMAND kernel_tests)
