cmake_minimum_required(VERSION 3.28)

# Explicit architecture (adjust 86 → 89/90 depending on GPU)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set C++ Standard BEFORE project() declaration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

# ------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------
project(cuda_image_kernels
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Generate compile_commands.json for VS Code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable separable compilation for multi-file kernels
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Enable testing
enable_testing()

# Find Google Test
find_package(GTest QUIET)

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
add_subdirectory(src)

# Only add tests if Google Test is found
if(GTest_FOUND)
    add_subdirectory(tests)
else()
    message(STATUS "Google Test not found - tests will be skipped")
    message(STATUS "Install with: sudo apt install libgtest-dev")
endif()


