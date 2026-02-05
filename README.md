# CUDA Image Kernels

Personal toy project of a GPU image processing library with various CUDA kernels for image filtering.
Aided by AI for the implementation of various ideas.

## Features

- 8 CUDA kernels: greyscale, threshold, invert, brightness, gamma, channel, reorder, normalize
- Filter chaining support
- GPU-accelerated image processing
- PNG file I/O with automatic format conversion (RGB, RGBA, Grayscale, Palette, 16-bit)
- Modular PNG utilities (png_utils.h/cpp)
- Comprehensive testing suite with 24 test cases
- Python test framework with diverse test images (stripe, gradient, circles, noise, real-world)
- High-quality image output handling (single-channel to RGBA conversion)

## Requirements

### Build Requirements
- NVIDIA GPU (Compute Capability 7.5+)
- CUDA Toolkit 12.0+
- CMake 3.28+
- GCC 11.0+
- libpng-dev

### Testing Requirements
- Python 3.8+
- Pillow (PIL)
- NumPy

## Building

```bash
# Debug build 
mkdir -p build/debug
cd build/debug
cmake ../.. -DCMAKE_BUILD_TYPE=Debug
cmake --build .

# Release build
mkdir -p build/release
cd build/release
cmake ../.. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Testing

```bash
./tests/kernel_tests
```

## Usage

### Basic Filtering
```bash
# Read PNG, apply filter, write result
./src/run_kernels --input image.png --output output.png greyscale

# Chained filters
./src/run_kernels --input image.png --output output.png greyscale brightness --brightness 1.5

# Available filters
greyscale, threshold, invert, brightness, gamma, channel, reorder, normalize
```

### Testing
```bash
# Run unit tests
./tests/kernel_tests

# Generate test images (default 256x256) and run integration tests
cd scripts
python3 generate_test_images.py
python3 test_filters.py
```

## Project Structure

```
src/
  ├── main.cpp              # CLI entry point with PNG I/O
  ├── png_utils.h/cpp       # PNG reading/writing utilities
  ├── greyscale.cu/cuh      # Greyscale filter kernel
  ├── kernel_handlers.cpp   # Filter implementations
  └── CMakeLists.txt        # Build configuration

tests/
  ├── kernel_tests.cpp      # Unit tests (18 test cases)
  └── CMakeLists.txt        # Test build configuration

scripts/
  ├── generate_test_images.py    # NumPy-based test image generator
  ├── test_filters.py            # Automated integration test runner
  └── README.md                  # Testing documentation

build/                # Build artifacts
benchmarks/           # Benchmark utilities
```
