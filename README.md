# CUDA Image Kernels

Personal toy project of a GPU image processing library with various CUDA kernels for image filtering.
Aided by AI for the implementation of various ideas.

## Features

- kernels: greyscale, threshold, invert, brightness, gamma, channel, reorder, normalize
- Filter chaining support
- GPU-accelerated
- unit tests

## Requirements

- NVIDIA GPU (Compute Capability 7.5+)
- CUDA Toolkit 12.0+
- CMake 3.28+
- GCC 11.0+

## Building

```bash
# Debug build 
mkdir -p build/debug
cd build/debug
cmake ../.. -DCMAKE_BUILD_TYPE=Debug
cmake --build .

# Relase build
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

```bash
# Single filter
./src/run_kernels greyscale

# Chained filters
./src/run_kernels greyscale brightness --brightness 1.5

# Available filters
greyscale, threshold, invert, brightness, gamma, channel, reorder, normalize
```

## Project Structure

```
src/              # CUDA kernels and handlers
tests/            # Unit tests
build/            # Build artifacts
```
