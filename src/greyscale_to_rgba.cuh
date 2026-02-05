#pragma once
#include <cuda_runtime.h>
#include "conversion_traits.cuh"

// ============================================================
// Greyscale to RGBA Conversion Kernel
// ============================================================
// Converts a single-channel greyscale image to RGBA
// Input: Single-channel greyscale (1 byte per pixel)
// Output: RGBA (4 bytes per pixel) - replicates grey to R,G,B and sets A to 255

template<typename InputType, typename OutputType>
__global__
void greyscaleToRGBAKernel(const InputType* __restrict__ input,
                           OutputType* __restrict__ output,
                           int width,
                           int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    
    // Read single greyscale value
    InputType grey = input[idx];
    
    // Convert to normalized value
    float grey_f = ConversionTraits<InputType>::toNormalized(grey);
    
    // Write RGBA output (replicate grey to R, G, B and set A to full opacity)
    int rgba_idx = 4 * idx;
    output[rgba_idx + 0] = ConversionTraits<OutputType>::clamp(grey_f);  // R
    output[rgba_idx + 1] = ConversionTraits<OutputType>::clamp(grey_f);  // G
    output[rgba_idx + 2] = ConversionTraits<OutputType>::clamp(grey_f);  // B
    output[rgba_idx + 3] = ConversionTraits<OutputType>::clamp(1.0f);    // A (full opacity)
}

// Template host launch wrapper for the greyscale to RGBA conversion kernel
template<typename InputType, typename OutputType>
void launchGreyscaleToRGBAKernel(
    const InputType* d_input,
    OutputType* d_output,
    int width,
    int height,
    dim3 grid,
    dim3 block
)
{
    greyscaleToRGBAKernel<InputType, OutputType><<<grid, block>>>(d_input, d_output, width, height);
}
