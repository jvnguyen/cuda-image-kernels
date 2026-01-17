#pragma once
#include <cuda_runtime.h>
#include "conversion_traits.cuh"

// Template device kernel: adjust brightness
// Scales pixel values by a brightness factor (1.0 = normal, >1.0 = brighter, <1.0 = darker)
template<typename InputType, typename OutputType>
__global__
void brightnessKernel(const InputType* __restrict__ input,
                      OutputType* __restrict__ output,
                      float brightness,
                      int width,
                      int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    InputType pixel = input[idx];

    // Use conversion traits to normalize input
    float normalized = ConversionTraits<InputType>::toNormalized(pixel);

    // Apply brightness scaling
    float brightened = normalized * brightness;

    // Convert back to output type using traits (clamps to valid range)
    output[idx] = ConversionTraits<OutputType>::clamp(brightened);
}

// Template host launch wrapper for the brightness kernel
template<typename InputType, typename OutputType>
void launchBrightnessKernel(
    const InputType* d_input,
    OutputType* d_output,
    float brightness,
    int width,
    int height,
    dim3 grid,
    dim3 block
)
{
    brightnessKernel<InputType, OutputType><<<grid, block>>>(d_input, d_output, brightness, width, height);
}
