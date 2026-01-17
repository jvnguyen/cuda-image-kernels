#pragma once
#include <cuda_runtime.h>
#include "conversion_traits.cuh"

// Template device kernel: apply threshold to create binary mask
// Converts input to normalized [0, 1] range, applies threshold, returns binary output
template<typename InputType, typename OutputType>
__global__
void thresholdKernel(const InputType* __restrict__ input,
                     OutputType* __restrict__ output,
                     float threshold,
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

    // Apply threshold
    float result = (normalized > threshold) ? 1.0f : 0.0f;

    // Convert back to output type using traits
    output[idx] = ConversionTraits<OutputType>::clamp(result);
}

// Template host launch wrapper for the threshold kernel
template<typename InputType, typename OutputType>
void launchThresholdKernel(
    const InputType* d_input,
    OutputType* d_output,
    float threshold,
    int width,
    int height,
    dim3 grid,
    dim3 block
)
{
    thresholdKernel<InputType, OutputType><<<grid, block>>>(d_input, d_output, threshold, width, height);
}
