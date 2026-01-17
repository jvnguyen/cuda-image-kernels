#pragma once
#include <cuda_runtime.h>
#include "conversion_traits.cuh"

// Template device kernel: invert pixel values
// Converts input to normalized [0, 1] range, inverts (1 - value), then converts back
template<typename InputType, typename OutputType>
__global__
void invertKernel(const InputType* __restrict__ input,
                  OutputType* __restrict__ output,
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

    // Invert: 1.0 - value
    float inverted = 1.0f - normalized;

    // Convert back to output type using traits
    output[idx] = ConversionTraits<OutputType>::clamp(inverted);
}

// Template host launch wrapper for the invert kernel
template<typename InputType, typename OutputType>
void launchInvertKernel(
    const InputType* d_input,
    OutputType* d_output,
    int width,
    int height,
    dim3 grid,
    dim3 block
)
{
    invertKernel<InputType, OutputType><<<grid, block>>>(d_input, d_output, width, height);
}
