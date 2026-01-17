#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "conversion_traits.cuh"

// Template device kernel: apply gamma correction
// Applies power law transformation: output = input^(1/gamma)
// gamma < 1.0 = brighter, gamma > 1.0 = darker, gamma = 1.0 = no change
template<typename InputType, typename OutputType>
__global__
void gammaKernel(const InputType* __restrict__ input,
                 OutputType* __restrict__ output,
                 float gamma,
                 int width,
                 int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    InputType pixel = input[idx];

    // Use conversion traits to normalize input to [0, 1]
    float normalized = ConversionTraits<InputType>::toNormalized(pixel);

    // Apply gamma correction: output = input^(1/gamma)
    // Clamp to avoid numerical issues at boundaries
    float gamma_corrected;
    if (normalized <= 0.0f) {
        gamma_corrected = 0.0f;
    } else if (normalized >= 1.0f) {
        gamma_corrected = 1.0f;
    } else {
        gamma_corrected = powf(normalized, 1.0f / gamma);
    }

    // Convert back to output type using traits (clamps to valid range)
    output[idx] = ConversionTraits<OutputType>::clamp(gamma_corrected);
}

// Template host launch wrapper for the gamma kernel
template<typename InputType, typename OutputType>
void launchGammaKernel(
    const InputType* d_input,
    OutputType* d_output,
    float gamma,
    int width,
    int height,
    dim3 grid,
    dim3 block
)
{
    gammaKernel<InputType, OutputType><<<grid, block>>>(d_input, d_output, gamma, width, height);
}
