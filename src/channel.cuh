#pragma once

#include "conversion_traits.cuh"

// ============================================================
// Channel Extraction Kernel
// ============================================================
// Extracts a single color channel (R=0, G=1, B=2) from RGB image
// and outputs as grayscale

template<typename InputType, typename OutputType>
__global__ void channelKernel(
    const InputType* __restrict__ input,
    OutputType* __restrict__ output,
    int channel,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;

    // Extract the requested channel from RGBA input
    // Each pixel in input is 4 values: R, G, B, A
    // Channel: 0=R, 1=G, 2=B, 3=A
    int rgba_idx = 4 * idx + channel;
    InputType channel_value = input[rgba_idx];

    // Convert from input type to normalized float
    float normalized = ConversionTraits<InputType>::toNormalized(channel_value);

    // Convert from normalized float to output type
    output[idx] = ConversionTraits<OutputType>::fromNormalized(normalized);
}

// ============================================================
// Channel Extraction Launcher
// ============================================================

template<typename InputType, typename OutputType>
void launchChannelKernel(
    const InputType* d_input,
    OutputType* d_output,
    int channel,
    int width,
    int height,
    dim3 grid,
    dim3 block)
{
    channelKernel<InputType, OutputType><<<grid, block>>>(
        d_input, d_output, channel, width, height
    );
}
