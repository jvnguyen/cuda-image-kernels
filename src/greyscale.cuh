#pragma once
#include <cuda_runtime.h>
#include "conversion_traits.cuh"

// Template device kernel: convert RGB → grayscale using BT.601 luminance
template<typename InputType, typename OutputType>
__global__
void greyscaleKernel(const InputType* __restrict__ rgb,
                     OutputType* __restrict__ grey,
                     int width,
                     int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int rgba_idx = 4 * idx;

    InputType r = rgb[rgba_idx + 0];
    InputType g = rgb[rgba_idx + 1];
    InputType b = rgb[rgba_idx + 2];
    // Note: rgba_idx + 3 is the alpha channel, not used for grayscale conversion

    // For vector types, use LuminanceTraits which applies BT.601
    // For scalar types, apply the formula manually after normalization
    float grey_f;
    
    if constexpr (std::is_same_v<InputType, uchar3> || std::is_same_v<InputType, uchar4> ||
                  std::is_same_v<InputType, float3> || std::is_same_v<InputType, float4>) {
        // Vector types: use LuminanceTraits which already applies BT.601
        grey_f = LuminanceTraits<InputType>::toLuminance(r);
    } else {
        // Scalar types: manually apply BT.601 formula
        float r_f = ConversionTraits<InputType>::toNormalized(r);
        float g_f = ConversionTraits<InputType>::toNormalized(g);
        float b_f = ConversionTraits<InputType>::toNormalized(b);
        grey_f = 0.299f * r_f + 0.587f * g_f + 0.114f * b_f;
    }

    grey[idx] = ConversionTraits<OutputType>::clamp(grey_f);
}

// Template host launch wrapper for the grayscale kernel
template<typename InputType, typename OutputType>
void launchGreyscaleKernel(
    const InputType* d_rgb,
    OutputType* d_grey,
    int width,
    int height,
    dim3 grid,
    dim3 block
)
{
    greyscaleKernel<InputType, OutputType><<<grid, block>>>(d_rgb, d_grey, width, height);
}