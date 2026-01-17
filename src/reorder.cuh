#pragma once

#include "conversion_traits.cuh"

// ============================================================
// Channel Reorder Kernel
// ============================================================
// Reorders RGBA channels according to a specified permutation
// Input: RGBA (4 channels)
// Output: RGBA with channels reordered (still 4 channels)
// Parameters: r_idx, g_idx, b_idx, a_idx specify which input channel
//            goes to each output position (0=R, 1=G, 2=B, 3=A)

template<typename InputType, typename OutputType>
__global__ void reorderKernel(
    const InputType* __restrict__ input,
    OutputType* __restrict__ output,
    int r_idx, int g_idx, int b_idx, int a_idx,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;

    // Read RGBA channels from input
    // Each pixel in input is 4 values: R, G, B, A
    int rgba_idx = 4 * idx;
    InputType channels[4];
    channels[0] = input[rgba_idx + 0];  // R
    channels[1] = input[rgba_idx + 1];  // G
    channels[2] = input[rgba_idx + 2];  // B
    channels[3] = input[rgba_idx + 3];  // A

    // Reorder: output[out_ch] = input[channels[in_ch]]
    int out_rgba_idx = 4 * idx;
    output[out_rgba_idx + 0] = channels[r_idx];
    output[out_rgba_idx + 1] = channels[g_idx];
    output[out_rgba_idx + 2] = channels[b_idx];
    output[out_rgba_idx + 3] = channels[a_idx];
}

// ============================================================
// Channel Reorder Launcher
// ============================================================

template<typename InputType, typename OutputType>
void launchReorderKernel(
    const InputType* d_input,
    OutputType* d_output,
    int r_idx, int g_idx, int b_idx, int a_idx,
    int width,
    int height,
    dim3 grid,
    dim3 block)
{
    reorderKernel<InputType, OutputType><<<grid, block>>>(
        d_input, d_output, r_idx, g_idx, b_idx, a_idx, width, height
    );
}
