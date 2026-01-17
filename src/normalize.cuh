#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

// ============================================================
// Normalization Kernel
// ============================================================
// Performs histogram stretching (min-max normalization)
// Maps the value range [min, max] to [0, 255]

// Reduction kernel to find min and max across entire image
template <typename InputType>
__global__ void reduceMinMaxKernel(const InputType* input,
                                    int pixels,
                                    InputType* output_min,
                                    InputType* output_max) {
    __shared__ InputType shared_min[256];
    __shared__ InputType shared_max[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize thread-local min/max
    InputType local_min = 255;
    InputType local_max = 0;

    // Load data
    if (idx < pixels) {
        InputType val = input[idx];
        local_min = val;
        local_max = val;
    }

    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        output_min[blockIdx.x] = shared_min[0];
        output_max[blockIdx.x] = shared_max[0];
    }
}

// Single block reduction to get final min/max
__global__ void finalReduceMinMaxKernel(const unsigned char* block_mins,
                                         const unsigned char* block_maxs,
                                         int num_blocks,
                                         unsigned char* d_min,
                                         unsigned char* d_max) {
    __shared__ unsigned char shared_min[256];
    __shared__ unsigned char shared_max[256];

    int tid = threadIdx.x;

    // Load block results
    unsigned char local_min = 255;
    unsigned char local_max = 0;
    if (tid < num_blocks) {
        local_min = block_mins[tid];
        local_max = block_maxs[tid];
    }

    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        d_min[0] = shared_min[0];
        d_max[0] = shared_max[0];
    }
}

template <typename InputType, typename OutputType>
__global__ void normalizeKernel(const InputType* input,
                                 OutputType* output,
                                 int width,
                                 int height,
                                 InputType min_val,
                                 InputType max_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_idx = y * width + x;
        InputType value = input[pixel_idx];

        // Avoid division by zero
        if (min_val == max_val) {
            output[pixel_idx] = (OutputType)0;
        } else {
            // Normalize to [0, 1] then scale to [0, 255]
            float normalized = (float)(value - min_val) / (float)(max_val - min_val);
            output[pixel_idx] = (OutputType)(normalized * 255.0f);
        }
    }
}

// ============================================================
// RGBA Normalization Launcher
// ============================================================
// Normalizes each channel of RGBA image independently

template <typename InputType, typename OutputType>
void launchNormalizeRGBA(const InputType* d_rgb,
                          OutputType* d_normalized,
                          int width,
                          int height,
                          const dim3& grid,
                          const dim3& block) {
    int pixels = width * height;
    int pixels_rgba = pixels * 4;

    // Allocate temporary device memory for block-level reductions
    unsigned char *d_block_mins, *d_block_maxs;
    unsigned char *d_min, *d_max;
    
    cudaMalloc(&d_block_mins, 256 * sizeof(unsigned char));
    cudaMalloc(&d_block_maxs, 256 * sizeof(unsigned char));
    cudaMalloc(&d_min, sizeof(unsigned char));
    cudaMalloc(&d_max, sizeof(unsigned char));

    // Initialize block arrays with neutral values
    cudaMemset(d_block_mins, 255, 256 * sizeof(unsigned char));
    cudaMemset(d_block_maxs, 0, 256 * sizeof(unsigned char));

    // First: reduce within each block
    int num_blocks = (pixels_rgba + 255) / 256;
    reduceMinMaxKernel<unsigned char><<<num_blocks, 256>>>(
        d_rgb, pixels_rgba, d_block_mins, d_block_maxs);

    // Second: final reduction across blocks
    finalReduceMinMaxKernel<<<1, 256>>>(
        d_block_mins, d_block_maxs, num_blocks, d_min, d_max);

    // Copy min/max back to host
    unsigned char h_min, h_max;
    cudaMemcpy(&h_min, d_min, sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Third: normalize using found min/max
    // For RGBA data, treat as 1D array: width=pixels_rgba, height=1
    int linear_blocks = (pixels_rgba + 255) / 256;
    normalizeKernel<unsigned char, unsigned char><<<linear_blocks, 256>>>(
        d_rgb, d_normalized, pixels_rgba, 1, h_min, h_max);

    // Cleanup
    cudaFree(d_block_mins);
    cudaFree(d_block_maxs);
    cudaFree(d_min);
    cudaFree(d_max);
}
