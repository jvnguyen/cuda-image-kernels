#include "kernel_handlers.h"
#include "kernel_traits.cuh"
#include <iostream>
#include <cmath>

// Helper for CUDA error checking
static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " -> "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ============================================================
// Kernel Handler Implementations
// ============================================================

void handleGreyscale(const KernelContext& ctx, float /* unused param */, const std::string& /* extra */, unsigned char* output)
{
    launchKernel<GreyscaleKernelTag, unsigned char, unsigned char>(
        ctx.d_rgb, output, ctx.width, ctx.height, ctx.grid, ctx.block
    );
    check(cudaDeviceSynchronize(), "greyscale kernel sync");
}

void handleThreshold(const KernelContext& ctx, float thresholdValue, const std::string& /* extra */, unsigned char* output)
{
    // First convert to greyscale (use pipeline_a as temp)
    launchKernel<GreyscaleKernelTag, unsigned char, unsigned char>(
        ctx.d_rgb, ctx.d_pipeline_a, ctx.width, ctx.height, ctx.grid, ctx.block
    );
    check(cudaDeviceSynchronize(), "greyscale kernel sync");

    // Then apply threshold to greyscale output
    launchKernel<ThresholdKernelTag, unsigned char, unsigned char>(
        ctx.d_pipeline_a, output, ctx.width, ctx.height, ctx.grid, ctx.block, thresholdValue
    );
    check(cudaDeviceSynchronize(), "threshold kernel sync");
}

void handleInvert(const KernelContext& ctx, float /* unused param */, const std::string& /* extra */, unsigned char* output)
{
    // First convert to greyscale (use pipeline_a as temp)
    launchKernel<GreyscaleKernelTag, unsigned char, unsigned char>(
        ctx.d_rgb, ctx.d_pipeline_a, ctx.width, ctx.height, ctx.grid, ctx.block
    );
    check(cudaDeviceSynchronize(), "greyscale kernel sync");

    // Then invert the greyscale output
    launchKernel<InvertKernelTag, unsigned char, unsigned char>(
        ctx.d_pipeline_a, output, ctx.width, ctx.height, ctx.grid, ctx.block
    );
    check(cudaDeviceSynchronize(), "invert kernel sync");
}

void handleBrightness(const KernelContext& ctx, float brightnessValue, const std::string& /* extra */, unsigned char* output)
{
    // First convert to greyscale (use pipeline_a as temp)
    launchKernel<GreyscaleKernelTag, unsigned char, unsigned char>(
        ctx.d_rgb, ctx.d_pipeline_a, ctx.width, ctx.height, ctx.grid, ctx.block
    );
    check(cudaDeviceSynchronize(), "greyscale kernel sync");

    // Then apply brightness to greyscale output
    launchKernel<BrightnessKernelTag, unsigned char, unsigned char>(
        ctx.d_pipeline_a, output, ctx.width, ctx.height, ctx.grid, ctx.block, brightnessValue
    );
    check(cudaDeviceSynchronize(), "brightness kernel sync");
}

void handleGamma(const KernelContext& ctx, float gammaValue, const std::string& /* extra */, unsigned char* output)
{
    // First convert to greyscale (use pipeline_a as temp)
    launchKernel<GreyscaleKernelTag, unsigned char, unsigned char>(
        ctx.d_rgb, ctx.d_pipeline_a, ctx.width, ctx.height, ctx.grid, ctx.block
    );
    check(cudaDeviceSynchronize(), "greyscale kernel sync");

    // Then apply gamma correction to greyscale output
    launchKernel<GammaKernelTag, unsigned char, unsigned char>(
        ctx.d_pipeline_a, output, ctx.width, ctx.height, ctx.grid, ctx.block, gammaValue
    );
    check(cudaDeviceSynchronize(), "gamma kernel sync");
}

void handleChannel(const KernelContext& ctx, float channelParam, const std::string& /* extra */, unsigned char* output)
{
    // Map parameter to channel index: 0=R, 1=G, 2=B, 3=A
    int channel = static_cast<int>(channelParam);
    if (channel < 0 || channel > 3) {
        std::cerr << "Invalid channel: " << channel << ". Use 0 (R), 1 (G), 2 (B), or 3 (A)\n";
        return;
    }

    launchKernel<ChannelKernelTag, unsigned char, unsigned char>(
        ctx.d_rgb, output, ctx.width, ctx.height, ctx.grid, ctx.block, channelParam
    );
    check(cudaDeviceSynchronize(), "channel kernel sync");
}

void handleReorder(const KernelContext& ctx, float /* unused */, const std::string& order, unsigned char* output)
{
    // Parse the reorder string to get channel indices
    // Format: "BGRA", "ARGB", "RGBA" (identity), etc.
    // Maps: R->0, G->1, B->2, A->3
    // Default is RGBA (identity)
    
    std::string reorder_map = order.empty() ? "RGBA" : order;
    
    // Convert to uppercase and validate
    if (reorder_map.length() != 4) {
        std::cerr << "Invalid reorder pattern: " << reorder_map << " (expected 4 channels)\n";
        return;
    }
    
    int r_idx = -1, g_idx = -1, b_idx = -1, a_idx = -1;
    
    // Parse each character in the reorder string
    for (size_t i = 0; i < 4; ++i) {
        char ch = std::toupper(reorder_map[i]);
        switch (ch) {
            case 'R': 
                if (i == 0) r_idx = 0;
                else if (i == 1) g_idx = 0;
                else if (i == 2) b_idx = 0;
                else if (i == 3) a_idx = 0;
                break;
            case 'G':
                if (i == 0) r_idx = 1;
                else if (i == 1) g_idx = 1;
                else if (i == 2) b_idx = 1;
                else if (i == 3) a_idx = 1;
                break;
            case 'B':
                if (i == 0) r_idx = 2;
                else if (i == 1) g_idx = 2;
                else if (i == 2) b_idx = 2;
                else if (i == 3) a_idx = 2;
                break;
            case 'A':
                if (i == 0) r_idx = 3;
                else if (i == 1) g_idx = 3;
                else if (i == 2) b_idx = 3;
                else if (i == 3) a_idx = 3;
                break;
            default:
                std::cerr << "Invalid channel in reorder pattern: " << ch << "\n";
                return;
        }
    }
    
    launchKernel<ReorderKernelTag, unsigned char, unsigned char>(
        ctx.d_rgb, output, ctx.width, ctx.height, ctx.grid, ctx.block,
        static_cast<float>(r_idx), static_cast<float>(g_idx),
        static_cast<float>(b_idx), static_cast<float>(a_idx)
    );
    check(cudaDeviceSynchronize(), "reorder kernel sync");
}

// ============================================================
// Normalize Handler
// ============================================================
void handleNormalize(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output) {
    // Launch normalize kernel using generic launcher
    launchKernel<NormalizeKernelTag, unsigned char, unsigned char>(
        ctx.d_rgb,
        output,
        ctx.width,
        ctx.height,
        ctx.grid,
        ctx.block
    );
    check(cudaDeviceSynchronize(), "normalize kernel sync");
}

// ============================================================
// Kernel Handler Registry
// ============================================================

const std::map<std::string, KernelHandler> kernelHandlers = {
    {"greyscale", handleGreyscale},
    {"threshold", handleThreshold},
    {"invert", handleInvert},
    {"brightness", handleBrightness},
    {"gamma", handleGamma},
    {"channel", handleChannel},
    {"reorder", handleReorder},
    {"normalize", handleNormalize}
};
