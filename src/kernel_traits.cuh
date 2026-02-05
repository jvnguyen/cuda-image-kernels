#pragma once
#include <cuda_runtime.h>

#include "greyscale.cuh"
#include "threshold.cuh"
#include "invert.cuh"
#include "brightness.cuh"
#include "gamma.cuh"
#include "channel.cuh"
#include "reorder.cuh"
#include "normalize.cuh"
#include "greyscale_to_rgba.cuh"

// ============================================================
// Image Format Tags
// ============================================================
// Tags to specify input and output image formats

struct GreyscaleFormat {};  // Single-channel greyscale (1 byte per pixel)
struct RGBAFormat {};       // 4-channel RGBA (4 bytes per pixel)

// ============================================================
// Kernel Type Tags
// ============================================================
// Empty tag types used for compile-time dispatch

struct GreyscaleKernelTag {};
struct ThresholdKernelTag {};
struct InvertKernelTag {};
struct BrightnessKernelTag {};
struct GammaKernelTag {};
struct ChannelKernelTag {};
struct ReorderKernelTag {};
struct NormalizeKernelTag {};
struct GreyscaleToRGBAKernelTag {};

// ============================================================
// Generic Kernel Traits
// ============================================================
// Specializations define launch behavior for each kernel type

// Default (undefined) - will cause compilation error if specialized for unknown type
template<typename KernelTag, typename InputType, typename OutputType>
struct KernelTraits;

// ============================================================
// Greyscale Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<GreyscaleKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = RGBAFormat;    // Input: RGBA
    using OutputFormat = GreyscaleFormat;  // Output: Greyscale

    // Launches greyscale kernel with no additional parameters
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block
    )
    {
        greyscaleKernel<InputType, OutputType><<<grid, block>>>(
            d_input, d_output, width, height
        );
    }
};

// ============================================================
// Threshold Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<ThresholdKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = RGBAFormat;     // Input: RGBA
    using OutputFormat = GreyscaleFormat;  // Output: Greyscale

    // Launches threshold kernel with an additional threshold parameter
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block,
        float threshold = 0.5f
    )
    {
        thresholdKernel<InputType, OutputType><<<grid, block>>>(
            d_input, d_output, threshold, width, height
        );
    }
};

// ============================================================
// Invert Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<InvertKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = RGBAFormat;     // Input: RGBA
    using OutputFormat = GreyscaleFormat;  // Output: Greyscale

    // Launches invert kernel with no additional parameters
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block
    )
    {
        invertKernel<InputType, OutputType><<<grid, block>>>(
            d_input, d_output, width, height
        );
    }
};

// ============================================================
// Brightness Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<BrightnessKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = RGBAFormat;     // Input: RGBA
    using OutputFormat = GreyscaleFormat;  // Output: Greyscale

    // Launches brightness kernel with a brightness scale factor
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block,
        float brightness = 1.0f
    )
    {
        brightnessKernel<InputType, OutputType><<<grid, block>>>(
            d_input, d_output, brightness, width, height
        );
    }
};

// ============================================================
// Gamma Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<GammaKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = RGBAFormat;     // Input: RGBA
    using OutputFormat = GreyscaleFormat;  // Output: Greyscale

    // Launches gamma kernel with a gamma correction factor
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block,
        float gamma = 1.0f
    )
    {
        gammaKernel<InputType, OutputType><<<grid, block>>>(
            d_input, d_output, gamma, width, height
        );
    }
};

// ============================================================
// Channel Extraction Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<ChannelKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = RGBAFormat;     // Input: RGBA
    using OutputFormat = GreyscaleFormat;  // Output: Greyscale

    // Launches channel extraction kernel with channel index parameter
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block,
        float channel = 0.0f  // 0=R, 1=G, 2=B, 3=A
    )
    {
        channelKernel<InputType, OutputType><<<grid, block>>>(
            d_input, d_output, static_cast<int>(channel), width, height
        );
    }
};

// ============================================================
// Channel Reorder Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<ReorderKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = RGBAFormat;     // Input: RGBA
    using OutputFormat = RGBAFormat;    // Output: RGBA

    // Launches channel reorder kernel
    // Uses 4 float parameters encoding channel indices
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block,
        float r_idx = 0.0f,
        float g_idx = 1.0f,
        float b_idx = 2.0f,
        float a_idx = 3.0f
    )
    {
        reorderKernel<InputType, OutputType><<<grid, block>>>(
            d_input, d_output,
            static_cast<int>(r_idx), static_cast<int>(g_idx),
            static_cast<int>(b_idx), static_cast<int>(a_idx),
            width, height
        );
    }
};

// ============================================================
// Normalize Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<NormalizeKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = RGBAFormat;     // Input: RGBA
    using OutputFormat = GreyscaleFormat;  // Output: Greyscale

    // Launches normalize kernel (histogram stretching)
    // Takes no parameters - normalizes based on min/max values in image
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block
    )
    {
        launchNormalizeRGBA<InputType, OutputType>(
            d_input, d_output, width, height, grid, block
        );
    }
};

// ============================================================
// Greyscale to RGBA Conversion Kernel Traits
// ============================================================
template<typename InputType, typename OutputType>
struct KernelTraits<GreyscaleToRGBAKernelTag, InputType, OutputType>
{
    using InputTypeT = InputType;
    using OutputTypeT = OutputType;
    using InputFormat = GreyscaleFormat;  // Input: Greyscale
    using OutputFormat = RGBAFormat;      // Output: RGBA

    // Launches greyscale to RGBA conversion kernel
    static void launch(
        const InputType* d_input,
        OutputType* d_output,
        int width,
        int height,
        dim3 grid,
        dim3 block
    )
    {
        launchGreyscaleToRGBAKernel<InputType, OutputType>(
            d_input, d_output, width, height, grid, block
        );
    }
};;

// ============================================================
// Generic Kernel Launcher
// ============================================================
// High-level interface that works with any kernel type

template<typename KernelTag, typename InputType, typename OutputType, typename... Args>
void launchKernel(
    const InputType* d_input,
    OutputType* d_output,
    int width,
    int height,
    dim3 grid,
    dim3 block,
    Args&&... args
)
{
    KernelTraits<KernelTag, InputType, OutputType>::launch(
        d_input, d_output, width, height, grid, block, std::forward<Args>(args)...
    );
}
