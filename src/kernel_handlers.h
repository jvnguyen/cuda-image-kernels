#pragma once

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <cuda_runtime.h>

// ============================================================
// Kernel Execution Context
// ============================================================
/// @brief Context structure holding all data needed by kernel handlers.
/// 
/// Contains device memory pointers for various intermediate buffers, host vector references,
/// and grid/block dimensions for CUDA kernel launches. Passed to all kernel handler functions.
struct KernelContext {
    // Device memory pointers
    unsigned char* d_rgb;                ///< Input RGBA image (width*height*4 bytes)
    unsigned char* d_grey;               ///< Grayscale intermediate buffer (width*height bytes)
    unsigned char* d_mask;               ///< Mask intermediate buffer (width*height bytes)
    unsigned char* d_inverted;           ///< Inverted image buffer (width*height bytes)
    unsigned char* d_brightened;         ///< Brightness-adjusted buffer (width*height bytes)
    unsigned char* d_gamma_corrected;    ///< Gamma-corrected buffer (width*height bytes)
    unsigned char* d_reordered;          ///< Channel-reordered RGBA buffer (width*height*4 bytes)
    unsigned char* d_pipeline_a;         ///< Pipeline buffer A for chaining filters (width*height*4 bytes)
    unsigned char* d_pipeline_b;         ///< Pipeline buffer B for chaining filters (width*height*4 bytes)

    // Host vectors
    const std::vector<unsigned char>& rgb;                ///< Reference to input RGBA image data
    std::vector<unsigned char>& grey;                     ///< Reference to grayscale output
    std::vector<unsigned char>& mask;                     ///< Reference to mask output
    std::vector<unsigned char>& inverted;                 ///< Reference to inverted output
    std::vector<unsigned char>& brightened;               ///< Reference to brightness output
    std::vector<unsigned char>& gamma_corrected;          ///< Reference to gamma-corrected output
    std::vector<unsigned char>& reordered;                ///< Reference to reordered output

    // Image and grid dimensions
    int width;      ///< Image width in pixels
    int height;     ///< Image height in pixels
    dim3 grid;      ///< CUDA grid dimensions for kernel launches
    dim3 block;     ///< CUDA block dimensions for kernel launches
};

// ============================================================
// Filter Operation Struct
// ============================================================
/// @brief Represents a single filter operation in a processing pipeline.
/// 
/// Specifies which kernel to execute, along with its numeric and string parameters.
/// Multiple FilterOperation objects can be chained together to create complex processing pipelines.
struct FilterOperation {
    std::string kernel_type;  ///< Kernel type identifier: "greyscale", "threshold", "invert", "brightness", "gamma", "channel", "reorder", or "normalize"
    float param;              ///< Numeric parameter for the kernel (e.g., threshold value, brightness factor, gamma exponent, channel index)
    std::string extra;        ///< Extra string parameter (e.g., channel reorder format: "BGRA", "ARGB")
};

// ============================================================
// Kernel Handler Type Definition
// ============================================================
/// @brief Function signature for all kernel handler functions.
/// 
/// A KernelHandler is a std::function that executes a specific image processing kernel.
/// All handlers accept the execution context, numeric and string parameters, and an output buffer.
/// 
/// @param ctx The kernel execution context (contains device memory, grid/block dimensions)
/// @param float Numeric parameter specific to each kernel type
/// @param string Extra string parameter specific to each kernel type
/// @param unsigned char* Output device memory buffer for the processed result
using KernelHandler = std::function<void(const KernelContext&, float, const std::string&, unsigned char*)>;

// ============================================================
// Kernel Handler Declarations
// ============================================================

/// @brief Convert RGB image to grayscale using luminance formula.
/// 
/// Converts RGBA image to single-channel grayscale using the standard luminance formula:
/// gray = 0.299*R + 0.587*G + 0.114*B
/// 
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param param Unused parameter (ignored)
/// @param extra Unused parameter (ignored)
/// @param output Device pointer to output grayscale image (size: width*height bytes)
void handleGreyscale(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output);

/// @brief Apply binary thresholding to grayscale image.
/// 
/// First converts RGBA to grayscale, then applies binary threshold.
/// Pixels with gray value >= threshold*255 become white (255), otherwise black (0).
/// 
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param param Threshold value in range [0.0, 1.0] (default: 0.5)
/// @param extra Unused parameter (ignored)
/// @param output Device pointer to output binary image (size: width*height bytes)
void handleThreshold(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output);

/// @brief Invert pixel values of grayscale image.
/// 
/// First converts RGBA to grayscale, then inverts values: output = 255 - input.
/// 
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param param Unused parameter (ignored)
/// @param extra Unused parameter (ignored)
/// @param output Device pointer to output inverted grayscale image (size: width*height bytes)
void handleInvert(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output);

/// @brief Adjust image brightness using linear scaling.
/// 
/// Multiplies each RGBA channel by the brightness factor: output = input * factor.
/// Values are clamped to [0, 255] range.
/// 
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param param Brightness scale factor (default: 1.0, typical range: 0.5 to 2.0)
/// @param extra Unused parameter (ignored)
/// @param output Device pointer to output brightened RGBA image (size: width*height*4 bytes)
void handleBrightness(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output);

/// @brief Apply gamma correction to grayscale image.
/// 
/// First converts RGBA to grayscale, then applies gamma correction: 
/// output = 255 * (input/255)^(1/gamma)
/// 
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param param Gamma correction factor (default: 1.0, typical range: 0.5 to 3.0)
/// @param extra Unused parameter (ignored)
/// @param output Device pointer to output gamma-corrected grayscale image (size: width*height bytes)
void handleGamma(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output);

/// @brief Extract a single color channel from RGBA image.
/// 
/// Extracts one of the four RGBA channels and returns as grayscale.
/// 
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param param Channel index (0=Red, 1=Green, 2=Blue, 3=Alpha, default: 0)
/// @param extra Unused parameter (ignored)
/// @param output Device pointer to output single-channel image (size: width*height bytes)
void handleChannel(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output);

/// @brief Reorder RGBA channels to different channel format.
/// 
/// Rearranges the four RGBA channels to the specified output format.
/// Supported formats: "RGBA" (identity), "BGRA" (blue-green-red-alpha), "ARGB" (alpha-red-green-blue)
/// 
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param param Unused parameter (ignored)
/// @param extra Channel reorder format as string: "BGRA" or "ARGB" (default: "RGBA")
/// @param output Device pointer to output reordered RGBA image (size: width*height*4 bytes)
void handleReorder(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output);

/// @brief Apply histogram normalization (min-max stretch) to RGBA image.
/// 
/// Stretches the histogram of each channel to span the full [0, 255] range.
/// Finds the min and max values for each channel, then rescales: output = 255 * (input - min) / (max - min)
/// 
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param param Unused parameter (ignored)
/// @param extra Unused parameter (ignored)
/// @param output Device pointer to output normalized RGBA image (size: width*height*4 bytes)
void handleNormalize(const KernelContext& ctx, float param, const std::string& extra, unsigned char* output);

// ============================================================
// Greyscale to RGBA Conversion
// ============================================================

/// @brief Convert single-channel greyscale image to RGBA on the GPU.
///
/// Converts a single-channel greyscale image to RGBA by replicating the grey value
/// across R, G, B channels and setting alpha to full opacity (255).
///
/// @param ctx Kernel execution context containing device memory and grid/block dimensions
/// @param input Device pointer to input greyscale image (size: width*height bytes)
/// @param output Device pointer to output RGBA image (size: width*height*4 bytes)
void convertGreyscaleToRGBA(const KernelContext& ctx, unsigned char* input, unsigned char* output);

// ============================================================
// Kernel Handler Registry
// ============================================================

extern const std::map<std::string, KernelHandler> kernelHandlers;
