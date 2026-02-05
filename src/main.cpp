#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "kernel_handlers.h"
#include "png_utils.h"

// Simple CUDA error check helper
static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " -> "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Print usage information
static void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " --input <input.png> [kernel1] [options1] [kernel2] [options2] ... --output <output.png>\n"
              << "\nKernels:\n"
              << "  greyscale   - Convert RGB to grayscale\n"
              << "  threshold   - Apply binary threshold\n"
              << "  invert      - Invert pixel values\n"
              << "  brightness  - Adjust brightness (linear scaling)\n"
              << "  gamma       - Apply gamma correction (non-linear)\n"
              << "  channel     - Extract a single color channel\n"
              << "  reorder     - Reorder RGBA channels\n"
              << "  normalize   - Histogram stretch (min-max normalization)\n"
              << "\nKernel Options:\n"
              << "  threshold --threshold <value>   - Threshold value in [0, 1] (default: 0.5)\n"
              << "  brightness --brightness <value> - Brightness scale factor (default: 1.0)\n"
              << "  gamma --gamma <value>           - Gamma correction factor (default: 1.0)\n"
              << "  channel --channel <index>       - Channel to extract (default: 0)\n"
              << "  reorder --reorder <format>      - Channel reorder format (default: RGBA)\n"
              << "  normalize                       - Takes no parameters\n"
              << "\nExamples:\n"
              << "  " << programName << " --input input.png greyscale --output output.png\n"
              << "  " << programName << " --input input.png greyscale brightness --brightness 1.5 --output output.png\n"
              << "  " << programName << " --input in.png normalize threshold --threshold 0.5 brightness --brightness 1.2 --output out.png\n";
}

// Parse command line into filter operations
static std::vector<FilterOperation> parseFilters(int argc, char* argv[], 
                                                  std::string& inputFile, std::string& outputFile) {
    std::vector<FilterOperation> filters;
    inputFile = "";
    outputFile = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        // Parse --input
        if (arg == "--input" && i + 1 < argc) {
            inputFile = argv[++i];
            continue;
        }
        
        // Parse --output
        if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
            continue;
        }
        
        // Check if this is a kernel name
        if (kernelHandlers.find(arg) != kernelHandlers.end()) {
            FilterOperation op;
            op.kernel_type = arg;
            op.param = 0.5f;  // default param
            op.extra = "RGBA";  // default extra
            
            // Check for parameters for this kernel
            if (i + 1 < argc && argv[i + 1][0] == '-') {
                std::string optName = argv[i + 1];
                
                // Parse parameter based on option name
                if ((optName == "--threshold" || optName == "--brightness" || 
                     optName == "--gamma" || optName == "--channel") && i + 2 < argc) {
                    op.param = std::stof(argv[i + 2]);
                    i += 2;
                } else if (optName == "--reorder" && i + 2 < argc) {
                    op.extra = argv[i + 2];
                    i += 2;
                }
            }
            
            filters.push_back(op);
        }
    }
    
    return filters;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    // Parse command line arguments
    std::string inputFile, outputFile;
    std::vector<FilterOperation> filters = parseFilters(argc, argv, inputFile, outputFile);
    
    // Validate input/output files
    if (inputFile.empty()) {
        std::cerr << "Error: No input file specified (use --input <file.png>)\n";
        printUsage(argv[0]);
        return 1;
    }
    
    if (outputFile.empty()) {
        std::cerr << "Error: No output file specified (use --output <file.png>)\n";
        printUsage(argv[0]);
        return 1;
    }
    
    if (filters.empty()) {
        std::cerr << "Error: No valid kernels specified\n";
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "\n=== Image Filter Pipeline ===\n";
    std::cout << "Filters: ";
    for (size_t i = 0; i < filters.size(); ++i) {
        std::cout << filters[i].kernel_type;
        if (i < filters.size() - 1) std::cout << " -> ";
    }
    std::cout << "\n\n";

    // ================================================================
    // Load PNG image
    // ================================================================
    std::vector<unsigned char> imageData;
    int width, height;
    
    if (!readPNG(inputFile, imageData, width, height)) {
        return 1;
    }

    int pixels = width * height;

    // ================================================================
    // Allocate device memory (based on actual image size)
    // ================================================================
    unsigned char* d_rgb  = nullptr;
    unsigned char* d_grey = nullptr;
    unsigned char* d_mask = nullptr;
    unsigned char* d_inverted = nullptr;
    unsigned char* d_brightened = nullptr;
    unsigned char* d_gamma_corrected = nullptr;
    unsigned char* d_reordered = nullptr;
    unsigned char* d_pipeline_a = nullptr;
    unsigned char* d_pipeline_b = nullptr;

    check(cudaMalloc(&d_rgb,  4 * pixels), "alloc d_rgb");
    check(cudaMalloc(&d_grey, pixels),     "alloc d_grey");
    check(cudaMalloc(&d_mask, pixels),     "alloc d_mask");
    check(cudaMalloc(&d_inverted, pixels), "alloc d_inverted");
    check(cudaMalloc(&d_brightened, pixels), "alloc d_brightened");
    check(cudaMalloc(&d_gamma_corrected, pixels), "alloc d_gamma_corrected");
    check(cudaMalloc(&d_reordered, 4 * pixels), "alloc d_reordered");
    check(cudaMalloc(&d_pipeline_a, 4 * pixels), "alloc d_pipeline_a");
    check(cudaMalloc(&d_pipeline_b, 4 * pixels), "alloc d_pipeline_b");

    // Prepare host-side buffers
    std::vector<unsigned char> rgb = imageData;
    std::vector<unsigned char> grey(pixels);
    std::vector<unsigned char> mask(pixels);
    std::vector<unsigned char> inverted(pixels);
    std::vector<unsigned char> brightened(pixels);
    std::vector<unsigned char> gamma_corrected(pixels);
    std::vector<unsigned char> reordered(4 * pixels);

    check(cudaMemcpy(d_rgb, rgb.data(), 4 * pixels,
                     cudaMemcpyHostToDevice),
          "copy rgba to device");


    // ----------------------------------------------------------------
    // Setup grid and block dimensions
    // ----------------------------------------------------------------
    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // ----------------------------------------------------------------
    // Setup pipeline buffers for chaining
    // ----------------------------------------------------------------
    unsigned char *current_input = d_rgb;
    unsigned char *current_output = d_pipeline_a;

    // ================================================================
    // Execute Filter Pipeline
    // ================================================================
    for (size_t filter_idx = 0; filter_idx < filters.size(); ++filter_idx) {
        const FilterOperation& op = filters[filter_idx];
        
        std::cout << "--- Executing filter " << (filter_idx + 1) << ": " << op.kernel_type << " ---\n";
        
        // Create context with current buffers
        KernelContext ctx{
            d_rgb, d_grey, d_mask, d_inverted, d_brightened, d_gamma_corrected, d_reordered,
            d_pipeline_a, d_pipeline_b,
            rgb, grey, mask, inverted, brightened, gamma_corrected, reordered,
            width, height, grid, block
        };
        
        // Launch the handler with current input/output buffers
        kernelHandlers.at(op.kernel_type)(ctx, op.param, op.extra, current_output);
        
        // Swap buffers for next filter
        // (unless this is the last filter, in which case keep output in current_output)
        if (filter_idx < filters.size() - 1) {
            std::swap(current_input, current_output);
        }
        
        std::cout << "\n";
    }

    // ================================================================
    // Copy Final Result Back to Host and Save
    // ================================================================
    // Check if the final output is single-channel or RGBA.
    // Note: this loop intentionally overwrites `is_single_channel` for each filter
    // so that the value after the loop reflects the *last* filter in the chain,
    // whose output format is the one that will be written to disk.
    bool is_single_channel = false;
    for (const auto& op : filters) {
        // These filters output single-channel greyscale
        if (op.kernel_type == "greyscale" || op.kernel_type == "threshold" || 
            op.kernel_type == "invert" || op.kernel_type == "brightness" || 
            op.kernel_type == "gamma") {
            is_single_channel = true;
        }
        // Channel extraction also produces single-channel
        else if (op.kernel_type == "channel") {
            is_single_channel = true;
        }
        // Normalize produces single-channel
        else if (op.kernel_type == "normalize") {
            is_single_channel = true;
        }
        // Reorder produces RGBA
        else if (op.kernel_type == "reorder") {
            is_single_channel = false;
        }
    }
    
    std::vector<unsigned char> final_result;
    
    if (is_single_channel) {
        // If output is single-channel, need to convert to RGBA for PNG output
        std::vector<unsigned char> grey_result(pixels);
        check(cudaMemcpy(grey_result.data(), current_output, pixels,
                         cudaMemcpyDeviceToHost),
              "copy greyscale result to host");

        // Convert greyscale to RGBA (replicate grey value to R, G, B and set A to 255)
        final_result.resize(static_cast<std::size_t>(pixels) * 4);
        unsigned char* dst = final_result.data();
        const unsigned char* src = grey_result.data();
        for (std::size_t i = 0; i < static_cast<std::size_t>(pixels); ++i) {
            const unsigned char g = src[i];
            dst[0] = g;   // R
            dst[1] = g;   // G
            dst[2] = g;   // B
            dst[3] = 255; // A
            dst += 4;
        }
    } else {
        // RGBA output - copy directly
        final_result.resize(static_cast<std::size_t>(pixels) * 4);
        check(cudaMemcpy(final_result.data(), current_output, 4 * pixels,
                         cudaMemcpyDeviceToHost),
              "copy RGBA result to host");
    }

    // ================================================================
    // Save PNG image
    // ================================================================
    if (!writePNG(outputFile, final_result, width, height)) {
        return 1;
    }

    std::cout << "✓ Output saved to: " << outputFile << "\n\n";

    // ================================================================
    // Cleanup
    // ================================================================
    cudaFree(d_rgb);
    cudaFree(d_grey);
    cudaFree(d_mask);
    cudaFree(d_inverted);
    cudaFree(d_brightened);
    cudaFree(d_gamma_corrected);
    cudaFree(d_reordered);
    cudaFree(d_pipeline_a);
    cudaFree(d_pipeline_b);

    return 0;
}
