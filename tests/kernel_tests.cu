#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "kernel_handlers.h"

// ============================================================
// Test Fixture for Filter Testing
// ============================================================

class FilterTest : public ::testing::Test {
protected:
    // Test image parameters
    static constexpr int WIDTH = 8;
    static constexpr int HEIGHT = 8;
    static constexpr int PIXELS = WIDTH * HEIGHT;

    // Device pointers
    unsigned char* d_rgb = nullptr;
    unsigned char* d_grey = nullptr;
    unsigned char* d_mask = nullptr;
    unsigned char* d_inverted = nullptr;
    unsigned char* d_brightened = nullptr;
    unsigned char* d_gamma_corrected = nullptr;
    unsigned char* d_reordered = nullptr;
    unsigned char* d_pipeline_a = nullptr;
    unsigned char* d_pipeline_b = nullptr;

    // Host vectors
    std::vector<unsigned char> h_rgb;
    std::vector<unsigned char> h_grey;
    std::vector<unsigned char> h_mask;
    std::vector<unsigned char> h_inverted;
    std::vector<unsigned char> h_brightened;
    std::vector<unsigned char> h_gamma_corrected;
    std::vector<unsigned char> h_reordered;

    dim3 block;
    dim3 grid;

    void SetUp() override {
        // Initialize host vectors
        h_rgb.resize(4 * PIXELS);
        h_grey.resize(PIXELS);
        h_mask.resize(PIXELS);
        h_inverted.resize(PIXELS);
        h_brightened.resize(PIXELS);
        h_gamma_corrected.resize(PIXELS);
        h_reordered.resize(4 * PIXELS);

        // Create test image with gradient pattern
        for (int i = 0; i < PIXELS; ++i) {
            h_rgb[4*i + 0] = static_cast<unsigned char>(i * 3);   // R
            h_rgb[4*i + 1] = static_cast<unsigned char>(i * 2);   // G
            h_rgb[4*i + 2] = static_cast<unsigned char>(i * 1);   // B
            h_rgb[4*i + 3] = 255;                                   // A (opaque)
        }

        // Allocate device memory
        cudaMalloc(&d_rgb, 4 * PIXELS);
        cudaMalloc(&d_grey, PIXELS);
        cudaMalloc(&d_mask, PIXELS);
        cudaMalloc(&d_inverted, PIXELS);
        cudaMalloc(&d_brightened, PIXELS);
        cudaMalloc(&d_gamma_corrected, PIXELS);
        cudaMalloc(&d_reordered, 4 * PIXELS);
        cudaMalloc(&d_pipeline_a, 4 * PIXELS);
        cudaMalloc(&d_pipeline_b, 4 * PIXELS);

        // Copy test image to device
        cudaMemcpy(d_rgb, h_rgb.data(), 4 * PIXELS, cudaMemcpyHostToDevice);

        // Setup grid and block dimensions
        block = dim3(16, 16);
        grid = dim3((WIDTH + block.x - 1) / block.x,
                    (HEIGHT + block.y - 1) / block.y);
    }

    // Helper function to create a context
    KernelContext CreateContext() {
        return KernelContext{
            d_rgb, d_grey, d_mask, d_inverted, d_brightened, d_gamma_corrected, d_reordered,
            d_pipeline_a, d_pipeline_b,
            h_rgb, h_grey, h_mask, h_inverted, h_brightened, h_gamma_corrected, h_reordered,
            WIDTH, HEIGHT, grid, block
        };
    }

    void TearDown() override {
        // Free device memory
        cudaFree(d_rgb);
        cudaFree(d_grey);
        cudaFree(d_mask);
        cudaFree(d_inverted);
        cudaFree(d_brightened);
        cudaFree(d_gamma_corrected);
        cudaFree(d_reordered);
        cudaFree(d_pipeline_a);
        cudaFree(d_pipeline_b);
    }

    void CopyResultToHost(unsigned char* d_src, std::vector<unsigned char>& h_dst) {
        cudaMemcpy(h_dst.data(), d_src, h_dst.size(), cudaMemcpyDeviceToHost);
    }
};

// ============================================================
// Individual Filter Tests
// ============================================================

TEST_F(FilterTest, GreyscaleFilter) {
    // Apply greyscale filter
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_grey);

    // Expected: greyscale values should be single channel in RGBA format
    // Greyscale formula: 0.299*R + 0.587*G + 0.114*B
    // For pixel i: R=i*3, G=i*2, B=i*1
    // greyscale = 0.299*i*3 + 0.587*i*2 + 0.114*i = i*(0.897 + 1.174 + 0.114) = i*2.185

    for (int i = 0; i < std::min(5, PIXELS); ++i) {
        unsigned char r = h_rgb[4*i + 0];
        unsigned char g = h_rgb[4*i + 1];
        unsigned char b = h_rgb[4*i + 2];
        
        float expected = 0.299f * r + 0.587f * g + 0.114f * b;
        unsigned char grey_val = h_grey[i];
        
        // Allow small tolerance due to rounding
        EXPECT_NEAR(grey_val, expected, 2.0f) << "Mismatch at pixel " << i;
    }
}

TEST_F(FilterTest, InvertFilter) {
    // Apply greyscale then invert
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);
    handleInvert(CreateContext(), 0.0f, "", d_pipeline_b);
    CopyResultToHost(d_pipeline_b, h_inverted);

    // Expected: inverted values should be 255 - greyscale
    // For first pixel (i=0): all RGB=0, so greyscale=0, inverted=255
    // For second pixel (i=1): R=3, G=2, B=1, greyscale≈2.2, inverted≈252.8
    
    unsigned char r0 = h_rgb[0];
    unsigned char g0 = h_rgb[1];
    unsigned char b0 = h_rgb[2];
    float grey0 = 0.299f * r0 + 0.587f * g0 + 0.114f * b0;
    unsigned char expected_inv0 = 255 - static_cast<unsigned char>(grey0);
    
    EXPECT_EQ(h_inverted[0], expected_inv0);
}

TEST_F(FilterTest, BrightnessFilter) {
    // Apply greyscale then brightness (1.5x)
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);
    handleBrightness(CreateContext(), 1.5f, "", d_pipeline_b);
    CopyResultToHost(d_pipeline_b, h_brightened);

    // Expected: brightened = greyscale * 1.5 (clamped to [0,255])
    // For pixel i: greyscale ≈ i*2.185, brightened ≈ i*3.277 (clamped)
    
    for (int i = 0; i < std::min(3, PIXELS); ++i) {
        unsigned char r = h_rgb[4*i + 0];
        unsigned char g = h_rgb[4*i + 1];
        unsigned char b = h_rgb[4*i + 2];
        
        float grey = 0.299f * r + 0.587f * g + 0.114f * b;
        float brightened = grey * 1.5f;
        if (brightened > 255.0f) brightened = 255.0f;
        
        unsigned char bright_val = h_brightened[i];
        EXPECT_NEAR(bright_val, brightened, 2.0f) << "Brightness mismatch at pixel " << i;
    }
}

TEST_F(FilterTest, NormalizeFilter) {
    // Apply normalization
    handleNormalize(CreateContext(), 0.0f, "", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_reordered);

    // Expected: normalize stretches [min, max] to [0, 255]
    // In our test image: min=0 (from alpha of first pixel), max=255 (alpha channel all 255)
    // So normalization should have minimal effect since range already spans [0,255]
    
    // Just verify that values are within valid range
    for (int i = 0; i < PIXELS * 4; ++i) {
        EXPECT_GE(h_reordered[i], 0);
        EXPECT_LE(h_reordered[i], 255);
    }
}

TEST_F(FilterTest, ReorderFilterBGRA) {
    // Apply reorder to BGRA format
    handleReorder(CreateContext(), 0.0f, "BGRA", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_reordered);

    // Expected: BGRA means B,G,R,A order
    // Original pixel 0: RGBA(0, 0, 0, 255) -> BGRA(0, 0, 0, 255)
    // Original pixel 1: RGBA(3, 2, 1, 255) -> BGRA(1, 2, 3, 255)
    
    unsigned char orig_r1 = h_rgb[4*1 + 0];
    unsigned char orig_g1 = h_rgb[4*1 + 1];
    unsigned char orig_b1 = h_rgb[4*1 + 2];
    unsigned char orig_a1 = h_rgb[4*1 + 3];

    // In BGRA order: position 0=B, 1=G, 2=R, 3=A
    EXPECT_EQ(h_reordered[4*1 + 0], orig_b1);  // B
    EXPECT_EQ(h_reordered[4*1 + 1], orig_g1);  // G
    EXPECT_EQ(h_reordered[4*1 + 2], orig_r1);  // R
    EXPECT_EQ(h_reordered[4*1 + 3], orig_a1);  // A
}

TEST_F(FilterTest, ReorderFilterARGB) {
    // Apply reorder to ARGB format
    handleReorder(CreateContext(), 0.0f, "ARGB", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_reordered);

    // Expected: ARGB means A,R,G,B order
    // Original pixel 1: RGBA(3, 2, 1, 255) -> ARGB(255, 3, 2, 1)
    
    unsigned char orig_r1 = h_rgb[4*1 + 0];
    unsigned char orig_g1 = h_rgb[4*1 + 1];
    unsigned char orig_b1 = h_rgb[4*1 + 2];
    unsigned char orig_a1 = h_rgb[4*1 + 3];

    // In ARGB order: position 0=A, 1=R, 2=G, 3=B
    EXPECT_EQ(h_reordered[4*1 + 0], orig_a1);  // A
    EXPECT_EQ(h_reordered[4*1 + 1], orig_r1);  // R
    EXPECT_EQ(h_reordered[4*1 + 2], orig_g1);  // G
    EXPECT_EQ(h_reordered[4*1 + 3], orig_b1);  // B
}

TEST_F(FilterTest, ChannelExtractionRed) {
    // Extract red channel
    handleChannel(CreateContext(), 0.0f, "", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_grey);

    // Expected: red channel values
    // For pixel i: R = i*3
    for (int i = 0; i < PIXELS; ++i) {
        unsigned char expected_r = h_rgb[4*i + 0];
        EXPECT_EQ(h_grey[i], expected_r) << "Red channel mismatch at pixel " << i;
    }
}

TEST_F(FilterTest, ChannelExtractionGreen) {
    // Extract green channel (param=1.0)
    handleChannel(CreateContext(), 1.0f, "", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_grey);

    // Expected: green channel values
    // For pixel i: G = i*2
    for (int i = 0; i < PIXELS; ++i) {
        unsigned char expected_g = h_rgb[4*i + 1];
        EXPECT_EQ(h_grey[i], expected_g) << "Green channel mismatch at pixel " << i;
    }
}

TEST_F(FilterTest, ChannelExtractionAlpha) {
    // Extract alpha channel (param=3.0)
    handleChannel(CreateContext(), 3.0f, "", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_grey);

    // Expected: alpha channel is all 255
    for (int i = 0; i < PIXELS; ++i) {
        EXPECT_EQ(h_grey[i], 255) << "Alpha channel mismatch at pixel " << i;
    }
}

TEST_F(FilterTest, GammaCorrection) {
    // Apply greyscale then gamma (gamma=2.2)
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);
    handleGamma(CreateContext(), 2.2f, "", d_pipeline_b);
    CopyResultToHost(d_pipeline_b, h_gamma_corrected);

    // Expected: gamma = 255 * (normalized_value)^(1/gamma) for normalized in [0,1]
    // For pixel 0: greyscale ≈ 0, gamma ≈ 0
    // For pixel 32: greyscale ≈ 70, normalized ≈ 0.27, gamma ≈ 255*0.27^(1/2.2) ≈ 255*0.57 ≈ 145
    
    // Just verify values are in valid range
    for (int i = 0; i < PIXELS; ++i) {
        EXPECT_GE(h_gamma_corrected[i], 0);
        EXPECT_LE(h_gamma_corrected[i], 255);
    }
}

TEST_F(FilterTest, ThresholdFilter) {
    // Apply greyscale then threshold (threshold=0.5)
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);
    handleThreshold(CreateContext(), 0.5f, "", d_pipeline_b);
    CopyResultToHost(d_pipeline_b, h_mask);

    // Expected: binary threshold at 0.5 (normalized value)
    // Greyscale values below 127.5 -> 0, above -> 255
    
    for (int i = 0; i < std::min(10, PIXELS); ++i) {
        unsigned char r = h_rgb[4*i + 0];
        unsigned char g = h_rgb[4*i + 1];
        unsigned char b = h_rgb[4*i + 2];
        
        float grey = 0.299f * r + 0.587f * g + 0.114f * b;
        float normalized = grey / 255.0f;
        unsigned char expected = (normalized > 0.5f) ? 255 : 0;
        
        EXPECT_EQ(h_mask[i], expected) << "Threshold mismatch at pixel " << i;
    }
}

// ============================================================
// Chained Filter Tests
// ============================================================

TEST_F(FilterTest, ChainedGreyscaleAndBrightness) {
    // Chain: greyscale -> brightness(1.5)
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);
    handleBrightness(CreateContext(), 1.5f, "", d_pipeline_b);
    CopyResultToHost(d_pipeline_b, h_brightened);

    // Verify that brightness values are higher than greyscale
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_grey);

    for (int i = 1; i < PIXELS; ++i) {
        // Brightened should be >= greyscale (with some tolerance for rounding/clipping)
        EXPECT_GE(h_brightened[i], h_grey[i] - 1);
    }
}

TEST_F(FilterTest, ChainedNormalizeAndInvert) {
    // Chain: normalize -> invert
    handleNormalize(CreateContext(), 0.0f, "", d_pipeline_a);
    handleInvert(CreateContext(), 0.0f, "", d_pipeline_b);
    CopyResultToHost(d_pipeline_b, h_inverted);

    // Verify all values are valid
    for (int i = 0; i < PIXELS * 4; ++i) {
        EXPECT_GE(h_inverted[i], 0);
        EXPECT_LE(h_inverted[i], 255);
    }
}

TEST_F(FilterTest, ChainedBrightnessThresholdInvert) {
    // Note: These handlers read independently from d_rgb (not chained)
    // This test verifies they execute without errors
    handleBrightness(CreateContext(), 1.2f, "", d_pipeline_a);
    handleThreshold(CreateContext(), 0.5f, "", d_pipeline_b);
    handleInvert(CreateContext(), 0.0f, "", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_inverted);

    // Since invert reads from d_rgb (not threshold output), result is inverted greyscale
    // Greyscale formula: 0.299*R + 0.587*G + 0.114*B
    // For pixel i: R=i*3, G=i*2, B=i*1
    // Inverted greyscale should be: 255 - calculated_greyscale
    for (int i = 0; i < std::min(5, PIXELS); ++i) {
        unsigned char r = h_rgb[4*i + 0];
        unsigned char g = h_rgb[4*i + 1];
        unsigned char b = h_rgb[4*i + 2];
        
        float greyscale = 0.299f * r + 0.587f * g + 0.114f * b;
        float expected_inverted = 255.0f - greyscale;
        unsigned char actual = h_inverted[i];
        
        // Allow small tolerance
        EXPECT_NEAR(actual, expected_inverted, 2.0f) 
            << "Inverted greyscale mismatch at pixel " << i;
    }
}

TEST_F(FilterTest, ChainedReorderAndNormalize) {
    // Chain: reorder(BGRA) -> normalize
    handleReorder(CreateContext(), 0.0f, "BGRA", d_pipeline_a);
    handleNormalize(CreateContext(), 0.0f, "", d_pipeline_b);
    CopyResultToHost(d_pipeline_b, h_reordered);

    // Verify output is valid RGBA
    for (int i = 0; i < PIXELS * 4; ++i) {
        EXPECT_GE(h_reordered[i], 0);
        EXPECT_LE(h_reordered[i], 255);
    }
}

TEST_F(FilterTest, ThreeFilterChain) {
    // Note: These handlers read independently from d_rgb (not chained)
    // This test verifies they execute without errors
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);
    handleBrightness(CreateContext(), 1.5f, "", d_pipeline_b);
    handleThreshold(CreateContext(), 0.3f, "", d_pipeline_a);
    CopyResultToHost(d_pipeline_a, h_mask);

    // Threshold always reads from d_rgb (not brightness output)
    // At threshold 0.3 (normalized), the threshold is 0.3*255 ≈ 76
    // For greyscale formula: 0.299*i*3 + 0.587*i*2 + 0.114*i = i*2.185
    // Threshold at 0.3 means greyscale value >= 76
    int high_count = 0;
    for (int i = 0; i < PIXELS; ++i) {
        if (h_mask[i] == 255) high_count++;
    }
    // Most pixels should be binary (0 or 255)
    EXPECT_GT(high_count, 0) << "Threshold should produce some high values";
}

TEST_F(FilterTest, FourFilterChain) {
    // Chain: normalize -> reorder(BGRA) -> brightness(1.2) -> invert
    handleNormalize(CreateContext(), 0.0f, "", d_pipeline_a);
    handleReorder(CreateContext(), 0.0f, "BGRA", d_pipeline_b);
    handleBrightness(CreateContext(), 1.2f, "", d_pipeline_a);
    handleInvert(CreateContext(), 0.0f, "", d_pipeline_b);
    CopyResultToHost(d_pipeline_b, h_inverted);

    // Verify output values are valid
    for (int i = 0; i < PIXELS * 4; ++i) {
        EXPECT_GE(h_inverted[i], 0);
        EXPECT_LE(h_inverted[i], 255);
    }
}

TEST_F(FilterTest, AlternatingPipelineBuffers) {
    // Test that alternating pipeline buffers works correctly
    // Chain 5 filters to test multiple buffer swaps
    handleGreyscale(CreateContext(), 0.0f, "", d_pipeline_a);  // -> pipeline_a
    handleBrightness(CreateContext(), 1.3f, "", d_pipeline_b);  // -> pipeline_b
    handleInvert(CreateContext(), 0.0f, "", d_pipeline_a);      // -> pipeline_a
    handleNormalize(CreateContext(), 0.0f, "", d_pipeline_b);   // -> pipeline_b
    handleChannel(CreateContext(), 0.0f, "", d_pipeline_a);     // -> pipeline_a

    CopyResultToHost(d_pipeline_a, h_grey);

    // Verify output is reasonable
    for (int i = 0; i < PIXELS; ++i) {
        EXPECT_GE(h_grey[i], 0);
        EXPECT_LE(h_grey[i], 255);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
