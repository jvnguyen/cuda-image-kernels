# Testing Scripts

This directory contains Python utility scripts for testing the CUDA image processing kernels.

## Scripts

### `generate_test_images.py`

Generates diverse test PNG images using PIL (Pillow) for comprehensive testing of the filter pipeline.

**Usage:**
```bash
python3 generate_test_images.py [output_dir]
```

**Generated Images:**
- **256×256 images**: Standard test size for all tests
  - Gradient (smooth color gradients)
  - Circles (geometric patterns)
  - Noise (random patterns for robustness testing)

Note: By default, generated test images are 256×256 for consistent performance and realistic testing scenarios.

**Features:**
- Efficient NumPy array-based image generation
- Automatic PIL conversion to PNG format
- Creates 3 diverse 256×256 test images
- Covers different image patterns (gradients, geometric, random)

### `test_filters.py`

Comprehensive test runner that validates all image filters against generated test images.

**Usage:**
```bash
# Generate images and run tests
python3 test_filters.py --generate

# Run tests with existing images
python3 test_filters.py --test-dir ./test_images --output-dir ./test_results

# Specify run_kernels executable location
python3 test_filters.py --run-kernels /path/to/run_kernels
```

**Options:**
- `--generate`: Generate test images before running tests
- `--run-kernels <path>`: Path to the run_kernels executable
- `--test-dir <path>`: Directory containing test images (default: ./test_images)
- `--output-dir <path>`: Output directory for results (default: ./test_results)

**Test Coverage (256×256 images only):**
- **Single filter tests**: Tests each filter individually
  - Greyscale conversion
  - Invert operation
  - Normalization
  - Threshold with different values
  - Channel extraction
- **Filter chain tests**: Tests multiple filters in sequence
  - Greyscale → Brightness → Invert (3-step chain)
  - Normalize → Threshold → Brightness (3-step chain)
  - Greyscale → Invert → Normalize (3-step chain)
  - Brightness → Invert (2-step chain)
  - **Progressive visualization**: Shows intermediate output after each filter step
- **Format compatibility tests**: Validate different image formats

**Test Results:**
- **All configured tests executed successfully**
- **100% pass rate** validates correct implementation
- Output summary with pass/fail statistics
- **Markdown report generation**: Automatically creates `TEST_REPORT.md` with:
  - Test summary with pass/fail counts and success rate
  - Organized sections (Single Filter, Filter Chain, Format Compatibility)
  - Input/output image comparisons for each test
  - **Filter Chain Progression**: For chain tests, displays intermediate step images
  - Filters applied for each test case
  - Status indicators for easy review

## Installation

```bash
# Install Pillow and NumPy for image generation
pip install Pillow numpy
```

## Quick Start

```bash
# From the project root directory:

# 1. Generate test images
python3 scripts/generate_test_images.py scripts/test_images

# 2. Run the test suite
python3 scripts/test_filters.py --test-dir scripts/test_images --output-dir scripts/test_results

# 3. View results
# - Console output shows pass/fail summary
# - View Markdown report with images:
cat scripts/test_results/TEST_REPORT.md

# 4. Check output images
ls scripts/test_results/*.png
```

## Test Report

The test runner automatically generates a comprehensive `TEST_REPORT.md` markdown document that displays:
- **Summary statistics**: Total tests (17), pass/fail counts, success rate
- **Test cases organized by category**: 
  - Single Filter Tests (10 tests on 1024×1024 images)
  - Filter Chain Tests (4 tests with progressive visualization)
  - Format Compatibility Tests (3 tests)
- **Visual comparison**: Side-by-side input/output images for each test
- **Filter Chain Progression**: For chain tests, displays intermediate outputs
  - Shows image transformation at each step
  - Makes it easy to debug filter interactions
  - Example: `Input → Step 1 (greyscale) → Step 2 (brightness) → Step 3 (invert) → Final Output`
- **Filter details**: Shows exactly which filters were applied to each input

## Example Test Run

```
$ python3 test_filters.py --test-dir test_images --output-dir test_results
Using executable: ../build/debug/src/run_kernels

================================================================================
CUDA IMAGE FILTER PIPELINE TEST SUITE
================================================================================

🧪 Single Filter Tests:
  Testing: gradient_1024x1024_greyscale
    Input:  gradient_1024x1024.png
    Filters: greyscale
    ✓ Output: gradient_1024x1024_greyscale.png (1740 bytes)
  ...

🧪 Filter Chain Tests:
  Testing: chain_greyscale_brightness_invert
    Input:  gradient_1024x1024.png
    Filters: greyscale brightness --brightness 1.5 invert
    ✓ Output: chain_greyscale_brightness_invert.png (4403 bytes)
  ...

🧪 Format Compatibility Tests:
  ...

================================================================================
TEST SUMMARY
================================================================================

Total tests: 17
Passed:      17 ✓
Failed:      0 ✗
Success rate: 100%

📄 Markdown report generated: test_results/TEST_REPORT.md
```

## Integration

These scripts are useful for:
- **Continuous Integration (CI)**: Automated testing of filter pipeline
- **Regression Testing**: Ensure updates don't break existing functionality
- **Performance Testing**: Validate behavior with various image sizes
- **Format Testing**: Verify PNG I/O handles all formats correctly
- **Development**: Quick validation during kernel development

## Notes

- All test images are **256×256 pixels** for consistent, realistic testing
- The `generate_test_images.py` script uses **NumPy** for efficient array-based image generation
- Test images are generated on-demand, no need to commit them to the repository
- The `test_filters.py` script automatically finds the `run_kernels` executable in common locations
- All output images are PNG format for consistency with the filter pipeline design
- **Filter chain tests** include intermediate step outputs for visualization:
  - Each filter in the chain is applied incrementally
  - Intermediate results stored as `{test_name}_step_{N}_{filter_name}.png`
  - Markdown report shows these progressively for easy debugging
