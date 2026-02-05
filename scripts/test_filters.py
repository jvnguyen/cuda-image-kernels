#!/usr/bin/env python3
"""
Test runner for CUDA image processing kernels.

This script runs the main.cpp executable against generated test images
to verify the filter pipeline works correctly.

Usage:
    python3 test_filters.py [--generate] [--run-kernels <path>]

Options:
    --generate              Generate test images before running tests
    --run-kernels <path>    Path to run_kernels executable (default: ../build/debug/src/run_kernels)
    --test-dir <path>       Directory with test images (default: ./test_images)
    --output-dir <path>     Output directory for results (default: ./test_results)
"""

import sys
import os
import subprocess
from pathlib import Path


IMAGE_SIZE = 256


class TestRunner:
    """Run tests against the filter pipeline."""
    
    def __init__(self, run_kernels_path, test_dir, output_dir):
        self.run_kernels = run_kernels_path
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.results = []
        self.test_details = []  # Store input/output pairs for markdown report
    
    def ensure_dir(self, path):
        """Create directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
    
    def run_test(self, input_image, filters, test_name):
        """Run a single filter test."""
        output_image = self.output_dir / f"{test_name}.png"
        
        # Build command
        cmd = [
            str(self.run_kernels),
            "--input", str(input_image),
            *filters,
            "--output", str(output_image)
        ]
        
        print(f"  Testing: {test_name}")
        print(f"    Input:  {input_image.name}")
        print(f"    Filters: {' '.join(filters[:-2])}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and output_image.exists():
                file_size = output_image.stat().st_size
                print(f"    ✓ Output: {output_image.name} ({file_size} bytes)\n")
                self.results.append((test_name, True, None))
                # Store for markdown report
                self.test_details.append({
                    'name': test_name,
                    'input': input_image,
                    'output': output_image,
                    'filters': ' '.join(filters[:-2]),
                    'success': True
                })
                return True
            else:
                error = result.stderr or result.stdout
                print(f"    ✗ Failed: {error[:100]}\n")
                self.results.append((test_name, False, error))
                self.test_details.append({
                    'name': test_name,
                    'input': input_image,
                    'output': None,
                    'filters': ' '.join(filters[:-2]),
                    'success': False,
                    'error': error[:200]
                })
                return False
        except subprocess.TimeoutExpired:
            print(f"    ✗ Timeout\n")
            self.results.append((test_name, False, "Timeout"))
            self.test_details.append({
                'name': test_name,
                'input': input_image,
                'output': None,
                'filters': ' '.join(filters[:-2]),
                'success': False,
                'error': 'Timeout'
            })
            return False
        except Exception as e:
            print(f"    ✗ Error: {str(e)}\n")
            self.results.append((test_name, False, str(e)))
            self.test_details.append({
                'name': test_name,
                'input': input_image,
                'output': None,
                'filters': ' '.join(filters[:-2]),
                'success': False,
                'error': str(e)[:200]
            })
            return False
    
    def run_chain_test(self, input_image, filters, test_name):
        """Run a chain test and generate intermediate outputs for visualization."""
        # Parse filters into individual filter commands (accounting for parameters)
        filter_steps = []
        i = 0
        while i < len(filters) - 2:  # -2 to skip the final "--output" and "dummy.png"
            if filters[i].startswith("--"):
                i += 1
            else:
                # This is a filter name
                filter_name = filters[i]
                step_filters = [filter_name]
                i += 1
                # Collect any parameters for this filter
                while i < len(filters) - 2 and filters[i].startswith("--"):
                    step_filters.append(filters[i])
                    step_filters.append(filters[i + 1])
                    i += 2
                filter_steps.append(step_filters)
        
        # Generate intermediate outputs
        intermediate_outputs = []
        
        for step_num, step_filter in enumerate(filter_steps, 1):
            # Build cumulative filters (all filters up to this step)
            cumulative_filters = []
            for step in filter_steps[:step_num]:
                cumulative_filters.extend(step)
            
            step_output = self.output_dir / f"{test_name}_step_{step_num}_{step_filter[0]}.png"
            
            cmd = [
                str(self.run_kernels),
                "--input", str(input_image),
                *cumulative_filters,
                "--output", str(step_output)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and step_output.exists():
                    intermediate_outputs.append({
                        'step': step_num,
                        'filters': ' '.join(step_filter),
                        'output': step_output
                    })
            except Exception:
                pass
        
        # Now run the complete chain test
        output_image = self.output_dir / f"{test_name}.png"
        
        cmd = [
            str(self.run_kernels),
            "--input", str(input_image),
            *filters,
            "--output", str(output_image)
        ]
        
        print(f"  Testing: {test_name}")
        print(f"    Input:  {input_image.name}")
        print(f"    Filters: {' '.join(filters[:-2])}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and output_image.exists():
                file_size = output_image.stat().st_size
                print(f"    ✓ Output: {output_image.name} ({file_size} bytes)\n")
                self.results.append((test_name, True, None))
                # Store for markdown report with intermediate steps
                self.test_details.append({
                    'name': test_name,
                    'input': input_image,
                    'output': output_image,
                    'filters': ' '.join(filters[:-2]),
                    'success': True,
                    'is_chain': True,
                    'intermediate_steps': intermediate_outputs
                })
                return True
            else:
                error = result.stderr or result.stdout
                print(f"    ✗ Failed: {error[:100]}\n")
                self.results.append((test_name, False, error))
                self.test_details.append({
                    'name': test_name,
                    'input': input_image,
                    'output': None,
                    'filters': ' '.join(filters[:-2]),
                    'success': False,
                    'is_chain': True,
                    'error': error[:200]
                })
                return False
        except subprocess.TimeoutExpired:
            print(f"    ✗ Timeout\n")
            self.results.append((test_name, False, "Timeout"))
            self.test_details.append({
                'name': test_name,
                'input': input_image,
                'output': None,
                'filters': ' '.join(filters[:-2]),
                'success': False,
                'is_chain': True,
                'error': 'Timeout'
            })
            return False
        except Exception as e:
            print(f"    ✗ Error: {str(e)}\n")
            self.results.append((test_name, False, str(e)))
            self.test_details.append({
                'name': test_name,
                'input': input_image,
                'output': None,
                'filters': ' '.join(filters[:-2]),
                'success': False,
                'is_chain': True,
                'error': str(e)[:200]
            })
            return False
    
    def run_all_tests(self):
        """Run all test cases."""
        print("\n" + "="*80)
        print("CUDA IMAGE FILTER PIPELINE TEST SUITE")
        print("="*80 + "\n")
        
        self.ensure_dir(self.output_dir)
        
        # Basic single-filter tests (1024x1024 only)
        print("🧪 Single Filter Tests:")
        print("-" * 80)
        
        test_images = {
            f"stripe_{IMAGE_SIZE}x{IMAGE_SIZE}.png": [
                ("greyscale", ["greyscale"]),
                ("invert", ["invert"]),
                ("normalize", ["normalize"]),
                ("threshold_0.3", ["threshold", "--threshold", "0.3"]),
                ("threshold_0.7", ["threshold", "--threshold", "0.7"]),
            ],
            f"gradient_{IMAGE_SIZE}x{IMAGE_SIZE}.png": [
                ("greyscale", ["greyscale"]),
                ("invert", ["invert"]),
                ("normalize", ["normalize"]),
                ("threshold_0.3", ["threshold", "--threshold", "0.3"]),
                ("threshold_0.7", ["threshold", "--threshold", "0.7"]),
            ],
            f"circles_{IMAGE_SIZE}x{IMAGE_SIZE}.png": [
                ("greyscale", ["greyscale"]),
                ("normalize", ["normalize"]),
            ],
            f"noise_{IMAGE_SIZE}x{IMAGE_SIZE}.png": [
                ("channel_0", ["channel", "--channel", "0"]),
                ("channel_1", ["channel", "--channel", "1"]),
                ("channel_2", ["channel", "--channel", "2"]),
            ],
        }
        
        test_count = 0
        for image_name, tests in test_images.items():
            image_path = self.test_dir / image_name
            if image_path.exists():
                for test_name, filters in tests:
                    full_name = f"{image_name.replace('.png', '')}_{test_name}"
                    self.run_test(image_path, filters + ["--output", "dummy.png"], full_name)
                    test_count += 1
        
        # Filter chain tests (1024x1024 only)
        print("🧪 Filter Chain Tests:")
        print("-" * 80)
        
        chain_tests = [
            (f"stripe_{IMAGE_SIZE}x{IMAGE_SIZE}.png", 
             ["greyscale", "brightness", "--brightness", "1.5", "invert"],
             "chain_greyscale_brightness_invert"),
            
            (f"gradient_{IMAGE_SIZE}x{IMAGE_SIZE}.png",
             ["normalize", "threshold", "--threshold", "0.5", "brightness", "--brightness", "1.2"],
             "chain_normalize_threshold_brightness"),
            
            (f"circles_{IMAGE_SIZE}x{IMAGE_SIZE}.png",
             ["greyscale", "invert", "normalize"],
             "chain_greyscale_invert_normalize"),
            
            (f"noise_{IMAGE_SIZE}x{IMAGE_SIZE}.png",
             ["brightness", "--brightness", "0.8", "invert"],
             "chain_brightness_invert"),
        ]
        
        for image_name, filters, test_name in chain_tests:
            image_path = self.test_dir / image_name
            if image_path.exists():
                self.run_chain_test(image_path, filters + ["--output", "dummy.png"], test_name)
                test_count += 1
        
        # Format tests (1024x1024 only)
        print("🧪 Format Compatibility Tests:")
        print("-" * 80)
        
        format_tests = [
            f"stripe_{IMAGE_SIZE}x{IMAGE_SIZE}.png",
            f"gradient_{IMAGE_SIZE}x{IMAGE_SIZE}.png",
            f"circles_{IMAGE_SIZE}x{IMAGE_SIZE}.png",
            f"noise_{IMAGE_SIZE}x{IMAGE_SIZE}.png",
            f"real_world_{IMAGE_SIZE}x{IMAGE_SIZE}.png",
        ]
        
        for image_name in format_tests:
            image_path = self.test_dir / image_name
            if image_path.exists():
                test_name = f"format_{image_name.replace('.png', '')}"
                self.run_test(image_path, ["greyscale", "--output", "dummy.png"], test_name)
                test_count += 1
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80 + "\n")
        
        passed = sum(1 for _, success, _ in self.results if success)
        failed = sum(1 for _, success, _ in self.results if not success)
        
        print(f"Total tests: {len(self.results)}")
        print(f"Passed:      {passed} ✓")
        print(f"Failed:      {failed} ✗")
        print(f"Success rate: {100 * passed // len(self.results)}%\n")
        
        if failed > 0:
            print("Failed tests:")
            for name, success, error in self.results:
                if not success:
                    print(f"  - {name}")
                    if error:
                        print(f"    {error[:60]}")
        
        # Generate markdown report
        self.generate_markdown_report()
        
        print()
        return failed == 0
    
    def generate_markdown_report(self):
        """Generate a markdown document with test results and images."""
        report_path = self.output_dir / "TEST_REPORT.md"
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# CUDA Image Filter Test Report\n\n")
            f.write(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            passed = sum(1 for d in self.test_details if d['success'])
            total = len(self.test_details)
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Tests:** {total}\n")
            f.write(f"- **Passed:** {passed} ✓\n")
            f.write(f"- **Failed:** {total - passed} ✗\n")
            f.write(f"- **Success Rate:** {100 * passed // total if total > 0 else 0}%\n\n")
            
            # Group tests by category
            categories = self._group_tests_by_category()
            
            for category, tests in categories.items():
                f.write(f"## {category}\n\n")
                
                for test in tests:
                    status_emoji = "✓" if test['success'] else "✗"
                    f.write(f"### {status_emoji} {test['name']}\n\n")
                    f.write(f"**Filters:** `{test['filters']}`\n\n")
                    
                    if test['success']:
                        # Get relative paths for markdown
                        # Input images are in test_dir, output images are in output_dir
                        input_rel = f"../{self.test_dir.name}/{Path(test['input']).name}"
                        output_rel = Path(test['output']).name
                        
                        # Check if this is a chain test with intermediate steps
                        if test.get('is_chain') and test.get('intermediate_steps'):
                            f.write("#### Filter Progression\n\n")
                            f.write("| Step | Filters Applied | Result |\n")
                            f.write("|------|-----------------|--------|\n")
                            f.write(f"| 0 | (input) | ![Input]({input_rel}) |\n")
                            
                            for step_info in test['intermediate_steps']:
                                step_num = step_info['step']
                                step_filters = step_info['filters']
                                step_output_rel = Path(step_info['output']).name
                                f.write(f"| {step_num} | {step_filters} | ![Step {step_num}]({step_output_rel}) |\n")
                            
                            f.write("\n")
                        else:
                            # Regular test with just input and output
                            f.write("| Input | Output |\n")
                            f.write("|-------|--------|\n")
                            f.write(f"| ![Input]({input_rel}) | ![Output]({output_rel}) |\n\n")
                    else:
                        f.write(f"**Status:** ✗ Failed\n\n")
                        if 'error' in test:
                            f.write(f"**Error:** {test['error']}\n\n")
                    
                    f.write("---\n\n")
        
        print(f"📄 Markdown report generated: {report_path}")
    
    def _group_tests_by_category(self):
        """Group tests by category based on their name."""
        categories = {}
        
        for test in self.test_details:
            name = test['name']
            
            if 'chain' in name:
                category = "Filter Chain Tests"
            elif 'large' in name:
                category = "Large Image Tests"
            elif 'format' in name:
                category = "Format Compatibility Tests"
            else:
                category = "Single Filter Tests"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(test)
        
        return categories


def generate_images(output_dir):
    """Generate test images."""
    script_dir = Path(__file__).parent
    gen_script = script_dir / "generate_test_images.py"
    
    if not gen_script.exists():
        print(f"Error: {gen_script} not found")
        return False
    
    print("Generating test images...")
    result = subprocess.run([sys.executable, str(gen_script), output_dir], 
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating test images: {result.stderr}")
        return False
    
    print(result.stdout)
    return True


def download_test_image(test_dir):
    """Download a real PNG image from the web for testing."""
    try:
        import urllib.request
        from PIL import Image
        import io
        
        test_dir = Path(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = test_dir / f"real_world_{IMAGE_SIZE}x{IMAGE_SIZE}.png"
        
        # Check if already downloaded
        if output_path.exists():
            print(f"✓ Real-world test image already exists: {output_path}")
            return True
        
        print("Downloading real-world test image from web...")
        
        # Download a sample image from a stable, public-domain source
        # Using a small, public domain test image hosted on Wikimedia Commons
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/320px-Fronalpstock_big.jpg"
        
        try:
            # Try to download from the configured public-domain source
            with urllib.request.urlopen(url, timeout=10) as response:
                image_data = response.read()
            
            # Convert to PNG and resize to desired size
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary (in case it's RGBA or another format)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to IMAGE_SIZE x IMAGE_SIZE
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            
            # Save as PNG
            img.save(output_path, 'PNG')
            print(f"  ✓ Downloaded and saved: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"  ⚠ Could not download from primary source: {str(e)[:60]}")
            
            # Fallback: create a synthetic "real-world-like" image if download fails
            print("  Creating synthetic real-world test image...")
            import numpy as np
            
            # Create a composite image that looks somewhat real-world (Vectorized)
            Y, X = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
            
            r = (Y * 255) // IMAGE_SIZE
            g = (X * 255) // IMAGE_SIZE
            b = ((Y + X) * 255) // (IMAGE_SIZE * 2)
            
            img_array = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            img_array[..., 0] = r.astype(np.uint8)
            img_array[..., 1] = g.astype(np.uint8)
            img_array[..., 2] = b.astype(np.uint8)
            
            # Add some noise to make it look more real
            noise = np.random.randint(-20, 20, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_array, 'RGB')
            img.save(output_path, 'PNG')
            print(f"  ✓ Created synthetic real-world image: {output_path.name}")
            return True
            
    except Exception as e:
        print(f"✗ Error downloading/creating real-world test image: {str(e)}")
        return False


def find_run_kernels(path):
    """Find the run_kernels executable."""
    if path and Path(path).exists():
        return Path(path)
    
    # Try common locations
    candidates = [
        Path("../build/debug/src/run_kernels"),
        Path("build/debug/src/run_kernels"),
        Path("./run_kernels"),
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Generate test images")
    parser.add_argument("--run-kernels", default=None, help="Path to run_kernels executable")
    parser.add_argument("--test-dir", default="./test_images", help="Test images directory")
    parser.add_argument("--output-dir", default="./test_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Generate test images if requested
    if args.generate:
        if not generate_images(args.test_dir):
            return 1
        # Also download/create real-world test image
        download_test_image(args.test_dir)
    else:
        # Always ensure real-world test image exists
        download_test_image(args.test_dir)
    
    # Find run_kernels executable
    run_kernels = find_run_kernels(args.run_kernels)
    if not run_kernels:
        print("Error: Could not find run_kernels executable")
        print("Specify with: python3 test_filters.py --run-kernels <path>")
        return 1
    
    print(f"Using executable: {run_kernels}\n")
    
    # Run tests
    runner = TestRunner(str(run_kernels), args.test_dir, args.output_dir)
    success = runner.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
