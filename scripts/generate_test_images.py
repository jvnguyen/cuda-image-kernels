#!/usr/bin/env python3
"""
Test image generator for CUDA image processing kernels.

This script generates 256x256 test PNG images using NumPy to test the
main.cpp image filter pipeline. It creates diverse test cases:
- Gradient patterns
- Circular patterns
- Random noise

All images are generated at 256x256 resolution for consistent testing.

Usage:
    python3 generate_test_images.py [output_dir]

Default output directory: ./test_images/
"""

import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")


def create_stripe_image(width, height, filename):
    """Create a stripe pattern image (vertical stripes with R-G-B colors)."""
    # Vectorized implementation using NumPy
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    stripe_width = width // 3
    
    # Red stripes
    img_array[:, 0:stripe_width, 0] = 255
    # Green stripes
    img_array[:, stripe_width:2*stripe_width, 1] = 255
    # Blue stripes
    img_array[:, 2*stripe_width:, 2] = 255
    
    img = Image.fromarray(img_array, 'RGB')
    img.save(filename)
    print(f"  ✓ {Path(filename).name} ({width}x{height})")


def create_gradient_image(width, height, filename):
    """Create a linear gradient image (R-G-B)."""
    # Vectorized implementation using NumPy
    x = np.linspace(0, 255, width)
    y = np.linspace(0, 255, height)
    xv, yv = np.meshgrid(x, y)
    
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    img_array[:, :, 0] = xv.astype(np.uint8)
    img_array[:, :, 1] = yv.astype(np.uint8)
    img_array[:, :, 2] = 128
    
    img = Image.fromarray(img_array, 'RGB')
    img.save(filename)
    print(f"  ✓ {Path(filename).name} ({width}x{height})")



def create_circles_image(width, height, filename):
    """Create an image with circles of different sizes."""
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Vectorized implementation
    Y, X = np.ogrid[:height, :width]
    
    # Draw circles with gradient colors
    num_circles = 5
    # Calculate spacing and radius to avoid overlap
    spacing = width // (num_circles + 1)
    radius = int(spacing * 0.4)

    for i in range(num_circles):
        cx = spacing * (i + 1)
        cy = height // 2
        
        # Gradient colors spanning full range
        r = int((i / (num_circles - 1)) * 255)
        g = int(((num_circles - 1 - i) / (num_circles - 1)) * 255)
        b = 128
        
        dist_sq = (X - cx)**2 + (Y - cy)**2
        mask = dist_sq <= radius**2
        img_array[mask] = [r, g, b]
    
    img = Image.fromarray(img_array, 'RGB')
    img.save(filename)
    print(f"  ✓ {Path(filename).name} ({width}x{height})")


def create_noise_image(width, height, filename):
    """Create a random noise image."""
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    img = Image.fromarray(img_array, 'RGB')
    img.save(filename)
    print(f"  ✓ {Path(filename).name} ({width}x{height})")


def create_test_images(width=256, height=256, output_dir="./test_images"):
    """Generate test images."""
    ensure_output_dir(output_dir)
    
    print(f"\n📷 Generating {width}x{height} test images...\n")
    print("Test images:")
    create_stripe_image(width, height, os.path.join(output_dir, f"stripe_{width}x{height}.png"))
    create_gradient_image(width, height, os.path.join(output_dir, f"gradient_{width}x{height}.png"))
    create_circles_image(width, height, os.path.join(output_dir, f"circles_{width}x{height}.png"))
    create_noise_image(width, height, os.path.join(output_dir, f"noise_{width}x{height}.png"))
    
    print("\n✅ All test images generated successfully!\n")
    print(f"Test images location: {os.path.abspath(output_dir)}/\n")


def print_usage():
    """Print usage information."""
    print(__doc__)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", default="./test_images",
                        help="Output directory for generated images (default: ./test_images)")
    parser.add_argument("--width", type=int, default=256,
                        help="Width of the test images (default: 256)")
    parser.add_argument("--height", type=int, default=256,
                        help="Height of the test images (default: 256)")
    args = parser.parse_args()
    output_dir = args.output_dir
    
    try:
        create_test_images(args.width, args.height, output_dir)
    except ImportError as e:
        print(f"Error: Missing required module: {e}")
        print("Install dependencies with: pip install Pillow numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
