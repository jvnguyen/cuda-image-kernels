#pragma once

#include <string>
#include <vector>

/**
 * Read PNG file into RGBA buffer
 * 
 * Automatically converts various PNG formats (RGB, grayscale, palette, etc.)
 * to RGBA format (8-bit per channel).
 * 
 * @param filename Input PNG file path
 * @param imageData Output buffer containing RGBA pixel data (4 bytes per pixel)
 * @param width Output image width
 * @param height Output image height
 * @return true on success, false on failure
 */
bool readPNG(const std::string& filename, std::vector<unsigned char>& imageData, 
             int& width, int& height);

/**
 * Write PNG file from RGBA buffer
 * 
 * Writes RGBA pixel data as a PNG file with 8-bit per channel.
 * 
 * @param filename Output PNG file path
 * @param imageData RGBA pixel data buffer (4 bytes per pixel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @return true on success, false on failure
 */
bool writePNG(const std::string& filename, const std::vector<unsigned char>& imageData,
              int width, int height);
