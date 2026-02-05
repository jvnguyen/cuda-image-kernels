#include "png_utils.h"
#include <png.h>
#include <cstdio>
#include <cstring>
#include <iostream>

bool readPNG(const std::string& filename, std::vector<unsigned char>& imageData, 
             int& width, int& height) {
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file '" << filename << "'\n";
        return false;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::fclose(fp);
        std::cerr << "Error: Failed to create PNG read structure\n";
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        std::fclose(fp);
        std::cerr << "Error: Failed to create PNG info structure\n";
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        std::cerr << "Error: PNG read failed\n";
        return false;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte colorType = png_get_color_type(png, info);
    png_byte bitDepth = png_get_bit_depth(png, info);

    // Convert to RGBA if needed
    if (bitDepth == 16)
        png_set_strip_16(png);
    if (colorType == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if (colorType == PNG_COLOR_TYPE_RGB || colorType == PNG_COLOR_TYPE_GRAY || 
        colorType == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    // Read image data
    std::vector<png_bytep> row_pointers(height);
    imageData.resize(4 * width * height);
    for (int i = 0; i < height; ++i) {
        row_pointers[i] = imageData.data() + i * 4 * width;
    }

    png_read_image(png, row_pointers.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);

    std::cout << "✓ Read PNG: " << filename << " (" << width << "x" << height << ")\n";
    return true;
}

bool writePNG(const std::string& filename, const std::vector<unsigned char>& imageData,
              int width, int height) {
    FILE* fp = std::fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "Error: Cannot create file '" << filename << "'\n";
        return false;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::fclose(fp);
        std::cerr << "Error: Failed to create PNG write structure\n";
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        std::fclose(fp);
        std::cerr << "Error: Failed to create PNG info structure\n";
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::fclose(fp);
        std::cerr << "Error: PNG write failed\n";
        return false;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // Write row pointers
    std::vector<png_bytep> row_pointers(height);
    // png_write_image requires non-const row pointers, but does not modify the pixel data;
    // imageData is const to this function only, so using const_cast here is safe.
    for (int i = 0; i < height; ++i) {
        row_pointers[i] = const_cast<png_bytep>(imageData.data() + i * 4 * width);
    }

    png_write_info(png, info);
    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    std::fclose(fp);

    std::cout << "✓ Wrote PNG: " << filename << " (" << width << "x" << height << ")\n";
    return true;
}
