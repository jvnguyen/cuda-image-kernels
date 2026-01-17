#pragma once
#include <cuda_runtime.h>

// ============================================================
// Generic Type Conversion Traits
// ============================================================
// Provides type conversion utilities for image processing kernels
// with support for scalar and CUDA vector types

// Generic trait (default specialization)
template<typename T>
struct ConversionTraits
{
    // Normalize value to [0, 1] float range
    __device__ __host__
    static float toNormalized(T value)
    {
        return static_cast<float>(value);
    }

    // Denormalize from [0, 1] float range
    __device__ __host__
    static T fromNormalized(float value)
    {
        return static_cast<T>(value);
    }

    // Clamp value to valid range for output type
    __device__ __host__
    static T clamp(float value)
    {
        return static_cast<T>(value);
    }
};

// Specialization for unsigned char (normalize to [0, 1], denormalize to [0, 255])
template<>
struct ConversionTraits<unsigned char>
{
    __device__ __host__
    static float toNormalized(unsigned char value)
    {
        return static_cast<float>(value) / 255.0f;
    }

    __device__ __host__
    static unsigned char fromNormalized(float value)
    {
        return static_cast<unsigned char>(value * 255.0f);
    }

    __device__ __host__
    static unsigned char clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        return static_cast<unsigned char>(clamped * 255.0f);
    }
};

// Specialization for float (keep in [0, 1] range)
template<>
struct ConversionTraits<float>
{
    __device__ __host__
    static float toNormalized(float value)
    {
        return value;
    }

    __device__ __host__
    static float fromNormalized(float value)
    {
        return value;
    }

    __device__ __host__
    static float clamp(float value)
    {
        return fmaxf(0.0f, fminf(1.0f, value));
    }
};

// ============================================================
// CUDA Vector Type Specializations
// ============================================================

// Specialization for uchar3 (RGB with 8-bit channels)
template<>
struct ConversionTraits<uchar3>
{
    __device__ __host__
    static float toNormalized(uchar3 value)
    {
        // Average the RGB channels
        float r = static_cast<float>(value.x) / 255.0f;
        float g = static_cast<float>(value.y) / 255.0f;
        float b = static_cast<float>(value.z) / 255.0f;
        return (r + g + b) / 3.0f;
    }

    __device__ __host__
    static uchar3 fromNormalized(float value)
    {
        unsigned char grey = static_cast<unsigned char>(value * 255.0f);
        return make_uchar3(grey, grey, grey);
    }

    __device__ __host__
    static uchar3 clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        unsigned char grey = static_cast<unsigned char>(clamped * 255.0f);
        return make_uchar3(grey, grey, grey);
    }
};

// Specialization for uchar4 (RGBA with 8-bit channels)
template<>
struct ConversionTraits<uchar4>
{
    __device__ __host__
    static float toNormalized(uchar4 value)
    {
        // Average the RGB channels (ignore alpha)
        float r = static_cast<float>(value.x) / 255.0f;
        float g = static_cast<float>(value.y) / 255.0f;
        float b = static_cast<float>(value.z) / 255.0f;
        return (r + g + b) / 3.0f;
    }

    __device__ __host__
    static uchar4 fromNormalized(float value)
    {
        unsigned char grey = static_cast<unsigned char>(value * 255.0f);
        return make_uchar4(grey, grey, grey, 255);
    }

    __device__ __host__
    static uchar4 clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        unsigned char grey = static_cast<unsigned char>(clamped * 255.0f);
        return make_uchar4(grey, grey, grey, 255);
    }
};

// Specialization for float3 (RGB with floating-point channels)
template<>
struct ConversionTraits<float3>
{
    __device__ __host__
    static float toNormalized(float3 value)
    {
        // Average the RGB channels
        return (value.x + value.y + value.z) / 3.0f;
    }

    __device__ __host__
    static float3 fromNormalized(float value)
    {
        return make_float3(value, value, value);
    }

    __device__ __host__
    static float3 clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        return make_float3(clamped, clamped, clamped);
    }
};

// Specialization for float4 (RGBA with floating-point channels)
template<>
struct ConversionTraits<float4>
{
    __device__ __host__
    static float toNormalized(float4 value)
    {
        // Average the RGB channels (ignore alpha)
        return (value.x + value.y + value.z) / 3.0f;
    }

    __device__ __host__
    static float4 fromNormalized(float value)
    {
        return make_float4(value, value, value, 1.0f);
    }

    __device__ __host__
    static float4 clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        return make_float4(clamped, clamped, clamped, 1.0f);
    }
};

// ============================================================
// Specialized Luminance Traits for Greyscale
// ============================================================
// For operations that need BT.601 luminance instead of averaging

// Generic luminance trait
template<typename T>
struct LuminanceTraits
{
    // Convert using BT.601 luminance formula
    __device__ __host__
    static float toLuminance(T value)
    {
        return ConversionTraits<T>::toNormalized(value);
    }

    __device__ __host__
    static T fromLuminance(float value)
    {
        return ConversionTraits<T>::fromNormalized(value);
    }

    __device__ __host__
    static T clamp(float value)
    {
        return ConversionTraits<T>::clamp(value);
    }
};

// Specialization for uchar3: apply BT.601 luminance
template<>
struct LuminanceTraits<uchar3>
{
    __device__ __host__
    static float toLuminance(uchar3 value)
    {
        float r = static_cast<float>(value.x) / 255.0f;
        float g = static_cast<float>(value.y) / 255.0f;
        float b = static_cast<float>(value.z) / 255.0f;
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    __device__ __host__
    static uchar3 fromLuminance(float value)
    {
        unsigned char grey = static_cast<unsigned char>(value * 255.0f);
        return make_uchar3(grey, grey, grey);
    }

    __device__ __host__
    static uchar3 clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        unsigned char grey = static_cast<unsigned char>(clamped * 255.0f);
        return make_uchar3(grey, grey, grey);
    }
};

// Specialization for uchar4: apply BT.601 luminance
template<>
struct LuminanceTraits<uchar4>
{
    __device__ __host__
    static float toLuminance(uchar4 value)
    {
        float r = static_cast<float>(value.x) / 255.0f;
        float g = static_cast<float>(value.y) / 255.0f;
        float b = static_cast<float>(value.z) / 255.0f;
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    __device__ __host__
    static uchar4 fromLuminance(float value)
    {
        unsigned char grey = static_cast<unsigned char>(value * 255.0f);
        return make_uchar4(grey, grey, grey, 255);
    }

    __device__ __host__
    static uchar4 clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        unsigned char grey = static_cast<unsigned char>(clamped * 255.0f);
        return make_uchar4(grey, grey, grey, 255);
    }
};

// Specialization for float3: apply BT.601 luminance
template<>
struct LuminanceTraits<float3>
{
    __device__ __host__
    static float toLuminance(float3 value)
    {
        return 0.299f * value.x + 0.587f * value.y + 0.114f * value.z;
    }

    __device__ __host__
    static float3 fromLuminance(float value)
    {
        return make_float3(value, value, value);
    }

    __device__ __host__
    static float3 clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        return make_float3(clamped, clamped, clamped);
    }
};

// Specialization for float4: apply BT.601 luminance
template<>
struct LuminanceTraits<float4>
{
    __device__ __host__
    static float toLuminance(float4 value)
    {
        return 0.299f * value.x + 0.587f * value.y + 0.114f * value.z;
    }

    __device__ __host__
    static float4 fromLuminance(float value)
    {
        return make_float4(value, value, value, 1.0f);
    }

    __device__ __host__
    static float4 clamp(float value)
    {
        float clamped = fmaxf(0.0f, fminf(1.0f, value));
        return make_float4(clamped, clamped, clamped, 1.0f);
    }
};
