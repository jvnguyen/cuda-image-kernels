# CUDA Image Filter Test Report

**Generated:** 2026-02-04 16:36:21

## Summary

- **Total Tests:** 24
- **Passed:** 24 ✓
- **Failed:** 0 ✗
- **Success Rate:** 100%

## Single Filter Tests

### ✓ stripe_256x256_greyscale

**Filters:** `greyscale`

| Input | Output |
|-------|--------|
| ![Input](../test_images/stripe_256x256.png) | ![Output](stripe_256x256_greyscale.png) |

---

### ✓ stripe_256x256_invert

**Filters:** `invert`

| Input | Output |
|-------|--------|
| ![Input](../test_images/stripe_256x256.png) | ![Output](stripe_256x256_invert.png) |

---

### ✓ stripe_256x256_normalize

**Filters:** `normalize`

| Input | Output |
|-------|--------|
| ![Input](../test_images/stripe_256x256.png) | ![Output](stripe_256x256_normalize.png) |

---

### ✓ stripe_256x256_threshold_0.3

**Filters:** `threshold --threshold 0.3`

| Input | Output |
|-------|--------|
| ![Input](../test_images/stripe_256x256.png) | ![Output](stripe_256x256_threshold_0.3.png) |

---

### ✓ stripe_256x256_threshold_0.7

**Filters:** `threshold --threshold 0.7`

| Input | Output |
|-------|--------|
| ![Input](../test_images/stripe_256x256.png) | ![Output](stripe_256x256_threshold_0.7.png) |

---

### ✓ gradient_256x256_greyscale

**Filters:** `greyscale`

| Input | Output |
|-------|--------|
| ![Input](../test_images/gradient_256x256.png) | ![Output](gradient_256x256_greyscale.png) |

---

### ✓ gradient_256x256_invert

**Filters:** `invert`

| Input | Output |
|-------|--------|
| ![Input](../test_images/gradient_256x256.png) | ![Output](gradient_256x256_invert.png) |

---

### ✓ gradient_256x256_normalize

**Filters:** `normalize`

| Input | Output |
|-------|--------|
| ![Input](../test_images/gradient_256x256.png) | ![Output](gradient_256x256_normalize.png) |

---

### ✓ gradient_256x256_threshold_0.3

**Filters:** `threshold --threshold 0.3`

| Input | Output |
|-------|--------|
| ![Input](../test_images/gradient_256x256.png) | ![Output](gradient_256x256_threshold_0.3.png) |

---

### ✓ gradient_256x256_threshold_0.7

**Filters:** `threshold --threshold 0.7`

| Input | Output |
|-------|--------|
| ![Input](../test_images/gradient_256x256.png) | ![Output](gradient_256x256_threshold_0.7.png) |

---

### ✓ circles_256x256_greyscale

**Filters:** `greyscale`

| Input | Output |
|-------|--------|
| ![Input](../test_images/circles_256x256.png) | ![Output](circles_256x256_greyscale.png) |

---

### ✓ circles_256x256_normalize

**Filters:** `normalize`

| Input | Output |
|-------|--------|
| ![Input](../test_images/circles_256x256.png) | ![Output](circles_256x256_normalize.png) |

---

### ✓ noise_256x256_channel_0

**Filters:** `channel --channel 0`

| Input | Output |
|-------|--------|
| ![Input](../test_images/noise_256x256.png) | ![Output](noise_256x256_channel_0.png) |

---

### ✓ noise_256x256_channel_1

**Filters:** `channel --channel 1`

| Input | Output |
|-------|--------|
| ![Input](../test_images/noise_256x256.png) | ![Output](noise_256x256_channel_1.png) |

---

### ✓ noise_256x256_channel_2

**Filters:** `channel --channel 2`

| Input | Output |
|-------|--------|
| ![Input](../test_images/noise_256x256.png) | ![Output](noise_256x256_channel_2.png) |

---

## Filter Chain Tests

### ✓ chain_greyscale_brightness_invert

**Filters:** `greyscale brightness --brightness 1.5 invert`

#### Filter Progression

| Step | Filters Applied | Result |
|------|-----------------|--------|
| 0 | (input) | ![Input](../test_images/stripe_256x256.png) |
| 1 | greyscale | ![Step 1](chain_greyscale_brightness_invert_step_1_greyscale.png) |
| 2 | brightness --brightness 1.5 | ![Step 2](chain_greyscale_brightness_invert_step_2_brightness.png) |
| 3 | invert | ![Step 3](chain_greyscale_brightness_invert_step_3_invert.png) |

---

### ✓ chain_normalize_threshold_brightness

**Filters:** `normalize threshold --threshold 0.5 brightness --brightness 1.2`

#### Filter Progression

| Step | Filters Applied | Result |
|------|-----------------|--------|
| 0 | (input) | ![Input](../test_images/gradient_256x256.png) |
| 1 | normalize | ![Step 1](chain_normalize_threshold_brightness_step_1_normalize.png) |
| 2 | threshold --threshold 0.5 | ![Step 2](chain_normalize_threshold_brightness_step_2_threshold.png) |
| 3 | brightness --brightness 1.2 | ![Step 3](chain_normalize_threshold_brightness_step_3_brightness.png) |

---

### ✓ chain_greyscale_invert_normalize

**Filters:** `greyscale invert normalize`

#### Filter Progression

| Step | Filters Applied | Result |
|------|-----------------|--------|
| 0 | (input) | ![Input](../test_images/circles_256x256.png) |
| 1 | greyscale | ![Step 1](chain_greyscale_invert_normalize_step_1_greyscale.png) |
| 2 | invert | ![Step 2](chain_greyscale_invert_normalize_step_2_invert.png) |
| 3 | normalize | ![Step 3](chain_greyscale_invert_normalize_step_3_normalize.png) |

---

### ✓ chain_brightness_invert

**Filters:** `brightness --brightness 0.8 invert`

#### Filter Progression

| Step | Filters Applied | Result |
|------|-----------------|--------|
| 0 | (input) | ![Input](../test_images/noise_256x256.png) |
| 1 | brightness --brightness 0.8 | ![Step 1](chain_brightness_invert_step_1_brightness.png) |
| 2 | invert | ![Step 2](chain_brightness_invert_step_2_invert.png) |

---

## Format Compatibility Tests

### ✓ format_stripe_256x256

**Filters:** `greyscale`

| Input | Output |
|-------|--------|
| ![Input](../test_images/stripe_256x256.png) | ![Output](format_stripe_256x256.png) |

---

### ✓ format_gradient_256x256

**Filters:** `greyscale`

| Input | Output |
|-------|--------|
| ![Input](../test_images/gradient_256x256.png) | ![Output](format_gradient_256x256.png) |

---

### ✓ format_circles_256x256

**Filters:** `greyscale`

| Input | Output |
|-------|--------|
| ![Input](../test_images/circles_256x256.png) | ![Output](format_circles_256x256.png) |

---

### ✓ format_noise_256x256

**Filters:** `greyscale`

| Input | Output |
|-------|--------|
| ![Input](../test_images/noise_256x256.png) | ![Output](format_noise_256x256.png) |

---

### ✓ format_real_world_256x256

**Filters:** `greyscale`

| Input | Output |
|-------|--------|
| ![Input](../test_images/real_world_256x256.png) | ![Output](format_real_world_256x256.png) |

---

