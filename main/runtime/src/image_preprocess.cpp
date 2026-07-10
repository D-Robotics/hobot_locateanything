// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include "locateanything_runtime/image_preprocess.hpp"

#include <cmath>
#include <cstring>

namespace locateanything_runtime {

namespace {

constexpr int32_t kPatchSize = 14;
constexpr int32_t kPatchGrid = 32;     // 448 / 14
constexpr int32_t kPatchCount = 1024;  // 32 * 32
constexpr int32_t kPatchDim = 588;     // 3 * 14 * 14

// Simple xorshift32 PRNG — deterministic, no <random> bloat, no Date.now()
// style concerns. Mirrors numpy's normal via Box-Muller.
uint32_t XorShift32(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

float StandardNormal(uint32_t &state) {
  // Box-Muller transform
  float u1 = (XorShift32(state) + 1.0f) / 4294967296.0f;  // (0, 1]
  float u2 = (XorShift32(state) + 1.0f) / 4294967296.0f;
  float r = std::sqrt(-2.0f * std::log(u1));
  return r * std::cos(2.0f * 3.14159265358979f * u2);
}

}  // namespace

bool BuildVisionPatchTensor(const void *image_hwc,
                            int32_t height,
                            int32_t width,
                            int32_t channels,
                            bool is_uint8,
                            ImagePatchTensor *out) {
  if (height != 448 || width != 448 || channels != 3) {
    return false;
  }
  out->shape = {1, kPatchCount, kPatchDim};
  out->data.resize(static_cast<size_t>(kPatchCount) * kPatchDim);

  // For each of the 32x32 patches, gather the 14x14x3 = 588 values from
  // the HWC image and write them out flat in fp32.
  for (int32_t py = 0; py < kPatchGrid; ++py) {
    for (int32_t px = 0; px < kPatchGrid; ++px) {
      int32_t patch_idx = py * kPatchGrid + px;
      float *dst = out->data.data() + static_cast<size_t>(patch_idx) * kPatchDim;
      int32_t dst_off = 0;
      for (int32_t cy = 0; cy < kPatchSize; ++cy) {
        for (int32_t cx = 0; cx < kPatchSize; ++cx) {
          int32_t iy = py * kPatchSize + cy;
          int32_t ix = px * kPatchSize + cx;
          size_t src_off = (static_cast<size_t>(iy) * width + ix) * channels;
          if (is_uint8) {
            const uint8_t *p = static_cast<const uint8_t *>(image_hwc);
            // BGR? RGB? upstream MoonViT preprocessing: RGB. Assume RGB.
            dst[dst_off + 0] = static_cast<float>(p[src_off + 0]) / 255.0f;
            dst[dst_off + 1] = static_cast<float>(p[src_off + 1]) / 255.0f;
            dst[dst_off + 2] = static_cast<float>(p[src_off + 2]) / 255.0f;
          } else {
            const float *p = static_cast<const float *>(image_hwc);
            dst[dst_off + 0] = p[src_off + 0];
            dst[dst_off + 1] = p[src_off + 1];
            dst[dst_off + 2] = p[src_off + 2];
          }
          dst_off += 3;
        }
      }
    }
  }
  return true;
}

bool BuildDummyVisionPatchTensor(ImagePatchTensor *out, uint32_t seed) {
  out->shape = {1, kPatchCount, kPatchDim};
  out->data.resize(static_cast<size_t>(kPatchCount) * kPatchDim);
  uint32_t state = seed == 0 ? 1 : seed;
  for (size_t i = 0; i < out->data.size(); ++i) {
    out->data[i] = StandardNormal(state);
  }
  return true;
}

}  // namespace locateanything_runtime
