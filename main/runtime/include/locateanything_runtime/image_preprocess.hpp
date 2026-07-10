// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// MoonViT image preprocessing for the C++ host runtime.
//
// Phase 1 scope: produce a (1, 1024, 588) fp32 patch tensor suitable for
// the visual graph's only input. We do NOT decode JPEGs yet — that's
// delegated to a tiny header-only stb_image drop in Phase 2. Here we only:
//   1) accept a 448x448x3 uint8/float RGB buffer in HWC layout
//   2) split into 32x32 patches of size 14x14x3
//   3) flatten each patch as a 588-element fp32 vector
//   4) stack into (1, 1024, 588)
//
// The output shape matches vision.hbm::visual's declared input:
//   _input_0 : (1, 1024, 588) fp32

#pragma once

#include <cstdint>
#include <vector>

namespace locateanything_runtime {

struct ImagePatchTensor {
  std::vector<int32_t> shape;   // [1, 1024, 588]
  std::vector<float> data;      // 1 * 1024 * 588 floats
};

// Convert an HxWx3 HWC RGB image (either uint8 or float source) into the
// (1, 1024, 588) fp32 patch stream MoonViT expects.
//
//   image_hwc:    raw pixel buffer, H rows * W cols * 3 channels, row-major
//   height:       must equal 448 (MoonViT config patch_size=14 -> 32x32 grid)
//   width:       must equal 448
//   channels:    must be 3
//   is_uint8:    if true, pixel values are 0..255 and get divided by 255; if
//                false, pixel values are already float in [0,1] (or normalized)
//
// Returns false on shape mismatch.
bool BuildVisionPatchTensor(const void *image_hwc,
                            int32_t height,
                            int32_t width,
                            int32_t channels,
                            bool is_uint8,
                            ImagePatchTensor *out);

// Build a dummy (1, 1024, 588) fp32 patch tensor filled with normal(0,1)
// noise. Used for the Phase-1 sanity run (we don't have a real image
// pipeline yet). Deterministic by seed so reruns are reproducible.
bool BuildDummyVisionPatchTensor(ImagePatchTensor *out, uint32_t seed = 42);

}  // namespace locateanything_runtime
