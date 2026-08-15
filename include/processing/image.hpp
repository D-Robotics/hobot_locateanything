#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "model_profile.hpp"

namespace locateanything {

/**
 * @brief Convert an NV12 buffer, including an optional row stride, to BGR.
 * @param data Source NV12 bytes.
 * @param data_size Available source bytes.
 * @param width Source width in pixels.
 * @param height Source height in pixels.
 * @param step Source row stride, or zero to infer tightly packed rows.
 * @return Owned three-channel BGR image.
 */
cv::Mat Nv12ToBgr(const uint8_t* data, size_t data_size, uint32_t width,
                  uint32_t height, uint32_t step = 0);

/**
 * @brief Decode a JPEG buffer to an owned BGR image.
 * @param data Source JPEG bytes.
 * @param data_size Available source bytes.
 * @return Owned three-channel BGR image.
 */
cv::Mat JpegToBgr(const uint8_t* data, size_t data_size);

/**
 * @brief Convert packed BGR/RGB bytes to a tightly packed BGR image.
 * @param data Source packed-color bytes.
 * @param data_size Available source bytes.
 * @param width Source width in pixels.
 * @param height Source height in pixels.
 * @param step Source row stride, or zero for tightly packed rows.
 * @param input_is_rgb Convert RGB channel order to BGR when true.
 * @return Owned three-channel BGR image.
 */
cv::Mat PackedColorToBgr(const uint8_t* data, size_t data_size,
                         uint32_t width, uint32_t height, uint32_t step,
                         bool input_is_rgb);

/** Geometric transform needed to map model coordinates to source pixels. */
struct ImageTransform {
  int source_width = 0;
  int source_height = 0;
  int canvas_width = 0;
  int canvas_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int pad_left = 0;
  int pad_top = 0;
  float scale_x = 1.0f;
  float scale_y = 1.0f;
};

/** Prepared Vision input and its source-to-model transform. */
struct PreparedImage {
  std::vector<uint16_t> patches;
  ImageTransform transform;
};

class ImagePreprocessor {
 public:
  explicit ImagePreprocessor(VisionProfile profile = {});
  /**
   * @brief Resize, pad, normalize, and tile a BGR image for Vision.
   * @param bgr Non-empty three-channel source image.
   * @return FP16 Vision patches and source-coordinate transform.
   */
  PreparedImage Prepare(const cv::Mat& bgr) const;

 private:
  VisionProfile profile_;
};

}  // namespace locateanything
