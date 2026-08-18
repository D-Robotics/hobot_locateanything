#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace locateanything {

/** Static MoonViT canvas plus architecture-derived runtime dimensions. */
class VisionProfile {
 public:
  static constexpr int32_t kPatchSize = 14;
  static constexpr int32_t kMergeSize = 2;
  static constexpr int32_t kChannels = 3;
  static constexpr int32_t kHiddenSize = 2048;

  VisionProfile(int32_t image_width = 336, int32_t image_height = 336,
                std::string resize_mode = "letterbox",
                int32_t letterbox_fill = 128)
      : image_width_(image_width),
        image_height_(image_height),
        resize_mode_(std::move(resize_mode)),
        letterbox_fill_(letterbox_fill) {
    Validate();
  }

  int32_t image_width() const { return image_width_; }
  int32_t image_height() const { return image_height_; }
  const std::string& resize_mode() const { return resize_mode_; }
  int32_t letterbox_fill() const { return letterbox_fill_; }
  int32_t grid_width() const { return image_width_ / kPatchSize; }
  int32_t grid_height() const { return image_height_ / kPatchSize; }
  int32_t patch_count() const { return grid_width() * grid_height(); }
  int32_t patch_flat_dim() const {
    return kChannels * kPatchSize * kPatchSize;
  }
  int32_t visual_token_count() const {
    return patch_count() / (kMergeSize * kMergeSize);
  }

 private:
  void Validate() const {
    constexpr int32_t kProfileMultiple = kPatchSize * kMergeSize;
    if (image_width_ <= 0 || image_height_ <= 0 ||
        image_width_ % kProfileMultiple != 0 ||
        image_height_ % kProfileMultiple != 0) {
      throw std::invalid_argument(
          "Vision image dimensions must be positive multiples of 28");
    }
    if (resize_mode_ != "letterbox" && resize_mode_ != "stretch") {
      throw std::invalid_argument(
          "Vision resize_mode must be letterbox or stretch");
    }
    if (letterbox_fill_ < 0 || letterbox_fill_ > 255) {
      throw std::invalid_argument("Vision letterbox_fill must be in [0, 255]");
    }
  }

  int32_t image_width_;
  int32_t image_height_;
  std::string resize_mode_;
  int32_t letterbox_fill_;
};

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
  std::vector<uint8_t> patches_fp16;
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
