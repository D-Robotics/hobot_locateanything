#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

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

}  // namespace locateanything
