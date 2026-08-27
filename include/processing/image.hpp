#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace locateanything {

/** Vision canvas and tensor dimensions discovered from the loaded HBM. */
class VisionProfile {
 public:
  static constexpr int32_t kChannels = 3;
  static constexpr int32_t kLetterboxFill = 128;

  /**
   * @brief Create a validated profile discovered from one Vision HBM.
   * @param[in] image_width Square model-canvas width in pixels.
   * @param[in] image_height Square model-canvas height in pixels.
   * @param[in] patch_size Spatial width and height of one RGB input patch.
   * @param[in] visual_tokens Number of visual feature rows emitted by Vision.
   * @param[in] hidden_size Number of fp16 values in each visual feature row.
   * @throws std::invalid_argument if the dimensions cannot describe the
   * supported single-image square Vision input.
   */
  VisionProfile(int32_t image_width, int32_t image_height,
                int32_t patch_size, int32_t visual_tokens,
                int32_t hidden_size)
      : image_width_(image_width),
        image_height_(image_height),
        patch_size_(patch_size),
        visual_tokens_(visual_tokens),
        hidden_size_(hidden_size) {
    Validate();
  }

  /** @brief Return the model-canvas width. @return Width in pixels. */
  int32_t image_width() const { return image_width_; }
  /** @brief Return the model-canvas height. @return Height in pixels. */
  int32_t image_height() const { return image_height_; }
  /** @brief Return the square patch edge. @return Patch size in pixels. */
  int32_t patch_size() const { return patch_size_; }
  /** @brief Return the horizontal patch grid. @return Patch columns. */
  int32_t grid_width() const { return image_width_ / patch_size_; }
  /** @brief Return the vertical patch grid. @return Patch rows. */
  int32_t grid_height() const { return image_height_ / patch_size_; }
  /** @brief Return the Vision patch count. @return Total input patches. */
  int32_t patch_count() const { return grid_width() * grid_height(); }
  /**
   * @brief Return the scalar width of one flattened RGB patch.
   * @return Three times the squared patch edge.
   */
  int32_t patch_flat_dim() const {
    return kChannels * patch_size_ * patch_size_;
  }
  /** @brief Return the Vision output rows. @return Visual-token count. */
  int32_t visual_token_count() const { return visual_tokens_; }
  /** @brief Return the Vision feature width. @return Hidden dimension. */
  int32_t hidden_size() const { return hidden_size_; }

 private:
  /**
   * @brief Validate the square canvas, patch tiling, and output dimensions.
   * @throws std::invalid_argument if the stored dimensions are inconsistent.
   */
  void Validate() const {
    if (image_width_ <= 0 || image_height_ <= 0 ||
        image_width_ != image_height_ || patch_size_ <= 0 ||
        image_width_ % patch_size_ != 0 ||
        image_height_ % patch_size_ != 0 || visual_tokens_ <= 0 ||
        hidden_size_ <= 0) {
      throw std::invalid_argument(
          "Vision profile must describe a positive square canvas with "
          "complete RGB patches and positive output dimensions");
    }
  }

  int32_t image_width_;
  int32_t image_height_;
  int32_t patch_size_;
  int32_t visual_tokens_;
  int32_t hidden_size_;
};

/**
 * @brief Convert an NV12 buffer, including an optional row stride, to BGR.
 * @param[in] data Source NV12 bytes.
 * @param[in] data_size Available source bytes.
 * @param[in] width Source width in pixels.
 * @param[in] height Source height in pixels.
 * @param[in] step Source row stride, or zero to infer tightly packed rows.
 * @return Owned three-channel BGR image.
 * @throws std::runtime_error if dimensions, stride, or source storage are
 * invalid.
 */
cv::Mat Nv12ToBgr(const uint8_t* data, size_t data_size, uint32_t width,
                  uint32_t height, uint32_t step = 0);

/**
 * @brief Decode a JPEG buffer to an owned BGR image.
 * @param[in] data Source JPEG bytes.
 * @param[in] data_size Available source bytes.
 * @return Owned three-channel BGR image.
 * @throws std::runtime_error if the source buffer is invalid or cannot be
 * decoded.
 */
cv::Mat JpegToBgr(const uint8_t* data, size_t data_size);

/**
 * @brief Convert packed BGR/RGB bytes to a tightly packed BGR image.
 * @param[in] data Source packed-color bytes.
 * @param[in] data_size Available source bytes.
 * @param[in] width Source width in pixels.
 * @param[in] height Source height in pixels.
 * @param[in] step Source row stride, or zero for tightly packed rows.
 * @param[in] input_is_rgb Convert RGB channel order to BGR when true.
 * @return Owned three-channel BGR image.
 * @throws std::runtime_error if dimensions, stride, or source storage are
 * invalid.
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
  /**
   * @brief Create preprocessing for one HBM-discovered Vision profile.
   * @param[in] profile Validated square model canvas and patch dimensions.
   */
  explicit ImagePreprocessor(VisionProfile profile);
  /**
   * @brief Resize, pad, normalize, and tile a BGR image for Vision.
   * @param[in] bgr Non-empty three-channel source image.
   * @return FP16 Vision patches and source-coordinate transform.
   * @throws std::invalid_argument if the source is empty or not three-channel.
   * @throws std::logic_error if generated patch storage violates the profile.
   */
  PreparedImage Prepare(const cv::Mat& bgr) const;

 private:
  VisionProfile profile_;
};

}  // namespace locateanything
