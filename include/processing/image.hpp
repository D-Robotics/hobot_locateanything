#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace locateanything {

struct ImageTransform {
  int source_width = 0;
  int source_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int pad_left = 0;
  int pad_top = 0;
  float scale_x = 1.0f;
  float scale_y = 1.0f;
};

struct PreparedImage {
  std::vector<uint16_t> patches;
  ImageTransform transform;
};

class ImagePreprocessor {
 public:
  PreparedImage Prepare(const cv::Mat& bgr) const;
};

}  // namespace locateanything
