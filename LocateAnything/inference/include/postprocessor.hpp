#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "image_preprocessor.hpp"
#include "inference_metrics.hpp"

namespace locateanything {

class Tokenizer;

struct Detection {
  std::string label;
  std::array<int32_t, 4> normalized_box{};
  std::array<float, 4> box{};
};

struct Point {
  std::string label;
  std::array<int32_t, 2> normalized_point{};
  std::array<float, 2> point{};
};

struct Prediction {
  std::vector<Detection> detections;
  std::vector<Point> points;
};

class Postprocessor {
 public:
  explicit Postprocessor(float nms_iou = 0.9f);

  Prediction Parse(const std::vector<int32_t>& tokens,
                   const ImageTransform& transform,
                   const Tokenizer& tokenizer,
                   const std::string& task) const;
  cv::Mat Draw(const cv::Mat& source, const Prediction& prediction) const;
  std::string ToJson(const Prediction& prediction,
                     const std::string& task,
                     const std::string& stop_reason,
                     uint64_t frame_index,
                     const InferenceMetrics& metrics) const;

 private:
  float nms_iou_;
};

}  // namespace locateanything
