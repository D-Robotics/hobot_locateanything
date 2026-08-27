#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "metrics.hpp"
#include "processing/image.hpp"

namespace locateanything {

class Tokenizer;

/** One model-detected rectangular region in source-image coordinates. */
struct Detection {
  std::string label;
  std::array<int32_t, 4> normalized_box{};
  std::array<float, 4> box{};
};

/** One model-predicted point in source-image coordinates. */
struct Point {
  std::string label;
  std::array<int32_t, 2> normalized_point{};
  std::array<float, 2> point{};
};

/** Structured LocateAnything result before ROS message conversion. */
struct Prediction {
  std::vector<Detection> detections;
  std::vector<Point> points;
};

class Postprocessor {
 public:
  /**
   * @brief Create a postprocessor with a detection NMS IoU threshold.
   * @param[in] nms_iou Duplicate-box suppression threshold in [0, 1].
   * @throws std::invalid_argument if nms_iou is outside [0, 1].
   */
  explicit Postprocessor(float nms_iou = 0.9f);

  /**
   * Parse generated token IDs into boxes and points.
   * @param[in] tokens Generated Language token IDs.
   * @param[in] transform Transform used to restore source-image coordinates.
   * @param[in] tokenizer Tokenizer used to decode reference labels.
   * @param[in] task Normalized LocateAnything task name.
   * @return Structured boxes and points in source-image coordinates.
   * @throws std::invalid_argument if the source-to-canvas transform is invalid.
   * @throws std::runtime_error if label token decoding fails.
   */
  Prediction Parse(const std::vector<int32_t>& tokens,
                   const ImageTransform& transform,
                   const Tokenizer& tokenizer,
                   const std::string& task) const;
  /**
   * @brief Draw boxes and points onto a copy for Console output.
   * @param[in] source Original source image.
   * @param[in] prediction Structured boxes and points.
   * @return Annotated image copy, or an empty image when source is empty.
   */
  cv::Mat Draw(const cv::Mat& source, const Prediction& prediction) const;
  /**
   * @brief Serialize a prediction and its diagnostics as one JSON object.
   * @param[in] prediction Structured boxes and points.
   * @param[in] task Normalized LocateAnything task name.
   * @param[in] stop_reason Language generation terminal reason.
   * @param[in] frame_index Source frame identifier serialized as metadata.
   * @param[in] metrics End-to-end inference metrics.
   * @param[in] pretty Add indentation and line breaks when true.
   * @return One JSON object without a trailing newline.
   */
  std::string ToJson(const Prediction& prediction,
                     const std::string& task,
                     const std::string& stop_reason,
                     uint64_t frame_index,
                     const InferenceMetrics& metrics,
                     bool pretty = false) const;

 private:
  float nms_iou_;
};

}  // namespace locateanything
