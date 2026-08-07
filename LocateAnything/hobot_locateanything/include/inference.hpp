#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "metrics.hpp"
#include "processing/postprocess.hpp"

namespace locateanything {

struct InferenceOptions {
  std::string vision_runner;
  std::string language_runner;
  std::string vision_model;
  std::string language_model;
  std::string embeddings;
  std::string tokenizer_directory;
  std::string temporary_directory;
  std::string generation_mode = "hybrid";
  int32_t max_new_tokens = 4096;
  uint32_t vision_backend_mask = 15;
  uint32_t language_backend_mask = 15;
  float nms_iou = 0.9f;
};

struct InferenceOutput {
  Prediction prediction;
  cv::Mat annotated_image;
  std::string json;
  std::string stop_reason;
  InferenceMetrics metrics;
};

class InferenceSession {
 public:
  explicit InferenceSession(InferenceOptions options);
  ~InferenceSession();
  InferenceSession(InferenceSession&&) noexcept;
  InferenceSession& operator=(InferenceSession&&) noexcept;
  InferenceSession(const InferenceSession&) = delete;
  InferenceSession& operator=(const InferenceSession&) = delete;

  void Initialize(
      const std::function<void(const std::string&)>& progress_callback = {});
  InferenceOutput Infer(const cv::Mat& bgr, const std::string& command,
                        uint64_t frame_index = 0);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
