#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "metrics.hpp"
#include "processing/postprocess.hpp"

namespace locateanything {

/** Runtime assets and generation settings shared by ROS and Console. */
struct InferenceOptions {
  std::string vision_model;
  std::string language_model;
  std::string embeddings;
  std::string tokenizer_directory;
  int32_t image_width = 672;
  int32_t image_height = 672;
  std::string resize_mode = "letterbox";
  int32_t letterbox_fill = 128;
  std::string generation_mode = "hybrid";
  int32_t max_new_tokens = 4096;
  uint32_t vision_backend_mask = 15;
  uint32_t language_backend_mask = 15;
  float nms_iou = 0.9f;
};

/** Optional presentation outputs requested by a caller. */
struct InferenceOutputOptions {
  // ROS publishes structured targets only; Console opts into presentation files.
  bool render_annotated = false;
  bool serialize_json = false;
  bool pretty_json = false;
};

/** Result of one image inference, including structured output and timings. */
struct InferenceOutput {
  Prediction prediction;
  cv::Mat annotated_image;
  std::string json;
  std::string generated_text;
  std::vector<int32_t> generated_token_ids;
  std::string stop_reason;
  InferenceMetrics metrics;
};

class InferenceSession {
 public:
  /**
   * @brief Create an uninitialized session with explicit runtime settings.
   * @param options Model paths, generation settings, and backend masks.
   */
  explicit InferenceSession(InferenceOptions options);
  /** Release the loaded HBM sessions and host-side runtime state. */
  ~InferenceSession();
  /** Move an initialized or uninitialized session to a new owner. */
  InferenceSession(InferenceSession&&) noexcept;
  /** Move-assign a session to a new owner. */
  InferenceSession& operator=(InferenceSession&&) noexcept;
  InferenceSession(const InferenceSession&) = delete;
  InferenceSession& operator=(const InferenceSession&) = delete;

  /**
   * Load and validate the Vision HBM, Language HBM, tokenizer, and embeddings.
   * @param progress_callback Optional callback invoked for each loading stage.
   * @throws std::exception if an asset or graph contract is invalid.
   */
  void Initialize(
      const std::function<void(const std::string&)>& progress_callback = {});
  /**
   * Run one image through preprocessing, Vision, Language, and postprocessing.
   * @param bgr Source image in non-empty three-channel BGR format.
   * @param command LocateAnything task command such as '/detect person'.
   * @param frame_index Source frame identifier copied into diagnostics/output.
   * @param output_options Select optional annotated image and JSON generation.
   * @return Structured prediction, generated tokens, stop reason, and metrics.
   * @throws std::exception if the session is not initialized or input is invalid.
   */
  InferenceOutput Infer(const cv::Mat& bgr, const std::string& command,
                        uint64_t frame_index = 0,
                        InferenceOutputOptions output_options = {});
  /**
   * Run multiple independent queries against one shared Vision result.
   * @param bgr Source image in non-empty three-channel BGR format.
   * @param commands Task commands with the same LocateAnything task prefix.
   * @param frame_index Source frame identifier copied into diagnostics/output.
   * @param output_options Select optional annotated image and JSON generation.
   * @return Merged predictions and aggregate timings for all queries.
   * @throws std::exception if commands are empty, incompatible, or inference
   * fails.
   */
  InferenceOutput InferQueries(
      const cv::Mat& bgr, const std::vector<std::string>& commands,
      uint64_t frame_index = 0,
      InferenceOutputOptions output_options = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
