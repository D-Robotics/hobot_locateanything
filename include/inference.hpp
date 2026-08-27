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
  std::string generation_mode = "hybrid";
  int32_t max_new_tokens = 4096;
  uint32_t vision_backend_mask = 15;
  uint32_t language_backend_mask = 15;
  float nms_iou = 0.9f;
};

/** Read-only model canvas discovered from the loaded Vision HBM. */
struct ModelCanvas {
  int32_t width = 0;
  int32_t height = 0;
  int32_t visual_tokens = 0;
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

/** Move-only output of Prompt preparation, image preprocessing, and Vision. */
class PreparedInference {
 public:
  /** @brief Create an empty prepared-inference handle. */
  PreparedInference();
  /** @brief Release prepared frame state. */
  ~PreparedInference();
  /**
   * @brief Move-construct a prepared frame handle.
   * @param other Prepared state whose ownership is transferred.
   */
  PreparedInference(PreparedInference&& other) noexcept;
  /**
   * @brief Move-assign a prepared frame handle.
   * @param other Prepared state whose ownership is transferred.
   * @return This handle after replacing its previous state.
   */
  PreparedInference& operator=(PreparedInference&& other) noexcept;
  PreparedInference(const PreparedInference&) = delete;
  PreparedInference& operator=(const PreparedInference&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  /**
   * @brief Take ownership of shared-core prepared frame state.
   * @param[in] impl Prepared state created by InferenceSession.
   */
  explicit PreparedInference(std::unique_ptr<Impl> impl);
  friend class InferenceSession;
};

class InferenceSession {
 public:
  /**
   * @brief Create an uninitialized session with explicit runtime settings.
   * @param[in] options Model paths, generation settings, and backend masks.
   * @throws std::invalid_argument if the NMS threshold is outside [0, 1].
   */
  explicit InferenceSession(InferenceOptions options);
  /** Release the loaded HBM sessions and host-side runtime state. */
  ~InferenceSession();
  /**
   * @brief Move an initialized or uninitialized session to a new owner.
   * @param other Session whose ownership is transferred.
   */
  InferenceSession(InferenceSession&& other) noexcept;
  /**
   * @brief Move-assign a session to a new owner.
   * @param other Session whose ownership is transferred.
   * @return This session after replacing its previous state.
   */
  InferenceSession& operator=(InferenceSession&& other) noexcept;
  InferenceSession(const InferenceSession&) = delete;
  InferenceSession& operator=(const InferenceSession&) = delete;

  /**
   * Load and validate the Vision HBM, Language HBM, tokenizer, and embeddings.
   * @param[in] progress_callback Optional callback invoked for each loading
   * stage.
   * @throws std::invalid_argument if generation settings are invalid.
   * @throws std::runtime_error if an asset is missing or an HBM contract is
   * incompatible.
   */
  void Initialize(
      const std::function<void(const std::string&)>& progress_callback = {});
  /**
   * @brief Return the model canvas discovered during initialization.
   * @return Square width, height, and visual-token count.
   * @throws std::logic_error if the session has not been initialized.
   */
  ModelCanvas model_canvas() const;
  /**
   * Run one image through preprocessing, Vision, Language, and postprocessing.
   * @param[in] bgr Source image in non-empty three-channel BGR format.
   * @param[in] command LocateAnything task command such as '/detect person'.
   * @param[in] frame_index Source frame identifier included in JSON when
   * serialization is requested.
   * @param[in] output_options Select optional annotated image and JSON output.
   * @return Structured prediction, generated tokens, stop reason, and metrics.
   * @throws std::logic_error if the session is not initialized.
   * @throws std::invalid_argument if the command or source image is invalid.
   * @throws std::runtime_error if model execution or output processing fails.
   */
  InferenceOutput Infer(const cv::Mat& bgr, const std::string& command,
                        uint64_t frame_index = 0,
                        InferenceOutputOptions output_options = {});
  /**
   * Run multiple independent queries against one shared Vision result.
   * @param[in] bgr Source image in non-empty three-channel BGR format.
   * @param[in] commands Task commands with the same LocateAnything task prefix.
   * @param[in] frame_index Source frame identifier included in JSON when
   * serialization is requested.
   * @param[in] output_options Select optional annotated image and JSON output.
   * @return Merged predictions and aggregate timings for all queries.
   * @throws std::logic_error if the session is not initialized.
   * @throws std::invalid_argument if commands are empty, incompatible, or the
   * source image is invalid.
   * @throws std::runtime_error if model execution or output processing fails.
   */
  InferenceOutput InferQueries(
      const cv::Mat& bgr, const std::vector<std::string>& commands,
      uint64_t frame_index = 0,
      InferenceOutputOptions output_options = {});

  /**
   * @brief Prepare one frame through Prompt caching, preprocessing, and Vision.
   * @param[in] bgr Source image in non-empty three-channel BGR format.
   * @param[in] command LocateAnything task command.
   * @return Move-only state ready for Complete().
   * @throws std::logic_error if the session is not initialized.
   * @throws std::invalid_argument if the command or source image is invalid.
   * @throws std::runtime_error if prompt encoding or Vision execution fails.
   */
  PreparedInference Prepare(const cv::Mat& bgr, const std::string& command);
  /**
   * @brief Prepare compatible queries while sharing one Vision result.
   * @param[in] bgr Source image in non-empty three-channel BGR format.
   * @param[in] commands Non-empty commands sharing one LocateAnything task.
   * @return Move-only state ready for Complete().
   * @throws std::logic_error if the session is not initialized.
   * @throws std::invalid_argument if commands are empty, incompatible, or the
   * source image is invalid.
   * @throws std::runtime_error if prompt encoding or Vision execution fails.
   */
  PreparedInference PrepareQueries(
      const cv::Mat& bgr, const std::vector<std::string>& commands);
  /**
   * @brief Complete prepared state through Language and postprocessing.
   * @param[in] prepared Move-only state produced by this session.
   * @param[in] frame_index Source frame identifier included in JSON when
   * serialization is requested.
   * @param[in] output_options Select optional annotated image and JSON output.
   * @return Structured prediction, generated tokens, and metrics.
   * @throws std::invalid_argument if prepared state is empty or belongs to a
   * different session.
   * @throws std::runtime_error if Language execution or output processing
   * fails.
   */
  InferenceOutput Complete(
      PreparedInference prepared, uint64_t frame_index = 0,
      InferenceOutputOptions output_options = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
