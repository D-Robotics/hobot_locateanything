#include "inference.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "processing/image.hpp"
#include "processing/prompt.hpp"
#include "processing/tokenizer.hpp"
#include "runtime/language.hpp"
#include "runtime/vision.hpp"

namespace locateanything {
namespace {

namespace fs = std::filesystem;

/**
 * @brief Return elapsed time between two monotonic clock points.
 * @param start Earlier clock point.
 * @param end Later clock point.
 * @return Elapsed milliseconds.
 */
double MillisecondsBetween(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief Convert stage clock bounds into offsets from inference start.
 * @param inference_started Start of the complete inference call.
 * @param stage_started Start of one pipeline stage.
 * @param stage_ended End of one pipeline stage.
 * @return Monotonic stage offsets in milliseconds.
 */
StageTiming Stage(std::chrono::steady_clock::time_point inference_started,
                  std::chrono::steady_clock::time_point stage_started,
                  std::chrono::steady_clock::time_point stage_ended) {
  StageTiming timing;
  timing.start_ms = MillisecondsBetween(inference_started, stage_started);
  timing.end_ms = MillisecondsBetween(inference_started, stage_ended);
  return timing;
}

}  // namespace

struct InferenceSession::Impl {
  /**
   * @brief Store runtime options and construct postprocessing state.
   * @param value Explicit shared-core runtime options.
   */
  explicit Impl(InferenceOptions value)
      : options(std::move(value)), postprocessor(options.nms_iou) {}

  InferenceOptions options;
  ImagePreprocessor image_preprocessor;
  PromptBuilder prompt_builder;
  Tokenizer tokenizer;
  Postprocessor postprocessor;
  VisionEngine vision;
  LanguageEngine language;
  std::mutex inference_mutex;
  bool initialized = false;
};

InferenceSession::InferenceSession(InferenceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
InferenceSession::~InferenceSession() = default;
InferenceSession::InferenceSession(InferenceSession&&) noexcept = default;
InferenceSession& InferenceSession::operator=(InferenceSession&&) noexcept = default;

void InferenceSession::Initialize(
    const std::function<void(const std::string&)>& progress_callback) {
  std::lock_guard<std::mutex> lock(impl_->inference_mutex);
  if (impl_->initialized) return;
  const InferenceOptions& options = impl_->options;
  if (options.max_new_tokens <= 0 ||
      (options.generation_mode != "hybrid" &&
       options.generation_mode != "slow")) {
    throw std::invalid_argument("invalid generation configuration");
  }
  for (const fs::path& path : {fs::path(options.vision_model),
                               fs::path(options.language_model),
                               fs::path(options.embeddings)}) {
    if (path.empty() || !fs::is_regular_file(path)) {
      throw std::runtime_error("missing inference asset: " + path.string());
    }
  }
  if (!fs::is_directory(options.tokenizer_directory)) {
    throw std::runtime_error("missing tokenizer directory: " +
                             options.tokenizer_directory);
  }

  impl_->tokenizer.Load(options.tokenizer_directory);
  if (progress_callback) progress_callback("Vision HBM");
  impl_->vision.Initialize(options.vision_model, options.vision_backend_mask);
  if (progress_callback) progress_callback("Language HBM");
  impl_->language.Initialize(options.language_model, options.embeddings,
                             options.language_backend_mask);
  impl_->initialized = true;
}

InferenceOutput InferenceSession::Infer(const cv::Mat& bgr,
                                        const std::string& command,
                                        uint64_t frame_index,
                                        InferenceOutputOptions output_options) {
  std::lock_guard<std::mutex> lock(impl_->inference_mutex);
  if (!impl_->initialized) {
    throw std::logic_error("inference session is not initialized");
  }
  const auto total_started = std::chrono::steady_clock::now();
  const auto preprocess_started = std::chrono::steady_clock::now();
  const Prompt prompt = impl_->prompt_builder.Build(command);
  std::vector<int32_t> prompt_tokens =
      impl_->tokenizer.Encode(prompt.model_input);
  if (std::count(prompt_tokens.begin(), prompt_tokens.end(),
                 impl_->tokenizer.TokenId("<IMG_CONTEXT>")) != 576) {
    throw std::runtime_error("prompt does not contain 576 visual tokens");
  }
  const PreparedImage image = impl_->image_preprocessor.Prepare(bgr);
  const auto preprocess_ended = std::chrono::steady_clock::now();

  const auto vision_started = std::chrono::steady_clock::now();
  VisionResult vision = impl_->vision.Infer(image.patches);
  const auto vision_ended = std::chrono::steady_clock::now();
  LanguageInput language_input;
  language_input.prompt_ids = std::move(prompt_tokens);
  language_input.visual_features_fp16 =
      std::move(vision.visual_features_fp16);
  const bool protect_structure = prompt.task == "object_detection";
  const auto language_started = std::chrono::steady_clock::now();
  LanguageResult language = impl_->language.Generate(
      std::move(language_input), impl_->options.max_new_tokens,
      impl_->options.generation_mode, protect_structure);
  const auto language_ended = std::chrono::steady_clock::now();

  InferenceOutput output;
  output.stop_reason = language.stop_reason;
  output.metrics.preprocess_timing =
      Stage(total_started, preprocess_started, preprocess_ended);
  output.metrics.vision_timing =
      Stage(total_started, vision_started, vision_ended);
  output.metrics.language_timing =
      Stage(total_started, language_started, language_ended);
  output.metrics.preprocess_ms = output.metrics.preprocess_timing.DurationMs();
  output.metrics.vision_ms = output.metrics.vision_timing.DurationMs();
  output.metrics.language_ms = output.metrics.language_timing.DurationMs();
  output.metrics.language = std::move(language.metrics);
  const auto postprocess_started = std::chrono::steady_clock::now();
  output.prediction = impl_->postprocessor.Parse(
      language.token_ids, image.transform, impl_->tokenizer, prompt.task);
  output.generated_text = impl_->tokenizer.Decode(language.token_ids);
  output.generated_token_ids = std::move(language.token_ids);
  const auto postprocess_ended = std::chrono::steady_clock::now();
  output.metrics.postprocess_timing =
      Stage(total_started, postprocess_started, postprocess_ended);
  output.metrics.postprocess_ms = output.metrics.postprocess_timing.DurationMs();
  if (output_options.render_annotated) {
    output.annotated_image = impl_->postprocessor.Draw(bgr, output.prediction);
  }
  output.metrics.total_ms = MillisecondsBetween(
      total_started, std::chrono::steady_clock::now());
  if (output_options.serialize_json) {
    output.json = impl_->postprocessor.ToJson(
        output.prediction, prompt.task, output.stop_reason, frame_index,
        output.metrics, output_options.pretty_json);
  }
  return output;
}

}  // namespace locateanything
