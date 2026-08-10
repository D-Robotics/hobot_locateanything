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

double Milliseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

struct InferenceSession::Impl {
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
  const double preprocess_ms = Milliseconds(preprocess_started);

  VisionResult vision = impl_->vision.Infer(image.patches);
  LanguageInput language_input;
  language_input.prompt_ids = std::move(prompt_tokens);
  language_input.visual_features_fp16 =
      std::move(vision.visual_features_fp16);
  const bool protect_structure = prompt.task == "object_detection";
  const auto language_started = std::chrono::steady_clock::now();
  LanguageResult language = impl_->language.Generate(
      std::move(language_input), impl_->options.max_new_tokens,
      impl_->options.generation_mode, protect_structure);
  const double language_ms = Milliseconds(language_started);

  InferenceOutput output;
  output.stop_reason = language.stop_reason;
  output.metrics.preprocess_ms = preprocess_ms;
  output.metrics.vision_ms = vision.elapsed_ms;
  output.metrics.language_ms = language_ms;
  output.metrics.language = std::move(language.metrics);
  output.prediction = impl_->postprocessor.Parse(
      language.token_ids, image.transform, impl_->tokenizer, prompt.task);
  if (output_options.render_annotated) {
    output.annotated_image = impl_->postprocessor.Draw(bgr, output.prediction);
  }
  output.metrics.total_ms = Milliseconds(total_started);
  if (output_options.serialize_json) {
    output.json = impl_->postprocessor.ToJson(
        output.prediction, prompt.task, output.stop_reason, frame_index,
        output.metrics);
  }
  return output;
}

}  // namespace locateanything
