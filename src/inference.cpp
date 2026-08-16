#include "inference.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iterator>
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

/**
 * @brief Expand comma-separated queries for tasks using independent prompts.
 * @param command One public LocateAnything task command.
 * @return One or more commands sharing the original task prefix.
 */
std::vector<std::string> QueryCommands(const std::string& command) {
  static const std::string query_tasks[] = {
      "/ground_single", "/ground_text", "/gui_box",
      "/ground",        "/gui",         "/point"};
  const auto task = std::find_if(
      std::begin(query_tasks), std::end(query_tasks),
      [&](const std::string& item) {
        return command == item || command.rfind(item + " ", 0) == 0;
      });
  if (task == std::end(query_tasks)) return {command};

  const size_t argument_start = command.find_first_not_of(" \t", task->size());
  if (argument_start == std::string::npos) return {command};
  const std::string argument = command.substr(argument_start);
  std::vector<std::string> queries;
  size_t start = 0;
  while (start <= argument.size()) {
    const size_t end = argument.find(',', start);
    const size_t item_begin = argument.find_first_not_of(" \t", start);
    const size_t item_limit = end == std::string::npos ? argument.size() : end;
    if (item_begin != std::string::npos && item_begin < item_limit) {
      const size_t item_end = argument.find_last_not_of(" \t", item_limit - 1);
      queries.push_back(*task + " " +
                        argument.substr(item_begin, item_end - item_begin + 1));
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (queries.empty()) {
    throw std::invalid_argument(*task + " requires at least one query");
  }
  return queries;
}

/**
 * @brief Add one Language run to aggregate metrics.
 * @param source Metrics produced by one query.
 * @param target Aggregate metrics for the complete inference request.
 */
void AddLanguageMetrics(const LanguageMetrics& source,
                        LanguageMetrics* target) {
  target->prompt_tokens += source.prompt_tokens;
  target->generated_tokens += source.generated_tokens;
  target->pbd_calls += source.pbd_calls;
  target->pbd_accepted_tokens += source.pbd_accepted_tokens;
  target->ar_calls += source.ar_calls;
  target->ar_tokens += source.ar_tokens;
  target->prefill_ms += source.prefill_ms;
  target->decode_ms += source.decode_ms;
  target->cache_initialize_ms += source.cache_initialize_ms;
  target->cache_seed_ms += source.cache_seed_ms;
  target->cache_update_ms += source.cache_update_ms;
  target->host_decode_ms += source.host_decode_ms;
  if (target->executed_mode.empty()) {
    target->executed_mode = source.executed_mode;
  } else if (target->executed_mode != source.executed_mode) {
    target->executed_mode = "mixed";
  }
  if (!source.fallback_reason.empty() &&
      target->fallback_reason.find(source.fallback_reason) == std::string::npos) {
    if (!target->fallback_reason.empty()) target->fallback_reason += ',';
    target->fallback_reason += source.fallback_reason;
  }
  for (const GraphTiming& item : source.graph_timings) {
    auto existing = std::find_if(
        target->graph_timings.begin(), target->graph_timings.end(),
        [&](const GraphTiming& value) { return value.graph == item.graph; });
    if (existing == target->graph_timings.end()) {
      target->graph_timings.push_back(item);
      continue;
    }
    existing->calls += item.calls;
    existing->total_ms += item.total_ms;
    existing->input_build_ms += item.input_build_ms;
    existing->buffer_prepare_ms += item.buffer_prepare_ms;
    existing->input_pack_ms += item.input_pack_ms;
    existing->input_flush_ms += item.input_flush_ms;
    existing->bpu_wait_ms += item.bpu_wait_ms;
    existing->submit_ms += item.submit_ms;
    existing->output_flush_ms += item.output_flush_ms;
    existing->output_unpack_ms += item.output_unpack_ms;
    existing->input_bytes += item.input_bytes;
    existing->resident_input_bytes += item.resident_input_bytes;
    existing->output_bytes += item.output_bytes;
  }
  for (const DecodeEventCount& item : source.decode_events) {
    auto existing = std::find_if(
        target->decode_events.begin(), target->decode_events.end(),
        [&](const DecodeEventCount& value) { return value.event == item.event; });
    if (existing == target->decode_events.end()) {
      target->decode_events.push_back(item);
    } else {
      existing->count += item.count;
    }
  }
}

}  // namespace

struct PreparedPromptCache {
  std::string command;
  Prompt prompt;
  std::vector<int32_t> tokens;
};

struct InferenceSession::Impl {
  /**
   * @brief Store runtime options and construct postprocessing state.
   * @param value Explicit shared-core runtime options.
   */
  explicit Impl(InferenceOptions value)
      : options(std::move(value)),
        vision_profile(options.image_width, options.image_height,
                       options.resize_mode, options.letterbox_fill),
        image_preprocessor(vision_profile),
        prompt_builder(vision_profile),
        postprocessor(options.nms_iou) {}

  InferenceOptions options;
  VisionProfile vision_profile;
  ImagePreprocessor image_preprocessor;
  PromptBuilder prompt_builder;
  Tokenizer tokenizer;
  Postprocessor postprocessor;
  VisionEngine vision;
  LanguageEngine language;
  std::mutex state_mutex;
  std::mutex prompt_cache_mutex;
  std::vector<std::shared_ptr<const PreparedPromptCache>> prompt_cache;
  bool initialized = false;
};

struct PreparedInference::Impl {
  const void* owner = nullptr;
  std::chrono::steady_clock::time_point total_started;
  std::vector<std::shared_ptr<const PreparedPromptCache>> prompts;
  std::vector<uint8_t> visual_features_fp16;
  ImageTransform transform;
  cv::Mat source_image;
  InferenceMetrics metrics;
};

PreparedInference::PreparedInference() = default;
PreparedInference::~PreparedInference() = default;
PreparedInference::PreparedInference(PreparedInference&&) noexcept = default;
PreparedInference& PreparedInference::operator=(PreparedInference&&) noexcept =
    default;
PreparedInference::PreparedInference(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

InferenceSession::InferenceSession(InferenceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
InferenceSession::~InferenceSession() = default;
InferenceSession::InferenceSession(InferenceSession&&) noexcept = default;
InferenceSession& InferenceSession::operator=(InferenceSession&&) noexcept = default;

void InferenceSession::Initialize(
    const std::function<void(const std::string&)>& progress_callback) {
  std::lock_guard<std::mutex> lock(impl_->state_mutex);
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
  impl_->vision.Initialize(options.vision_model, options.vision_backend_mask,
                           impl_->vision_profile);
  if (progress_callback) progress_callback("Language HBM");
  impl_->language.Initialize(options.language_model, options.embeddings,
                             options.language_backend_mask);
  impl_->initialized = true;
}

InferenceOutput InferenceSession::Infer(const cv::Mat& bgr,
                                        const std::string& command,
                                        uint64_t frame_index,
                                        InferenceOutputOptions output_options) {
  return InferQueries(bgr, QueryCommands(command), frame_index, output_options);
}

InferenceOutput InferenceSession::InferQueries(
    const cv::Mat& bgr, const std::vector<std::string>& commands,
    uint64_t frame_index, InferenceOutputOptions output_options) {
  return Complete(PrepareQueries(bgr, commands), frame_index, output_options);
}

PreparedInference InferenceSession::Prepare(const cv::Mat& bgr,
                                            const std::string& command) {
  return PrepareQueries(bgr, QueryCommands(command));
}

PreparedInference InferenceSession::PrepareQueries(
    const cv::Mat& bgr, const std::vector<std::string>& commands) {
  {
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    if (!impl_->initialized) {
      throw std::logic_error("inference session is not initialized");
    }
  }
  if (commands.empty()) {
    throw std::invalid_argument("at least one inference query is required");
  }
  auto prepared = std::make_unique<PreparedInference::Impl>();
  prepared->owner = impl_.get();
  prepared->total_started = std::chrono::steady_clock::now();
  const auto preprocess_started = std::chrono::steady_clock::now();
  prepared->prompts.reserve(commands.size());
  for (const std::string& command : commands) {
    std::shared_ptr<const PreparedPromptCache> cached;
    {
      std::lock_guard<std::mutex> lock(impl_->prompt_cache_mutex);
      const auto found = std::find_if(
          impl_->prompt_cache.begin(), impl_->prompt_cache.end(),
          [&](const std::shared_ptr<const PreparedPromptCache>& item) {
            return item->command == command;
          });
      if (found != impl_->prompt_cache.end()) {
        cached = *found;
      } else {
        auto created = std::make_shared<PreparedPromptCache>();
        created->command = command;
        created->prompt = impl_->prompt_builder.Build(command);
        created->tokens = impl_->tokenizer.Encode(created->prompt.model_input);
        const int expected_visual_tokens =
            impl_->vision_profile.visual_token_count();
        if (std::count(created->tokens.begin(), created->tokens.end(),
                       impl_->tokenizer.TokenId("<IMG_CONTEXT>")) !=
            expected_visual_tokens) {
          throw std::runtime_error(
              "prompt does not contain " +
              std::to_string(expected_visual_tokens) + " visual tokens");
        }
        if (impl_->prompt_cache.size() == 16) {
          impl_->prompt_cache.erase(impl_->prompt_cache.begin());
        }
        impl_->prompt_cache.push_back(created);
        cached = std::move(created);
      }
    }
    if (!prepared->prompts.empty() &&
        cached->prompt.task != prepared->prompts.front()->prompt.task) {
      throw std::invalid_argument("inference queries must use the same task");
    }
    prepared->prompts.push_back(std::move(cached));
  }
  PreparedImage image = impl_->image_preprocessor.Prepare(bgr);
  prepared->transform = image.transform;
  prepared->source_image = bgr;
  const auto preprocess_ended = std::chrono::steady_clock::now();

  const auto vision_started = std::chrono::steady_clock::now();
  VisionResult vision = impl_->vision.Infer(std::move(image.patches_fp16));
  const auto vision_ended = std::chrono::steady_clock::now();
  prepared->visual_features_fp16 = std::move(vision.visual_features_fp16);
  prepared->metrics.preprocess_timing = Stage(
      prepared->total_started, preprocess_started, preprocess_ended);
  prepared->metrics.vision_timing =
      Stage(prepared->total_started, vision_started, vision_ended);
  prepared->metrics.preprocess_ms =
      prepared->metrics.preprocess_timing.DurationMs();
  prepared->metrics.vision_ms =
      prepared->metrics.vision_timing.DurationMs();
  return PreparedInference(std::move(prepared));
}

InferenceOutput InferenceSession::Complete(
    PreparedInference prepared, uint64_t frame_index,
    InferenceOutputOptions output_options) {
  if (prepared.impl_ == nullptr || prepared.impl_->owner != impl_.get()) {
    throw std::invalid_argument(
        "prepared inference does not belong to this session");
  }
  PreparedInference::Impl& input = *prepared.impl_;
  const auto language_started = std::chrono::steady_clock::now();
  InferenceOutput output;
  output.metrics = std::move(input.metrics);
  std::vector<LanguageResult> language_results;
  language_results.reserve(input.prompts.size());
  for (const auto& prompt : input.prompts) {
    const LanguageInput language_input{prompt->tokens,
                                       input.visual_features_fp16};
    const bool protect_structure =
        prompt->prompt.task == "object_detection";
    language_results.push_back(impl_->language.Generate(
        language_input, impl_->options.max_new_tokens,
        impl_->options.generation_mode, protect_structure));
    AddLanguageMetrics(language_results.back().metrics,
                       &output.metrics.language);
    if (output.stop_reason.empty()) {
      output.stop_reason = language_results.back().stop_reason;
    } else if (output.stop_reason != language_results.back().stop_reason) {
      output.stop_reason = "mixed";
    }
  }
  const auto language_ended = std::chrono::steady_clock::now();

  output.metrics.language_timing =
      Stage(input.total_started, language_started, language_ended);
  output.metrics.language_ms = output.metrics.language_timing.DurationMs();
  const auto postprocess_started = std::chrono::steady_clock::now();
  for (size_t index = 0; index < language_results.size(); ++index) {
    LanguageResult& language = language_results[index];
    Prediction prediction = impl_->postprocessor.Parse(
        language.token_ids, input.transform, impl_->tokenizer,
        input.prompts[index]->prompt.task);
    output.prediction.detections.insert(
        output.prediction.detections.end(),
        std::make_move_iterator(prediction.detections.begin()),
        std::make_move_iterator(prediction.detections.end()));
    output.prediction.points.insert(
        output.prediction.points.end(),
        std::make_move_iterator(prediction.points.begin()),
        std::make_move_iterator(prediction.points.end()));
    if (!output.generated_text.empty()) output.generated_text += '\n';
    output.generated_text += impl_->tokenizer.Decode(language.token_ids);
    output.generated_token_ids.insert(
        output.generated_token_ids.end(), language.token_ids.begin(),
        language.token_ids.end());
  }
  const auto postprocess_ended = std::chrono::steady_clock::now();
  output.metrics.postprocess_timing =
      Stage(input.total_started, postprocess_started, postprocess_ended);
  output.metrics.postprocess_ms = output.metrics.postprocess_timing.DurationMs();
  if (output_options.render_annotated) {
    output.annotated_image =
        impl_->postprocessor.Draw(input.source_image, output.prediction);
  }
  output.metrics.total_ms = MillisecondsBetween(
      input.total_started, std::chrono::steady_clock::now());
  if (output_options.serialize_json) {
    output.json = impl_->postprocessor.ToJson(
        output.prediction, input.prompts.front()->prompt.task,
        output.stop_reason, frame_index, output.metrics,
        output_options.pretty_json);
  }
  return output;
}

int32_t InferenceSession::LanguageBatchSize() const {
  return impl_ == nullptr ? 1 : impl_->language.BatchSize();
}

std::vector<InferenceOutput> InferenceSession::CompleteBatch(
    std::vector<PreparedInference> prepared,
    const std::vector<uint64_t>& frame_indices,
    InferenceOutputOptions output_options) {
  if (prepared.empty() || prepared.size() > 2 ||
      prepared.size() != frame_indices.size()) {
    throw std::invalid_argument("CompleteBatch requires one or two frames");
  }
  if (prepared.size() == 1 || LanguageBatchSize() != 2) {
    std::vector<InferenceOutput> outputs;
    outputs.reserve(prepared.size());
    for (size_t index = 0; index < prepared.size(); ++index) {
      outputs.push_back(Complete(std::move(prepared[index]),
                                 frame_indices[index], output_options));
    }
    return outputs;
  }
  for (const PreparedInference& item : prepared) {
    if (item.impl_ == nullptr || item.impl_->owner != impl_.get()) {
      throw std::invalid_argument("prepared inference does not belong to this session");
    }
    if (item.impl_->prompts.size() != 1) {
      std::vector<InferenceOutput> outputs;
      outputs.reserve(prepared.size());
      for (size_t index = 0; index < prepared.size(); ++index) {
        outputs.push_back(Complete(std::move(prepared[index]),
                                   frame_indices[index], output_options));
      }
      return outputs;
    }
  }
  const auto task = prepared[0].impl_->prompts.front()->prompt.task;
  if (prepared[1].impl_->prompts.front()->prompt.task != task) {
    std::vector<InferenceOutput> outputs;
    outputs.reserve(prepared.size());
    for (size_t index = 0; index < prepared.size(); ++index) {
      outputs.push_back(Complete(std::move(prepared[index]),
                                 frame_indices[index], output_options));
    }
    return outputs;
  }

  const auto language_started = std::chrono::steady_clock::now();
  std::vector<LanguageInput> language_inputs;
  language_inputs.reserve(2);
  for (const PreparedInference& item : prepared) {
    const PreparedInference::Impl& input = *item.impl_;
    language_inputs.push_back(LanguageInput{
        input.prompts.front()->tokens, input.visual_features_fp16});
  }
  const bool protect_structure = task == "object_detection";
  std::vector<LanguageResult> language_results = impl_->language.GenerateBatch(
      language_inputs, impl_->options.max_new_tokens,
      impl_->options.generation_mode, protect_structure);
  if (language_results.size() != prepared.size()) {
    throw std::runtime_error("Language batch returned an invalid result count");
  }
  const auto language_ended = std::chrono::steady_clock::now();

  std::vector<InferenceOutput> outputs;
  outputs.resize(prepared.size());
  for (size_t index = 0; index < prepared.size(); ++index) {
    PreparedInference::Impl& input = *prepared[index].impl_;
    InferenceOutput& output = outputs[index];
    LanguageResult& language = language_results[index];
    output.metrics = std::move(input.metrics);
    AddLanguageMetrics(language.metrics, &output.metrics.language);
    output.stop_reason = language.stop_reason;
    output.metrics.language_timing =
        Stage(input.total_started, language_started, language_ended);
    output.metrics.language_ms = output.metrics.language_timing.DurationMs();

    const auto postprocess_started = std::chrono::steady_clock::now();
    Prediction prediction = impl_->postprocessor.Parse(
        language.token_ids, input.transform, impl_->tokenizer, task);
    output.prediction = std::move(prediction);
    output.generated_text = impl_->tokenizer.Decode(language.token_ids);
    output.generated_token_ids = std::move(language.token_ids);
    const auto postprocess_ended = std::chrono::steady_clock::now();
    output.metrics.postprocess_timing =
        Stage(input.total_started, postprocess_started, postprocess_ended);
    output.metrics.postprocess_ms = output.metrics.postprocess_timing.DurationMs();
    if (output_options.render_annotated) {
      output.annotated_image =
          impl_->postprocessor.Draw(input.source_image, output.prediction);
    }
    output.metrics.total_ms = MillisecondsBetween(
        input.total_started, std::chrono::steady_clock::now());
    if (output_options.serialize_json) {
      output.json = impl_->postprocessor.ToJson(
          output.prediction, task, output.stop_reason, frame_indices[index],
          output.metrics, output_options.pretty_json);
    }
  }
  return outputs;
}

}  // namespace locateanything
