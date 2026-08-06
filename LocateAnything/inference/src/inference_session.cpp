#include "inference_session.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "image_preprocessor.hpp"
#include "prompt_builder.hpp"
#include "runner_process.hpp"
#include "tokenizer.hpp"

namespace locateanything {
namespace {

namespace fs = std::filesystem;

template <typename T>
void WriteBinary(const fs::path& path, const std::vector<T>& values) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(T)))) {
    throw std::runtime_error("cannot write inference input: " + path.string());
  }
}

std::vector<int32_t> ReadGeneratedTokens(const fs::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot read Language output: " + path.string());
  std::string line;
  while (std::getline(stream, line)) {
    constexpr char prefix[] = "token_ids=";
    if (line.rfind(prefix, 0) != 0) continue;
    std::vector<int32_t> tokens;
    std::stringstream values(line.substr(sizeof(prefix) - 1));
    std::string value;
    while (std::getline(values, value, ',')) {
      if (!value.empty()) tokens.push_back(std::stoi(value));
    }
    return tokens;
  }
  throw std::runtime_error("Language output does not contain token_ids");
}

double Milliseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

struct TemporaryFiles {
  fs::path root;
  ~TemporaryFiles() {
    std::error_code error;
    fs::remove_all(root, error);
  }
};

std::string ProtocolPath(const fs::path& path) {
  const std::string value = path.string();
  if (value.find_first_of("\t\r\n") != std::string::npos) {
    throw std::invalid_argument("inference paths must not contain tabs or newlines");
  }
  return value;
}

void ParseGraphMetrics(const std::string& encoded,
                       LanguageMetrics* metrics) {
  std::stringstream entries(encoded);
  std::string entry;
  while (std::getline(entries, entry, ';')) {
    if (entry.empty()) continue;
    std::stringstream fields(entry);
    std::vector<std::string> values;
    std::string value;
    while (std::getline(fields, value, ',')) values.push_back(value);
    if (values.size() != 7 || values[0].empty()) continue;
    try {
      GraphTiming timing;
      timing.graph = values[0];
      timing.calls = std::stoi(values[1]);
      timing.total_ms = std::stod(values[2]);
      timing.bpu_wait_ms = std::stod(values[3]);
      timing.submit_ms = std::stod(values[4]);
      timing.input_bytes = std::stoull(values[5]);
      timing.output_bytes = std::stoull(values[6]);
      metrics->graph_timings.push_back(std::move(timing));
    } catch (const std::exception&) {
      // Keep prediction output even if one optional diagnostic entry is malformed.
    }
  }
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
  RunnerProcess vision;
  RunnerProcess language;
  bool initialized = false;
  std::atomic<uint64_t> sequence{0};
};

InferenceSession::InferenceSession(InferenceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
InferenceSession::~InferenceSession() = default;
InferenceSession::InferenceSession(InferenceSession&&) noexcept = default;
InferenceSession& InferenceSession::operator=(InferenceSession&&) noexcept = default;

void InferenceSession::Initialize(
    const std::function<void(const std::string&)>& progress_callback) {
  if (impl_->initialized) return;
  const InferenceOptions& options = impl_->options;
  if (options.max_new_tokens <= 0 ||
      (options.generation_mode != "hybrid" && options.generation_mode != "slow")) {
    throw std::invalid_argument("invalid generation configuration");
  }
  for (const std::string* path : {&options.vision_runner, &options.language_runner,
                                  &options.vision_model, &options.language_model,
                                  &options.embeddings}) {
    if (path->empty() || !fs::is_regular_file(*path)) {
      throw std::runtime_error("missing inference asset: " + *path);
    }
  }
  if (!fs::is_directory(options.tokenizer_directory)) {
    throw std::runtime_error("missing tokenizer directory: " +
                             options.tokenizer_directory);
  }
  fs::create_directories(options.temporary_directory);
  impl_->tokenizer.Load(options.tokenizer_directory);
  impl_->vision.Start(
      options.vision_runner,
      {"--model", options.vision_model, "--backend-mask",
       std::to_string(options.vision_backend_mask), "--server"},
      "visual", [&] {
        if (progress_callback) progress_callback("Vision HBM");
      });
  impl_->language.Start(
      options.language_runner,
      {"--model", options.language_model, "--embed", options.embeddings,
       "--backend-mask", std::to_string(options.language_backend_mask), "--server"},
      "language", [&] {
        if (progress_callback) progress_callback("Language HBM");
      });
  impl_->initialized = true;
}

InferenceOutput InferenceSession::Infer(const cv::Mat& bgr,
                                        const std::string& command,
                                        uint64_t frame_index) {
  if (!impl_->initialized) throw std::logic_error("inference session is not initialized");
  const auto total_started = std::chrono::steady_clock::now();
  const auto preprocess_started = std::chrono::steady_clock::now();
  const Prompt prompt = impl_->prompt_builder.Build(command);
  std::vector<int32_t> prompt_tokens = impl_->tokenizer.Encode(prompt.model_input);
  if (std::count(prompt_tokens.begin(), prompt_tokens.end(),
                 impl_->tokenizer.TokenId("<IMG_CONTEXT>")) != 576) {
    throw std::runtime_error("prompt does not contain 576 visual tokens");
  }
  const PreparedImage image = impl_->image_preprocessor.Prepare(bgr);
  const double preprocess_ms = Milliseconds(preprocess_started);

  const std::string request_id = std::to_string(++impl_->sequence);
  TemporaryFiles files{fs::path(impl_->options.temporary_directory) / request_id};
  fs::create_directories(files.root);
  const fs::path vision_input = files.root / "vision_input.f16.bin";
  const fs::path visual_tokens = files.root / "visual_tokens.f16.bin";
  const fs::path token_input = files.root / "prompt_tokens.i32.bin";
  const fs::path language_output = files.root / "generation.txt";
  WriteBinary(vision_input, image.patches);
  WriteBinary(token_input, prompt_tokens);

  const std::vector<std::string> vision_result = impl_->vision.Request(
      request_id, "LAHBM/1\tRUN\t" + request_id + "\t" +
                      ProtocolPath(vision_input) + "\t" + ProtocolPath(visual_tokens));
  const double vision_ms = vision_result.size() > 3 ? std::stod(vision_result[3]) : 0.0;

  const bool protect_structure = prompt.task == "object_detection";
  const auto language_started = std::chrono::steady_clock::now();
  const std::vector<std::string> language_result = impl_->language.Request(
      request_id, "LAHBM/1\tRUN\t" + request_id + "\t" +
                      ProtocolPath(token_input) + "\t" + ProtocolPath(visual_tokens) +
                      "\t" + ProtocolPath(language_output) + "\t" +
                       std::to_string(impl_->options.max_new_tokens) + "\t" +
                       impl_->options.generation_mode + "\t" +
                       (protect_structure ? "1" : "0") + "\t0");
  const double language_ms = Milliseconds(language_started);
  const std::string stop_reason = language_result.size() > 3 ? language_result[3] : "unknown";
  const std::vector<int32_t> generated = ReadGeneratedTokens(language_output);
  InferenceOutput output;
  output.stop_reason = stop_reason;
  output.metrics.preprocess_ms = preprocess_ms;
  output.metrics.vision_ms = vision_ms;
  output.metrics.language_ms = language_ms;
  LanguageMetrics& language = output.metrics.language;
  language.prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
  language.generated_tokens = static_cast<int32_t>(generated.size());
  if (language_result.size() > 6) language.prefill_ms = std::stod(language_result[6]);
  if (language_result.size() > 7) language.decode_ms = std::stod(language_result[7]);
  if (language_result.size() > 8) language.executed_mode = language_result[8];
  if (language_result.size() > 9) language.fallback_reason = language_result[9];
  if (language_result.size() > 10) language.pbd_calls = std::stoi(language_result[10]);
  if (language_result.size() > 11) {
    language.pbd_accepted_tokens = std::stoi(language_result[11]);
  }
  if (language_result.size() > 12) language.ar_calls = std::stoi(language_result[12]);
  if (language_result.size() > 13) language.ar_tokens = std::stoi(language_result[13]);
  if (language_result.size() > 14) ParseGraphMetrics(language_result[14], &language);
  if (language_result.size() > 15) language.cache_update_ms = std::stod(language_result[15]);
  if (language_result.size() > 16) language.host_decode_ms = std::stod(language_result[16]);
  output.prediction = impl_->postprocessor.Parse(
      generated, image.transform, impl_->tokenizer, prompt.task);
  output.annotated_image = impl_->postprocessor.Draw(bgr, output.prediction);
  output.metrics.total_ms = Milliseconds(total_started);
  output.json = impl_->postprocessor.ToJson(
      output.prediction, prompt.task, stop_reason, frame_index, output.metrics);
  return output;
}

}  // namespace locateanything
