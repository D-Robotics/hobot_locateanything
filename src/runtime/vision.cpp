// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include "runtime/vision.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/hbm.hpp"

namespace locateanything {
namespace {

namespace rt = locateanything_runtime;

constexpr int32_t kFp16 = 4;
constexpr int32_t kChannels = 3;

/**
 * @brief Multiply tensor dimensions into a scalar element count.
 * @param shape Positive tensor dimensions.
 * @return Number of scalar elements represented by the shape.
 */
int64_t ElementCount(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

/**
 * @brief Compute an exact positive integer square root.
 * @param value Candidate square value.
 * @return Exact positive root, or zero when no integer root exists.
 */
int32_t ExactSquareRoot(int32_t value) {
  if (value <= 0) return 0;
  const int64_t root = static_cast<int64_t>(std::llround(std::sqrt(value)));
  return root > 0 && root * root == value &&
                 root <= std::numeric_limits<int32_t>::max()
             ? static_cast<int32_t>(root)
             : 0;
}

/**
 * @brief Render a tensor shape for startup contract errors.
 * @param shape Tensor dimensions.
 * @return Bracketed comma-delimited shape text.
 */
std::string ShapeText(const std::vector<int32_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) output << ',';
    output << shape[index];
  }
  output << ']';
  return output.str();
}

/**
 * @brief Construct a detailed Vision HBM contract exception.
 * @param model_path Vision HBM path selected by the active configuration.
 * @param reason Human-readable contract failure.
 * @param input_shape Actual Vision input shape when available.
 * @param output_shape Actual Vision output shape when available.
 * @return Runtime error containing the model path and actual tensor details.
 */
std::runtime_error VisionContractError(
    const std::string& model_path, const std::string& reason,
    const std::vector<int32_t>& input_shape = {},
    const std::vector<int32_t>& output_shape = {}) {
  std::ostringstream message;
  message << "Vision HBM contract error: " << reason
          << ", model=" << model_path;
  if (!input_shape.empty()) {
    message << ", input_shape=" << ShapeText(input_shape);
    if (input_shape.size() > 1) {
      message << ", patch_count=" << input_shape[1];
    }
  }
  if (!output_shape.empty()) {
    message << ", output_shape=" << ShapeText(output_shape);
  }
  return std::runtime_error(message.str());
}

}  // namespace

struct VisionEngine::Impl {
  locateanything_runtime::HbmSession session;
  std::vector<int32_t> input_shape;
  std::vector<int32_t> output_shape;
  VisionModelInfo model_info;
  std::mutex mutex;
  bool initialized = false;
};

/**
 * @brief Create an uninitialized Vision engine and private runtime state.
 */
VisionEngine::VisionEngine() : impl_(std::make_unique<Impl>()) {}
/** @brief Release the loaded Vision HBM session and cached model metadata. */
VisionEngine::~VisionEngine() = default;
/**
 * @brief Move loaded Vision runtime state from another engine.
 * @param[in,out] other Engine whose state is transferred.
 */
VisionEngine::VisionEngine(VisionEngine&& other) noexcept = default;
/**
 * @brief Replace this engine with another Vision runtime state.
 * @param[in,out] other Engine whose state is transferred.
 * @return This engine after ownership transfer.
 */
VisionEngine& VisionEngine::operator=(VisionEngine&& other) noexcept = default;

/**
 * @brief Load a Vision HBM and derive its square model canvas from graph IO.
 * @param[in] model_path Vision HBM file path.
 * @param[in] backend_mask S600 BPU backend bit mask used for execution.
 * @return Derived canvas, patch, visual-token, and hidden-size metadata.
 * @throws std::invalid_argument if model_path is empty.
 * @throws std::runtime_error if loading fails or graph shapes/dtypes cannot
 *         describe a supported single-image square FP16 Vision model.
 */
VisionModelInfo VisionEngine::Initialize(const std::string& model_path,
                                         uint32_t backend_mask) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->initialized) return impl_->model_info;
  if (model_path.empty()) {
    throw std::invalid_argument("Vision HBM path is empty");
  }

  impl_->session.SetBackendMask(backend_mask);
  const rt::Result loaded = impl_->session.Load(model_path);
  if (!loaded.ok()) {
    throw VisionContractError(model_path, "cannot load HBM: " + loaded.message);
  }
  rt::Graph* graph = impl_->session.GetGraph("visual");
  if (graph == nullptr || graph->GetInputShapes().size() != 1 ||
      graph->GetInputDtypes().size() != 1 ||
      graph->GetOutputShapes().size() != 1 ||
      graph->GetOutputDtypes().size() != 1) {
    throw VisionContractError(
        model_path, "graph 'visual' must expose exactly one input and output");
  }
  const std::vector<int32_t>& input_shape = graph->GetInputShapes()[0];
  const std::vector<int32_t>& output_shape = graph->GetOutputShapes()[0];
  if (graph->GetInputDtypes()[0] != kFp16 ||
      graph->GetOutputDtypes()[0] != kFp16) {
    throw VisionContractError(model_path, "input and output must use FP16",
                              input_shape, output_shape);
  }
  if (input_shape.size() != 3 || output_shape.size() != 3 ||
      input_shape[0] != 1 || output_shape[0] != 1 || input_shape[1] <= 0 ||
      input_shape[2] <= 0 || output_shape[1] <= 0 || output_shape[2] <= 0) {
    throw VisionContractError(
        model_path,
        "expected positive single-image tensors [1,patch_count,patch_vector] and "
        "[1,visual_tokens,hidden_size]",
        input_shape, output_shape);
  }
  const int32_t patch_count = input_shape[1];
  const int32_t patch_vector = input_shape[2];
  if (patch_vector % kChannels != 0) {
    throw VisionContractError(
        model_path, "patch_vector is not divisible by three RGB channels",
        input_shape, output_shape);
  }
  const int32_t grid = ExactSquareRoot(patch_count);
  if (grid == 0) {
    throw VisionContractError(
        model_path,
        "patch_count=" + std::to_string(patch_count) +
            " cannot describe a square model canvas",
        input_shape, output_shape);
  }
  const int32_t patch_size =
      ExactSquareRoot(patch_vector / kChannels);
  if (patch_size == 0) {
    throw VisionContractError(
        model_path,
        "patch_vector=" + std::to_string(patch_vector) +
            " cannot describe square RGB patches",
        input_shape, output_shape);
  }
  const int64_t canvas = static_cast<int64_t>(grid) * patch_size;
  if (canvas > std::numeric_limits<int32_t>::max()) {
    throw VisionContractError(model_path, "derived canvas exceeds int32 range",
                              input_shape, output_shape);
  }

  impl_->input_shape = input_shape;
  impl_->output_shape = output_shape;
  impl_->model_info = {static_cast<int32_t>(canvas),
                       static_cast<int32_t>(canvas), patch_size, patch_count,
                       output_shape[1], output_shape[2]};
  impl_->initialized = true;
  return impl_->model_info;
}

/**
 * @brief Execute the loaded Vision graph for one prepared patch tensor.
 * @param[in] patches_fp16 Row-major FP16 patch bytes matching the HBM input.
 * @return FP16 visual features and measured graph execution time.
 * @throws std::logic_error if Initialize has not completed.
 * @throws std::invalid_argument if the input byte count is incompatible.
 * @throws std::runtime_error if graph execution or output validation fails.
 */
VisionResult VisionEngine::Infer(std::vector<uint8_t> patches_fp16) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) {
    throw std::logic_error("Vision engine is not initialized");
  }
  const size_t expected_elements =
      static_cast<size_t>(ElementCount(impl_->input_shape));
  if (patches_fp16.size() != expected_elements * sizeof(uint16_t)) {
    throw std::invalid_argument(
        "Vision input must contain exactly " +
        std::to_string(expected_elements) + " FP16 values");
  }

  rt::Tensor input;
  input.shape = impl_->input_shape;
  input.dtype = kFp16;
  input.data = std::move(patches_fp16);

  std::vector<rt::Tensor> outputs;
  const auto started = std::chrono::steady_clock::now();
  const rt::Result executed =
      impl_->session.ExecuteGraphByName("visual", {input}, &outputs);
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count();
  if (!executed.ok()) {
    throw std::runtime_error("Vision HBM inference failed: " +
                             executed.message);
  }
  if (outputs.size() != 1 || outputs[0].shape != impl_->output_shape ||
      outputs[0].dtype != kFp16 ||
      outputs[0].data.size() !=
          static_cast<size_t>(ElementCount(impl_->output_shape)) *
              sizeof(uint16_t)) {
    throw std::runtime_error("unexpected Vision HBM output contract");
  }

  VisionResult result;
  result.visual_features_fp16 = std::move(outputs[0].data);
  result.elapsed_ms = elapsed_ms;
  return result;
}

}  // namespace locateanything
