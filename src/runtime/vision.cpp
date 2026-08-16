// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include "runtime/vision.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/hbm.hpp"

namespace locateanything {
namespace {

namespace rt = locateanything_runtime;

constexpr int32_t kFp16 = 4;

/** Return the number of scalar elements represented by a tensor shape. */
int64_t ElementCount(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

}  // namespace

struct VisionEngine::Impl {
  locateanything_runtime::HbmSession session;
  std::vector<int32_t> input_shape;
  std::vector<int32_t> output_shape;
  std::mutex mutex;
  bool initialized = false;
};

VisionEngine::VisionEngine() : impl_(std::make_unique<Impl>()) {}
VisionEngine::~VisionEngine() = default;
VisionEngine::VisionEngine(VisionEngine&&) noexcept = default;
VisionEngine& VisionEngine::operator=(VisionEngine&&) noexcept = default;

void VisionEngine::Initialize(const std::string& model_path,
                              uint32_t backend_mask,
                              const VisionProfile& profile) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->initialized) return;
  if (model_path.empty()) {
    throw std::invalid_argument("Vision HBM path is empty");
  }

  const std::vector<int32_t> expected_input_shape{
      1, profile.patch_count(), profile.patch_flat_dim()};
  const std::vector<int32_t> expected_output_shape{
      1, profile.visual_token_count(), VisionProfile::kHiddenSize};
  impl_->session.SetBackendMask(backend_mask);
  const rt::Result loaded = impl_->session.Load(model_path);
  if (!loaded.ok()) {
    throw std::runtime_error("cannot load Vision HBM: " + loaded.message);
  }
  rt::Graph* graph = impl_->session.GetGraph("visual");
  if (graph == nullptr || graph->GetInputShapes().size() != 1 ||
      graph->GetInputDtypes().size() != 1 ||
      graph->GetOutputShapes().size() != 1 ||
      graph->GetOutputDtypes().size() != 1 ||
      graph->GetInputShapes()[0] != expected_input_shape ||
      graph->GetInputDtypes()[0] != kFp16 ||
      graph->GetOutputShapes()[0] != expected_output_shape ||
      graph->GetOutputDtypes()[0] != kFp16) {
    throw std::runtime_error("unexpected Vision HBM graph contract");
  }
  impl_->input_shape = expected_input_shape;
  impl_->output_shape = expected_output_shape;
  impl_->initialized = true;
}

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
