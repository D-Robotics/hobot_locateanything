#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace locateanything {

/** Visual embedding tensor and measured Vision execution time. */
struct VisionResult {
  std::vector<uint8_t> visual_features_fp16;
  double elapsed_ms = 0.0;
};

/** Vision tensor dimensions discovered from one loaded HBM graph. */
struct VisionModelInfo {
  int32_t canvas_width = 0;
  int32_t canvas_height = 0;
  int32_t patch_size = 0;
  int32_t patch_count = 0;
  int32_t visual_tokens = 0;
  int32_t hidden_size = 0;
};

class VisionEngine {
 public:
  /** Create an uninitialized Vision engine. */
  VisionEngine();
  /** Release the Vision HBM session. */
  ~VisionEngine();
  /**
   * @brief Move-construct a Vision engine.
   * @param other Engine whose runtime state is transferred.
   */
  VisionEngine(VisionEngine&& other) noexcept;
  /**
   * @brief Move-assign a Vision engine.
   * @param other Engine whose runtime state is transferred.
   * @return This engine after replacing its previous state.
   */
  VisionEngine& operator=(VisionEngine&& other) noexcept;
  VisionEngine(const VisionEngine&) = delete;
  VisionEngine& operator=(const VisionEngine&) = delete;

  /**
   * @brief Load and validate the Vision HBM graph.
   * @param[in] model_path Explicit Vision HBM file path.
   * @param[in] backend_mask S600 BPU backend bit mask.
   * @return Square canvas and tensor dimensions discovered from the graph.
   * @throws std::invalid_argument if model_path is empty.
   * @throws std::runtime_error if loading fails or the graph cannot describe a
   * supported square single-image FP16 Vision input.
   */
  VisionModelInfo Initialize(const std::string& model_path,
                             uint32_t backend_mask);
  /**
   * @brief Execute Vision for one prepared FP16 patch tensor.
   * @param[in] patches_fp16 FP16 bytes in the configured static Vision input
   * layout.
   * @return FP16 visual features and measured execution time.
   * @throws std::logic_error if the engine is not initialized.
   * @throws std::invalid_argument if the patch byte count is incompatible.
   * @throws std::runtime_error if graph execution or output validation fails.
   */
  VisionResult Infer(std::vector<uint8_t> patches_fp16);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
