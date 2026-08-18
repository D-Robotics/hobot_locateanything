#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "processing/image.hpp"

namespace locateanything {

/** Visual embedding tensor and measured Vision execution time. */
struct VisionResult {
  std::vector<uint8_t> visual_features_fp16;
  double elapsed_ms = 0.0;
};

class VisionEngine {
 public:
  /** Create an uninitialized Vision engine. */
  VisionEngine();
  /** Release the Vision HBM session. */
  ~VisionEngine();
  /** Move-construct a Vision engine. */
  VisionEngine(VisionEngine&&) noexcept;
  /** Move-assign a Vision engine. */
  VisionEngine& operator=(VisionEngine&&) noexcept;
  VisionEngine(const VisionEngine&) = delete;
  VisionEngine& operator=(const VisionEngine&) = delete;

  /**
   * @brief Load and validate the Vision HBM graph.
   * @param model_path Explicit Vision HBM file path.
   * @param backend_mask S600 BPU backend bit mask.
   */
  void Initialize(const std::string& model_path, uint32_t backend_mask,
                  const VisionProfile& profile);
  /**
   * @brief Execute Vision for one prepared FP16 patch tensor.
   * @param patches_fp16 FP16 bytes in the configured static Vision input layout.
   * @return FP16 visual features and measured execution time.
   */
  VisionResult Infer(std::vector<uint8_t> patches_fp16);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
