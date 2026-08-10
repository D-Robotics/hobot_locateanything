#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace locateanything {

struct VisionResult {
  std::vector<uint8_t> visual_features_fp16;
  double elapsed_ms = 0.0;
};

class VisionEngine {
 public:
  VisionEngine();
  ~VisionEngine();
  VisionEngine(VisionEngine&&) noexcept;
  VisionEngine& operator=(VisionEngine&&) noexcept;
  VisionEngine(const VisionEngine&) = delete;
  VisionEngine& operator=(const VisionEngine&) = delete;

  void Initialize(const std::string& model_path, uint32_t backend_mask);
  VisionResult Infer(const std::vector<uint16_t>& patches);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
