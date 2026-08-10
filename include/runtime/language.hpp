#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "metrics.hpp"

namespace locateanything {

struct LanguageInput {
  std::vector<int32_t> prompt_ids;
  std::vector<uint8_t> visual_features_fp16;
};

struct LanguageResult {
  std::vector<int32_t> token_ids;
  std::string stop_reason;
  LanguageMetrics metrics;
};

class LanguageEngine {
 public:
  LanguageEngine();
  ~LanguageEngine();
  LanguageEngine(LanguageEngine&&) noexcept;
  LanguageEngine& operator=(LanguageEngine&&) noexcept;
  LanguageEngine(const LanguageEngine&) = delete;
  LanguageEngine& operator=(const LanguageEngine&) = delete;

  void Initialize(const std::string& model_path,
                  const std::string& embeddings_path,
                  uint32_t backend_mask);
  LanguageResult Generate(LanguageInput input, int32_t max_new_tokens,
                          const std::string& generation_mode,
                          bool protect_detection_structure);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
