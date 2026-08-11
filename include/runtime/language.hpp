#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "metrics.hpp"

namespace locateanything {

/** Prompt tokens and Vision features consumed by Language generation. */
struct LanguageInput {
  std::vector<int32_t> prompt_ids;
  std::vector<uint8_t> visual_features_fp16;
};

/** Generated tokens, stop reason, and Language timing counters. */
struct LanguageResult {
  std::vector<int32_t> token_ids;
  std::string stop_reason;
  LanguageMetrics metrics;
};

class LanguageEngine {
 public:
  /** Create an uninitialized Language engine. */
  LanguageEngine();
  /** Release HBM graphs, embeddings, and KV-cache state. */
  ~LanguageEngine();
  /** Move-construct a Language engine. */
  LanguageEngine(LanguageEngine&&) noexcept;
  /** Move-assign a Language engine. */
  LanguageEngine& operator=(LanguageEngine&&) noexcept;
  LanguageEngine(const LanguageEngine&) = delete;
  LanguageEngine& operator=(const LanguageEngine&) = delete;

  /**
   * @brief Load and validate Language HBM plus its embedding table.
   * @param model_path Explicit Language HBM file path.
   * @param embeddings_path Explicit fp16 embedding-table file path.
   * @param backend_mask S600 BPU backend bit mask.
   */
  void Initialize(const std::string& model_path,
                  const std::string& embeddings_path,
                  uint32_t backend_mask);
  /**
   * @brief Generate a LocateAnything response for one prepared image prompt.
   * @param input Prompt token IDs and FP16 Vision features.
   * @param max_new_tokens Hard output-token limit.
   * @param generation_mode Requested `hybrid` or `slow` decoder.
   * @param protect_detection_structure Enable guarded detection fallback.
   * @return Generated tokens, stop reason, and Language metrics.
   */
  LanguageResult Generate(LanguageInput input, int32_t max_new_tokens,
                          const std::string& generation_mode,
                          bool protect_detection_structure);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
