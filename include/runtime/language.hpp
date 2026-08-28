#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "metrics.hpp"

namespace locateanything {

/** Read-only Prompt tokens and Vision features for Language generation. */
struct LanguageInput {
  const std::vector<int32_t>& prompt_ids;
  const std::vector<uint8_t>& visual_features_fp16;
};

/** Generated tokens, stop reason, and Language timing counters. */
struct LanguageResult {
  std::vector<int32_t> token_ids;
  std::string stop_reason;
  LanguageMetrics metrics;
};

/** Language dimensions discovered and validated during HBM initialization. */
struct LanguageModelInfo {
  int32_t hidden_size = 0;
};

class LanguageEngine {
 public:
  /** Create an uninitialized Language engine. */
  LanguageEngine();
  /** Release HBM graphs, embeddings, and KV-cache state. */
  ~LanguageEngine();
  /**
   * @brief Move-construct a Language engine.
   * @param other Engine whose runtime state is transferred.
   */
  LanguageEngine(LanguageEngine&& other) noexcept;
  /**
   * @brief Move-assign a Language engine.
   * @param other Engine whose runtime state is transferred.
   * @return This engine after replacing its previous state.
   */
  LanguageEngine& operator=(LanguageEngine&& other) noexcept;
  LanguageEngine(const LanguageEngine&) = delete;
  LanguageEngine& operator=(const LanguageEngine&) = delete;

  /**
   * @brief Load and validate Language HBM plus its embedding table.
   * @param[in] model_path Explicit Language HBM file path.
   * @param[in] embeddings_path Explicit fp16 embedding-table file path.
   * @param[in] backend_mask S600 BPU backend bit mask.
   * @return Validated Language dimensions needed by the shared core.
   * Repeated calls return the dimensions loaded by the first successful call
   * and do not replace its model, embeddings, or backend mask.
   * @throws std::invalid_argument if a path is empty.
   * @throws std::runtime_error if loading or graph-layout validation fails.
   */
  LanguageModelInfo Initialize(const std::string& model_path,
                               const std::string& embeddings_path,
                               uint32_t backend_mask);
  /**
   * @brief Generate a LocateAnything response for one prepared image prompt.
   * @param[in] input Prompt token IDs and FP16 Vision features.
   * @param[in] max_new_tokens Hard output-token limit.
   * @param[in] generation_mode Requested `hybrid` or `slow` decoder.
   * @param[in] protect_detection_structure Enable guarded detection fallback.
   * @return Generated tokens, stop reason, and Language metrics.
   * @throws std::logic_error if the engine is not initialized.
   * @throws std::invalid_argument if input or generation settings are invalid.
   * @throws std::runtime_error if graph execution or cache updates fail.
   */
  LanguageResult Generate(const LanguageInput& input, int32_t max_new_tokens,
                          const std::string& generation_mode,
                          bool protect_detection_structure);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
