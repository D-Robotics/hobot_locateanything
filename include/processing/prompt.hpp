#pragma once

#include <string>

#include "processing/image.hpp"

namespace locateanything {

/** Normalized task command and model-ready prompt text. */
struct Prompt {
  std::string task;
  std::string normalized;
  std::string model_input;
};

class PromptBuilder {
 public:
  /**
   * @brief Create model prompts for one HBM-discovered Vision profile.
   * @param[in] profile Profile supplying the exact visual-token count.
   */
  explicit PromptBuilder(const VisionProfile& profile);
  /**
   * @brief Validate a public command without constructing model image tokens.
   * @param[in] command User-facing LocateAnything task command.
   * @throws std::invalid_argument if the command is empty or unsupported.
   */
  static void Validate(const std::string& command);
  /**
   * @brief Parse and normalize a public '/task ...' command.
   * @param[in] command User-facing LocateAnything task command.
   * @return Normalized task name, prompt text, and model-ready input.
   * @throws std::invalid_argument if the command is empty, unsupported, or
   * lacks a required argument.
   */
  Prompt Build(const std::string& command) const;

 private:
  int visual_tokens_;
};

}  // namespace locateanything
