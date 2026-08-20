#pragma once

#include <string>

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
   * @brief Parse and normalize a public '/task ...' command.
   * @param command User-facing LocateAnything task command.
   * @return Normalized task name, prompt text, and model-ready input.
   */
  Prompt Build(const std::string& command) const;
};

}  // namespace locateanything
