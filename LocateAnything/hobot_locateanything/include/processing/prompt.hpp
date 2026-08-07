#pragma once

#include <string>

namespace locateanything {

struct Prompt {
  std::string task;
  std::string normalized;
  std::string model_input;
};

class PromptBuilder {
 public:
  Prompt Build(const std::string& command) const;
};

}  // namespace locateanything
