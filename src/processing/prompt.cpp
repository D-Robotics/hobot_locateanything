#include "processing/prompt.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>

namespace locateanything {
namespace {

/**
 * @brief Remove leading and trailing whitespace from a task command.
 * @param[in] value Raw command text.
 * @return Trimmed command text.
 */
std::string Trim(std::string value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char item) {
    return std::isspace(item) != 0;
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char item) {
    return std::isspace(item) != 0;
  }).base();
  return first < last ? std::string(first, last) : std::string{};
}

/**
 * @brief Extract and validate the argument after a public command prefix.
 * @param[in] command Complete normalized command.
 * @param[in] prefix Matched command prefix.
 * @return Non-empty argument without a trailing period.
 * @throws std::invalid_argument if no argument remains.
 */
std::string Argument(const std::string& command, const std::string& prefix) {
  std::string value = Trim(command.substr(prefix.size()));
  if (!value.empty() && value.back() == '.') value.pop_back();
  value = Trim(value);
  if (value.empty()) throw std::invalid_argument(prefix + " requires an argument");
  return value;
}

/**
 * @brief Convert comma-separated categories into model category markup.
 * @param[in] value User-provided category list.
 * @return Non-empty `</c>`-delimited model text.
 * @throws std::invalid_argument if the list contains no non-empty category.
 */
std::string Categories(const std::string& value) {
  std::string result;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t end = value.find(',', start);
    const std::string item = Trim(value.substr(start, end - start));
    if (!item.empty()) {
      if (!result.empty()) result += "</c>";
      result += item;
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (result.empty()) throw std::invalid_argument("at least one category is required");
  return result;
}

/**
 * @brief Check whether a normalized command belongs to a task prefix.
 * @param[in] value Normalized complete command.
 * @param[in] command Task prefix.
 * @return True for an exact match or a space-delimited argument.
 */
bool IsCommand(const std::string& value, const std::string& command) {
  return value == command || value.rfind(command + " ", 0) == 0;
}

/**
 * @brief Parse a public command without adding model-specific image tokens.
 * @param[in] command User-facing LocateAnything task command.
 * @return Normalized task and instruction text.
 * @throws std::invalid_argument if the command is empty, unsupported, or
 * lacks a required argument.
 */
Prompt ParseCommand(const std::string& command) {
  const std::string raw = Trim(command);
  if (raw.empty()) throw std::invalid_argument("prompt must not be empty");

  if (raw == "/text") {
    return {"text_ocr", "Detect all the text in box format.", {}};
  }
  if (IsCommand(raw, "/detect")) {
    return {"object_detection",
            "Locate all the instances that matches the following description: " +
                Categories(Argument(raw, "/detect")) + ".",
            {}};
  }
  if (IsCommand(raw, "/layout")) {
    return {"layout_grounding",
            "Detect all the objects in the image that belong to the category set: " +
                Categories(Argument(raw, "/layout")) + ".",
            {}};
  }
  if (IsCommand(raw, "/ground_single")) {
    return {"referring_comprehension_single",
            "Locate a single instance that matches the following description: " +
                Argument(raw, "/ground_single") + ".",
            {}};
  }
  if (IsCommand(raw, "/ground_text")) {
    return {"text_ocr_grounding",
            "Please locate the text referred as " +
                Argument(raw, "/ground_text") + ".",
            {}};
  }
  if (IsCommand(raw, "/ground")) {
    return {"referring_comprehension",
            "Locate all the instances that match the following description: " +
                Argument(raw, "/ground") + ".",
            {}};
  }
  if (IsCommand(raw, "/gui_box")) {
    return {"gui_grounding_box",
            "Locate the region that matches the following description: " +
                Argument(raw, "/gui_box") + ".",
            {}};
  }
  if (IsCommand(raw, "/gui")) {
    return {"gui_grounding", "Point to: " + Argument(raw, "/gui") + ".", {}};
  }
  if (IsCommand(raw, "/point")) {
    return {"point_localization",
            "Point to: " + Argument(raw, "/point") + ".", {}};
  }
  throw std::invalid_argument("unsupported LocateAnything task command");
}

}  // namespace

/**
 * @brief Create a prompt builder using the HBM-discovered visual-token count.
 * @param[in] profile Validated Vision profile supplying visual token count.
 */
PromptBuilder::PromptBuilder(const VisionProfile& profile)
    : visual_tokens_(profile.visual_token_count()) {}

/**
 * @brief Validate one public task command without constructing image tokens.
 * @param[in] command User-facing LocateAnything command.
 * @throws std::invalid_argument if the command is empty, unsupported, or lacks
 *         a required query.
 */
void PromptBuilder::Validate(const std::string& command) {
  (void)ParseCommand(command);
}

/**
 * @brief Parse a command and construct its complete model-ready chat prompt.
 * @param[in] command User-facing LocateAnything command.
 * @return Normalized task metadata and a prompt containing the exact number of
 *         HBM-required image-context tokens.
 * @throws std::invalid_argument if the command is invalid.
 */
Prompt PromptBuilder::Build(const std::string& command) const {
  Prompt output = ParseCommand(command);
  std::string image = "<image 1><img>";
  for (int index = 0; index < visual_tokens_; ++index) image += "<IMG_CONTEXT>";
  image += "</img>";
  output.model_input =
      "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
      "<|im_start|>user\n" +
      image + output.normalized +
      "<|im_end|>\n<|im_start|>assistant\n";
  return output;
}

}  // namespace locateanything
