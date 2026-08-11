#include "processing/prompt.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>

namespace locateanything {
namespace {

constexpr int kVisualTokens = 576;

/**
 * @brief Remove leading and trailing whitespace from a task command.
 * @param value Raw command text.
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
 * @param command Complete normalized command.
 * @param prefix Matched command prefix.
 * @return Non-empty argument without a trailing period.
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
 * @param value User-provided category list.
 * @return Non-empty `</c>`-delimited model text.
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
 * @param value Normalized complete command.
 * @param command Task prefix.
 * @return True for an exact match or a space-delimited argument.
 */
bool IsCommand(const std::string& value, const std::string& command) {
  return value == command || value.rfind(command + " ", 0) == 0;
}

}  // namespace

Prompt PromptBuilder::Build(const std::string& command) const {
  const std::string raw = Trim(command);
  if (raw.empty()) throw std::invalid_argument("prompt must not be empty");

  Prompt output;
  if (raw == "/text") {
    output = {"text_ocr", "Detect all the text in box format.", {}};
  } else if (IsCommand(raw, "/detect")) {
    output = {"object_detection",
              "Locate all the instances that matches the following description: " +
                  Categories(Argument(raw, "/detect")) + ".",
              {}};
  } else if (IsCommand(raw, "/layout")) {
    output = {"layout_grounding",
              "Detect all the objects in the image that belong to the category set: " +
                  Categories(Argument(raw, "/layout")) + ".",
              {}};
  } else if (IsCommand(raw, "/ground_single")) {
    output = {"referring_comprehension_single",
              "Locate a single instance that matches the following description: " +
                  Argument(raw, "/ground_single") + ".",
              {}};
  } else if (IsCommand(raw, "/ground_text")) {
    output = {"text_ocr_grounding",
              "Please locate the text referred as " + Argument(raw, "/ground_text") + ".",
              {}};
  } else if (IsCommand(raw, "/ground")) {
    output = {"referring_comprehension",
              "Locate all the instances that match the following description: " +
                  Argument(raw, "/ground") + ".",
              {}};
  } else if (IsCommand(raw, "/gui_box")) {
    output = {"gui_grounding_box",
              "Locate the region that matches the following description: " +
                  Argument(raw, "/gui_box") + ".",
              {}};
  } else if (IsCommand(raw, "/gui")) {
    output = {"gui_grounding", "Point to: " + Argument(raw, "/gui") + ".", {}};
  } else if (IsCommand(raw, "/point")) {
    output = {"point_localization", "Point to: " + Argument(raw, "/point") + ".", {}};
  } else {
    throw std::invalid_argument("unsupported LocateAnything task command");
  }

  std::string image = "<image 1><img>";
  for (int index = 0; index < kVisualTokens; ++index) image += "<IMG_CONTEXT>";
  image += "</img>";
  output.model_input =
      "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
      "<|im_start|>user\n" +
      image + output.normalized +
      "<|im_end|>\n<|im_start|>assistant\n";
  return output;
}

}  // namespace locateanything
