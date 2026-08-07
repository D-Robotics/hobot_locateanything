#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace locateanything_runtime {

inline std::vector<std::string> BaseLanguageGraphNames() {
  return {"prefill", "decode", "decode_ar"};
}

inline std::vector<std::string> DefaultLanguageGraphNames() {
  std::vector<std::string> names = BaseLanguageGraphNames();
  for (int q_len = 7; q_len <= 12; ++q_len) {
    names.push_back("decode_pbd_q" + std::to_string(q_len));
  }
  for (int q_len = 2; q_len <= 5; ++q_len) {
    names.push_back("decode_ar_q" + std::to_string(q_len));
  }
  return names;
}

struct GraphSetValidation {
  std::vector<std::string> missing;
  std::vector<std::string> unexpected;
  std::vector<std::string> duplicates;

  // Extra graphs are allowed so users can extend the catalog without changing
  // the loader. Missing or duplicate graphs remain actionable errors.
  bool ok() const { return missing.empty() && duplicates.empty(); }
};

inline GraphSetValidation ValidateGraphNames(
    const std::vector<std::string>& expected,
    const std::vector<std::string>& actual) {
  const std::unordered_set<std::string> expected_set(expected.begin(), expected.end());
  std::unordered_set<std::string> actual_set;
  GraphSetValidation result;
  for (const auto& name : actual) {
    if (!actual_set.insert(name).second) result.duplicates.push_back(name);
  }
  for (const auto& name : expected) {
    if (actual_set.find(name) == actual_set.end()) result.missing.push_back(name);
  }
  for (const auto& name : actual_set) {
    if (expected_set.find(name) == expected_set.end()) result.unexpected.push_back(name);
  }
  std::sort(result.unexpected.begin(), result.unexpected.end());
  std::sort(result.duplicates.begin(), result.duplicates.end());
  return result;
}

inline bool HasDefaultLanguageGraphs(const std::vector<std::string>& actual) {
  return ValidateGraphNames(DefaultLanguageGraphNames(), actual).ok();
}

}  // namespace locateanything_runtime
