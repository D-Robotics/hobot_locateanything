#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace locateanything_runtime {

enum class LanguageGraphSet {
  kStandard,
  kFusedDecode,
};

inline bool ParseLanguageGraphSet(const std::string& value,
                                  LanguageGraphSet* graph_set) {
  if (graph_set == nullptr) return false;
  if (value == "standard") {
    *graph_set = LanguageGraphSet::kStandard;
    return true;
  }
  if (value == "fused_decode") {
    *graph_set = LanguageGraphSet::kFusedDecode;
    return true;
  }
  return false;
}

inline std::vector<std::string> ExpectedGraphNames(LanguageGraphSet graph_set) {
  std::vector<std::string> names{"prefill", "decode", "decode_ar"};
  if (graph_set == LanguageGraphSet::kFusedDecode) {
    for (int q_len = 7; q_len <= 12; ++q_len) {
      names.push_back("decode_pbd_q" + std::to_string(q_len));
    }
    for (int q_len = 2; q_len <= 5; ++q_len) {
      names.push_back("decode_ar_q" + std::to_string(q_len));
    }
  }
  return names;
}

struct GraphSetValidation {
  std::vector<std::string> missing;
  std::vector<std::string> unexpected;
  std::vector<std::string> duplicates;

  bool ok() const {
    return missing.empty() && unexpected.empty() && duplicates.empty();
  }
};

inline GraphSetValidation ValidateGraphSet(
    LanguageGraphSet graph_set, const std::vector<std::string>& actual) {
  const std::vector<std::string> expected = ExpectedGraphNames(graph_set);
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

}  // namespace locateanything_runtime
