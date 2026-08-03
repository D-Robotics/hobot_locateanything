#include <cstdio>
#include <string>
#include <vector>

#include "locateanything_runtime/language_graph_set.hpp"

namespace rt = locateanything_runtime;

int main() {
  rt::LanguageGraphSet graph_set = rt::LanguageGraphSet::kFusedDecode;
  bool ok = rt::ParseLanguageGraphSet("standard", &graph_set) &&
            graph_set == rt::LanguageGraphSet::kStandard &&
            rt::ParseLanguageGraphSet("fused_decode", &graph_set) &&
            graph_set == rt::LanguageGraphSet::kFusedDecode &&
            !rt::ParseLanguageGraphSet("partial", &graph_set);

  const std::vector<std::string> standard{"prefill", "decode", "decode_ar"};
  const std::vector<std::string> fused_decode{
      "prefill",        "decode",        "decode_ar",
      "decode_pbd_q7",  "decode_pbd_q8", "decode_pbd_q9",
      "decode_pbd_q10", "decode_pbd_q11", "decode_pbd_q12",
      "decode_ar_q2",   "decode_ar_q3",  "decode_ar_q4",
      "decode_ar_q5"};
  ok = ok && !rt::ParseLanguageGraphSet("", &graph_set) &&
       rt::ExpectedGraphNames(rt::LanguageGraphSet::kStandard) == standard &&
       rt::ExpectedGraphNames(rt::LanguageGraphSet::kFusedDecode) == fused_decode &&
       rt::ValidateGraphSet(rt::LanguageGraphSet::kStandard, standard).ok() &&
       rt::ValidateGraphSet(rt::LanguageGraphSet::kFusedDecode, fused_decode).ok();

  std::vector<std::string> partial = fused_decode;
  partial.pop_back();
  const auto partial_result =
      rt::ValidateGraphSet(rt::LanguageGraphSet::kFusedDecode, partial);
  ok = ok && partial_result.missing.size() == 1 &&
       partial_result.missing[0] == "decode_ar_q5";

  std::vector<std::string> extra = standard;
  extra.push_back("decode_pbd_q7");
  const auto extra_result =
      rt::ValidateGraphSet(rt::LanguageGraphSet::kStandard, extra);
  ok = ok && extra_result.unexpected.size() == 1 &&
       extra_result.unexpected[0] == "decode_pbd_q7";

  std::vector<std::string> duplicate = standard;
  duplicate.push_back("decode");
  const auto duplicate_result =
      rt::ValidateGraphSet(rt::LanguageGraphSet::kStandard, duplicate);
  ok = ok && duplicate_result.duplicates.size() == 1 &&
       duplicate_result.duplicates[0] == "decode";

  std::printf("%s\n", ok ? "[PASS] language_graph_set_test"
                           : "[FAIL] language_graph_set_test");
  return ok ? 0 : 1;
}
