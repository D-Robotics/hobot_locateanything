#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "locateanything_runtime/hbm_session.hpp"

namespace locateanything_runtime {

struct HybridDecision {
  std::string type;
  std::vector<int32_t> tokens;
  bool switch_to_ar = false;
  bool terminal = false;
};

struct PbdDecodeConfig {
  float temperature = 0.7f;
  float top_p = 0.9f;
  float repetition_penalty = 1.1f;
};

struct PbdDiagnostics {
  bool valid = false;
  std::array<int32_t, 6> retained_tokens{};
  float legacy_box_start = 0.0f;
  float official_box_start = 0.0f;
  float legacy_ref_start = 0.0f;
  float official_ref_start = 0.0f;
  float legacy_end_score = 0.0f;
  float official_end_score = 0.0f;
  std::array<float, 4> legacy_coord_top{};
  std::array<float, 4> official_coord_top{};
};

HybridDecision DecodePbd(const Tensor &logits,
                         const std::vector<int32_t> &generated,
                         const PbdDecodeConfig &config = {},
                         PbdDiagnostics *diagnostics = nullptr);
HybridDecision DecodePbdGreedy(const Tensor &logits,
                               const std::vector<int32_t> &generated);
int32_t DecodeArGreedy(const Tensor &logits,
                       const std::vector<int32_t> &generated);
bool IsCoordinateToken(int32_t token);
std::string RenderLocateAnythingTokens(const std::vector<int32_t> &tokens);

}  // namespace locateanything_runtime
