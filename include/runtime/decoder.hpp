#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "runtime/hbm.hpp"

namespace locateanything_runtime {

/** Decision returned by one PBD or AR decode step. */
struct HybridDecision {
  std::string type;
  std::vector<int32_t> tokens;
  bool switch_to_ar = false;
  bool terminal = false;
};

/** Sampling controls used by host-side PBD decoding. */
struct PbdDecodeConfig {
  float temperature = 0.7f;
  float top_p = 0.9f;
  float repetition_penalty = 1.1f;
};

/** Optional diagnostics for comparing legacy and optimized PBD decoding. */
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

/**
 * @brief Decode one six-row PBD window and choose the next hybrid step.
 * @param logits FP16 logits shaped [1, rows, vocab].
 * @param generated Prompt and response token history.
 * @param config Temperature, top-p, and repetition penalty.
 * @param diagnostics Optional legacy-versus-current probability diagnostics.
 * @param row_start First of six rows to decode.
 * @param executor_lane Persistent Host decoder lane, zero for batch-1.
 * @return Accepted tokens and PBD/AR/terminal control decision.
 */
HybridDecision DecodePbd(const Tensor &logits,
                         const std::vector<int32_t> &generated,
                         const PbdDecodeConfig &config = {},
                         PbdDiagnostics *diagnostics = nullptr,
                         int32_t row_start = 0,
                         int32_t executor_lane = 0);
/**
 * @brief Greedily decode one PBD output without sampling.
 * @param logits FP16 logits shaped [1, rows, vocab].
 * @param generated Prompt and response token history.
 * @return Accepted tokens and PBD/AR/terminal control decision.
 */
HybridDecision DecodePbdGreedy(const Tensor &logits,
                               const std::vector<int32_t> &generated);
/**
 * @brief Decode compact multi-row outputs produced by BPU sampling graphs.
 * @param outputs Token IDs and score tensors in fused graph output order.
 * @return Accepted tokens and PBD/AR/terminal control decision.
 */
HybridDecision DecodePbdCompact(const std::vector<Tensor> &outputs);
/**
 * @brief Greedily decode one autoregressive logits row.
 * @param logits FP16 logits shaped [1, 1, vocab].
 * @param generated Prompt and response token history.
 * @return Selected next token ID.
 */
int32_t DecodeArGreedy(const Tensor &logits,
                       const std::vector<int32_t> &generated);
/**
 * @brief Check whether a token is in LocateAnything's coordinate range.
 * @param token Model token ID.
 * @return True for coordinate tokens representing 0 through 1000.
 */
bool IsCoordinateToken(int32_t token);
/**
 * @brief Render model token IDs into LocateAnything output markup.
 * @param tokens Generated token IDs.
 * @return Text representation used by diagnostics and postprocessing.
 */
std::string RenderLocateAnythingTokens(const std::vector<int32_t> &tokens);

}  // namespace locateanything_runtime
