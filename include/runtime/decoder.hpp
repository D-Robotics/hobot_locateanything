#pragma once

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

/**
 * @brief Decode one six-row PBD window and choose the next hybrid step.
 * @param[in] logits FP16 logits shaped [1, rows, vocab].
 * @param[in] generated Prompt and response token history.
 * @param[in] config Temperature, top-p, and repetition penalty.
 * @param[in] row_start First of six rows to decode.
 * @return Accepted tokens and PBD/AR/terminal control decision.
 * @throws std::invalid_argument if logits, row_start, or config are invalid.
 */
HybridDecision DecodePbd(const Tensor &logits,
                         const std::vector<int32_t> &generated,
                         const PbdDecodeConfig &config = {},
                         int32_t row_start = 0);
/**
 * @brief Decode one PBD output with the default host probability controls.
 * @param[in] logits FP16 logits shaped [1, rows, vocab].
 * @param[in] generated Prompt and response token history.
 * @return Accepted tokens and PBD/AR/terminal control decision using the
 * default temperature, top-p, and repetition penalty.
 */
HybridDecision DecodePbdGreedy(const Tensor &logits,
                               const std::vector<int32_t> &generated);
/**
 * @brief Greedily decode one autoregressive logits row.
 * @param[in] logits FP16 logits shaped [1, 1, vocab].
 * @param[in] generated Prompt and response token history.
 * @return Selected next token ID.
 * @throws std::invalid_argument if logits are not one FP16 vocabulary row.
 */
int32_t DecodeArGreedy(const Tensor &logits,
                       const std::vector<int32_t> &generated);
/**
 * @brief Check whether a token is in LocateAnything's coordinate range.
 * @param[in] token Model token ID.
 * @return True for coordinate tokens representing 0 through 1000.
 */
bool IsCoordinateToken(int32_t token);
/**
 * @brief Render model token IDs into LocateAnything output markup.
 * @param[in] tokens Generated token IDs.
 * @return Text representation used by diagnostics and postprocessing.
 */
std::string RenderLocateAnythingTokens(const std::vector<int32_t> &tokens);

}  // namespace locateanything_runtime
