// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Attention mask builder for the LocateAnything language hbm.
//
// Produces the `input_2` mask tensor language.hbm expects:
//   prefill: (1, chunk_size, cache_len) fp16   e.g. (1, 1024, 4096)
//   decode:  (1, q_len,      cache_len) fp16   e.g. (1, 6,    4096)
//
// Layout semantics (verified against hbm IO + upstream mask_sdpa_utils.py):
//   - mask[b, i, j] = 0.0           → token i MAY attend to cache slot j
//   - mask[b, i, j] = mask_value   → token i may NOT attend to cache slot j
//   where mask_value = -32768 (fp16 attention + this = effectively -inf).
//
// PBD (Parallel Block Decoding) adds two tweaks on top of a plain causal
// mask, vendored verbatim from upstream
// `mask_sdpa_utils.py::update_causal_mask_for_one_gen_window_2d`:
//   1. The last `block_size` rows × last `block_size` cols block is set
//      to 0.0 (bidirectional — the 6 generated tokens attend to each
//      other, this is the PBD parallelism).
//   2. The last `block_size` rows' column at offset `-block_size-1` is
//      set to mask_value (mask the previous round's trailing token so
//      it isn't recomputed).
// Both tweaks only apply when causal_attn=False (the LA default per
// config.text_config.causal_attn=False).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace locateanything_runtime {

struct AttentionMask {
  std::vector<int32_t> shape;   // [1, q_len, cache_len]
  std::vector<uint16_t> data;   // fp16 bit patterns, row-major
};

/**
 * @brief Build a PBD-aware causal mask for one prefill or decode step.
 *
 * The returned mask covers all cache columns. History and query rows are
 * right-aligned, with the PBD bidirectional block applied only when requested.
 * @param q_len Number of query positions in this step.
 * @param cache_len Total fixed KV-cache length.
 * @param past_len Number of cache rows committed before this step.
 * @param block_size PBD block width, or zero for plain causal behavior.
 * @param mask_value_fp16 Raw fp16 value used for masked positions.
 * @param causal_attn Keep strict causal attention when true.
 * @param out Destination shape and fp16 mask values.
 * @return True when dimensions are valid and the mask was built.
 */
bool BuildAttentionMask(int32_t q_len,
                        int32_t cache_len,
                        int32_t past_len,
                        int32_t block_size,
                        uint16_t mask_value_fp16,
                        bool causal_attn,
                        AttentionMask *out);

/**
 * @brief Build a mask directly into caller-owned reusable fp16 storage.
 * @param q_len Number of query positions in this step.
 * @param cache_len Total fixed KV-cache length.
 * @param past_len Number of cache rows committed before this step.
 * @param block_size PBD block width, or zero for plain causal behavior.
 * @param mask_value_fp16 Raw fp16 value used for masked positions.
 * @param causal_attn Keep strict causal attention when true.
 * @param data Destination containing at least q_len times cache_len values.
 * @param element_count Number of fp16 values available at data.
 * @return True when dimensions and destination storage are valid.
 */
bool BuildAttentionMaskData(int32_t q_len,
                            int32_t cache_len,
                            int32_t past_len,
                            int32_t block_size,
                            uint16_t mask_value_fp16,
                            bool causal_attn,
                            uint16_t *data,
                            size_t element_count);

/**
 * @brief Encode a host float as an IEEE-754 binary16 bit pattern.
 * @param f Host floating-point value.
 * @return Raw fp16 bits consumed by the HBM graph.
 */
uint16_t FloatToFp16Bits(float f);

}  // namespace locateanything_runtime
