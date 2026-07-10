// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Attention mask builder for the LocateAnything language hbm.
//
// Produces the `input_2` mask tensor language.hbm expects:
//   prefill: (1, chunk_size, cache_len) fp16   e.g. (1, 256, 1024)
//   decode:  (1, q_len,      cache_len) fp16   e.g. (1, 6,   1024)
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

#include <cstdint>
#include <vector>

namespace locateanything_runtime {

struct AttentionMask {
  std::vector<int32_t> shape;   // [1, q_len, cache_len]
  std::vector<uint16_t> data;   // fp16 bit patterns, row-major
};

// Build a PBD-aware causal mask for one prefill or decode step.
//
//   q_len       : number of query positions this step (256 prefill, 6 decode)
//   cache_len   : total KV cache length (1024). The first `past_len` slots
//                 are already-filled history; the rest are future padding.
//   past_len    : number of tokens already in the cache before this step
//                 (0 for cold-start prefill, grows by q_len each step).
//   block_size  : PBD block size (6 for LA). Only the decode step uses the
//                 bidirectional block tweak; prefill passes block_size=0 to
//                 skip it.
//   mask_value  : fp16 bit pattern for "masked" (-32768 for LA).
//   causal_attn : if true, keep strict causal (no PBD bidirectional block).
//
// The returned mask covers the full cache_len columns; rows correspond to
// the q_len query positions. History slots [0, past_len) are visible to
// every query (they're already committed). Within the q_len query window
// itself, the standard causal pattern applies, plus the PBD block tweak
// for the last block_size positions when not causal_attn.
bool BuildAttentionMask(int32_t q_len,
                        int32_t cache_len,
                        int32_t past_len,
                        int32_t block_size,
                        uint16_t mask_value_fp16,
                        bool causal_attn,
                        AttentionMask *out);

// Encode a float as an IEEE 754 binary16 bit pattern. Used to turn the
// LA mask_value (-32768.0f) into the fp16 bits the hbm expects. Also
// exported so callers can pass any float they like.
uint16_t FloatToFp16Bits(float f);

}  // namespace locateanything_runtime
