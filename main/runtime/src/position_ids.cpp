// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include "locateanything_runtime/position_ids.hpp"

namespace locateanything_runtime {

bool BuildPositionIds(int32_t q_len,
                      int32_t past_len,
                      int32_t block_size,
                      bool is_pbd,
                      PositionIds *out) {
  if (q_len <= 0 || past_len < 0) {
    return false;
  }
  out->shape = {1, 1, q_len};
  out->data.resize(q_len);

  // base: arange(past_len, past_len + q_len)
  for (int32_t i = 0; i < q_len; ++i) {
    out->data[i] = past_len + i;
  }

  // PBD tweak: pos_ids[-block_size:] -= 1. Mirror upstream
  // _prepare_inputs_in_mtp: position_ids[0, -n_future_tokens:] -= 1.
  // This makes the last `block_size` query positions share the position id
  // of the token immediately preceding them, which is what lets the 6
  // masked tokens be predicted in parallel (they "attend as if" they're
  // all the next-token position relative to the shared prefix).
  if (is_pbd && block_size > 0 && q_len >= block_size) {
    int32_t start = q_len - block_size;
    for (int32_t i = start; i < q_len; ++i) {
      out->data[i] -= 1;
    }
  }

  return true;
}

}  // namespace locateanything_runtime
