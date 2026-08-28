// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include "runtime/position.hpp"

namespace locateanything_runtime {

/**
 * @brief Build position IDs for one prefill, AR, or PBD execution step.
 * @param[in] q_len Number of query positions to generate.
 * @param[in] past_len Number of positions already committed to the KV cache.
 * @param[in] block_size Number of trailing PBD positions to shift back by one,
 *                       or zero when no PBD adjustment is required.
 * @param[in] is_pbd Apply the LocateAnything PBD position rule when true.
 * @param[out] out Destination tensor with shape [1, 1, q_len] and int32 values.
 * @return True on success; false when q_len is not positive or past_len is
 *         negative. The caller must provide a valid output pointer.
 */
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

  // Mirror upstream _prepare_inputs_in_mtp:
  // position_ids[0, -n_future_tokens:] -= 1. Each trailing position moves
  // back by one but remains distinct; the attention mask provides the
  // bidirectional visibility needed for parallel prediction.
  if (is_pbd && block_size > 0 && q_len >= block_size) {
    int32_t start = q_len - block_size;
    for (int32_t i = start; i < q_len; ++i) {
      out->data[i] -= 1;
    }
  }

  return true;
}

}  // namespace locateanything_runtime
