// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Position IDs builder for the LocateAnything language hbm.
//
// Produces the `input_1` position-ids tensor language.hbm expects:
//   prefill: (1, 1, prefill_len) int32
//   decode:  (1, 1, q_len)       int32
//
// Vendored flow from upstream modeling_locateanything.py::
//   full_position_ids = torch.arange(0, max_possible_len)
//   position_ids = full_position_ids[start_idx : start_idx + q_len]
//   # PBD: shift each position in the last n_future_tokens window back by one;
//   # parallel visibility between those rows is provided by the attention mask
//   position_ids[0, -block_size:] -= 1
//
// For a PBD window starting after past_len committed rows, each of its
// positions is one less than the corresponding causal position. The IDs stay
// distinct; the mask, rather than repeated position IDs, enables parallelism.
// The -1 trick only applies in PBD (non-causal) decode; plain causal AR
// keeps pos_ids strictly increasing.

#pragma once

#include <cstdint>
#include <vector>

namespace locateanything_runtime {

struct PositionIds {
  std::vector<int32_t> shape;   // [1, 1, q_len]
  std::vector<int32_t> data;    // int32, row-major
};

/**
 * @brief Build position IDs for one prefill or decode step.
 * @param[in] q_len Number of query positions.
 * @param[in] past_len Number of cache rows committed before this step.
 * @param[in] block_size Number of trailing PBD positions shifted back by one,
 * or zero to disable the adjustment.
 * @param[in] is_pbd Apply the upstream PBD position adjustment when true.
 * @param[out] out Destination [1, 1, q_len] position tensor.
 * @return True when dimensions are valid and positions were built.
 */
bool BuildPositionIds(int32_t q_len,
                      int32_t past_len,
                      int32_t block_size,
                      bool is_pbd,
                      PositionIds *out);

}  // namespace locateanything_runtime
