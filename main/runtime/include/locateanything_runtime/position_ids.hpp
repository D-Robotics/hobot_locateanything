// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Position IDs builder for the LocateAnything language hbm.
//
// Produces the `input_1` position-ids tensor language.hbm expects:
//   prefill: (1, 1, chunk_size) int32   e.g. (1, 1, 256)
//   decode:  (1, 1, q_len)     int32    e.g. (1, 1, 6)
//
// Vendored flow from upstream modeling_locateanything.py::
//   full_position_ids = torch.arange(0, max_possible_len)
//   position_ids = full_position_ids[start_idx : start_idx + q_len]
//   # PBD: the last n_future_tokens (=block_size) window positions share
//   # the previous token's position id, so they can be predicted in parallel
//   position_ids[0, -block_size:] -= 1
//
// Concretely for LA (block_size=6):
//   prefill (cold, past_len=0):    pos_ids = [0, 1, 2, ..., 255]
//   decode step (past_len=256):    pos_ids = [256, 256, 256, 256, 256, 256]
//                                  (all 6 share 256 because the -1 tweak
//                                   makes [-6:] collapse onto the trailing
//                                   prefill token's id)
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

// Build position IDs for one prefill or decode step.
//
//   q_len      : number of query positions (256 prefill, 6 decode)
//   past_len   : number of tokens already in the cache (0 for cold prefill)
//   block_size : PBD block size (6). Pass 0 to disable the [-block_size:]-=1
//                tweak (used for prefill and for plain AR decode).
//   is_pbd     : if true, apply the pos_ids[-block_size:] -= 1 tweak
//                (PBD decode only).
//
// Returns the (1, 1, q_len) int32 tensor.
bool BuildPositionIds(int32_t q_len,
                      int32_t past_len,
                      int32_t block_size,
                      bool is_pbd,
                      PositionIds *out);

}  // namespace locateanything_runtime
