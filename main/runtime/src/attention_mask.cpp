// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include "locateanything_runtime/attention_mask.hpp"

#include <cmath>
#include <cstring>

namespace locateanything_runtime {

uint16_t FloatToFp16Bits(float f) {
  // IEEE 754 binary32 -> binary16 conversion (round to nearest, ties to even).
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127;  // unbiased
  uint32_t mant = bits & 0x7fffff;

  if (exp >= 16) {
    // Overflow or inf -> inf
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  if (exp >= -14) {
    // Normal range
    uint32_t h_exp = static_cast<uint32_t>(exp + 15);
    // Round mantissa 23 -> 10 bits (round to nearest, ties to even)
    uint32_t mant10 = mant >> 13;
    uint32_t round_bits = mant & 0x1fff;
    if (round_bits > 0x1000 || (round_bits == 0x1000 && (mant10 & 1))) {
      ++mant10;
      if (mant10 > 0x3ff) {
        mant10 = 0;
        ++h_exp;
        if (h_exp >= 0x1f) {
          return static_cast<uint16_t>(sign | 0x7c00);
        }
      }
    }
    return static_cast<uint16_t>(sign | (h_exp << 10) | mant10);
  }
  if (exp >= -24) {
    // Subnormal
    uint32_t mant_full = mant | 0x800000;  // implicit 1
    int32_t shift = -14 - exp + 13;        // 23-bit mant -> 10-bit subnormal
    if (shift < 32) {
      uint32_t mant10 = mant_full >> shift;
      uint32_t round_bits = mant_full & ((1u << shift) - 1);
      uint32_t halfway = 1u << (shift - 1);
      if (round_bits > halfway || (round_bits == halfway && (mant10 & 1))) {
        ++mant10;
        if (mant10 > 0x3ff) {
          // Rolled into normal
          return static_cast<uint16_t>(sign | (1u << 10));
        }
      }
      return static_cast<uint16_t>(sign | mant10);
    }
  }
  // Underflow -> zero (preserve sign)
  return static_cast<uint16_t>(sign);
}

bool BuildAttentionMask(int32_t q_len,
                        int32_t cache_len,
                        int32_t past_len,
                        int32_t block_size,
                        uint16_t mask_value_fp16,
                        bool causal_attn,
                        AttentionMask *out) {
  if (q_len <= 0 || cache_len <= 0 || past_len < 0 || past_len + q_len > cache_len) {
    return false;
  }
  out->shape = {1, q_len, cache_len};
  out->data.assign(static_cast<size_t>(q_len) * cache_len, mask_value_fp16);

  const uint16_t kAllow = FloatToFp16Bits(0.0f);  // 0x0000

  // History slots [0, past_len) are visible to every query position.
  for (int32_t i = 0; i < q_len; ++i) {
    uint16_t *row = out->data.data() + static_cast<size_t>(i) * cache_len;
    for (int32_t j = 0; j < past_len; ++j) {
      row[j] = kAllow;
    }
  }

  // Within the current query window [past_len, past_len + q_len):
  //   standard causal — query at window-index i can see window-indices 0..i.
  for (int32_t wi = 0; wi < q_len; ++wi) {
    int32_t row_idx = wi;  // absolute row = wi (rows are q_len positions)
    uint16_t *row = out->data.data() + static_cast<size_t>(row_idx) * cache_len;
    for (int32_t wj = 0; wj <= wi; ++wj) {
      int32_t col_idx = past_len + wj;  // absolute col in cache
      if (col_idx < cache_len) {
        row[col_idx] = kAllow;
      }
    }
  }

  // PBD block tweak (decode step, non-causal). Mirror
  // update_causal_mask_for_one_gen_window_2d:
  //   1. last block_size rows × last block_size cols → all allow
  //      (bidirectional within the generation window)
  //   2. last block_size rows, col at [-block_size-1] → masked
  //      (hide the previous round's trailing token)
  // "last" here is relative to the query window: rows [q_len-block_size,
  // q_len), cols [past_len+q_len-block_size, past_len+q_len).
  if (block_size > 0 && !causal_attn && q_len >= block_size) {
    int32_t blk_row_start = q_len - block_size;
    int32_t blk_col_start = past_len + q_len - block_size;
    // (1) bidirectional block
    for (int32_t i = blk_row_start; i < q_len; ++i) {
      uint16_t *row = out->data.data() + static_cast<size_t>(i) * cache_len;
      for (int32_t j = blk_col_start; j < past_len + q_len && j < cache_len; ++j) {
        row[j] = kAllow;
      }
    }
    // (2) mask the previous round's trailing token: col = past_len - 1
    //     (the last committed history slot).
    int32_t prev_trail_col = past_len - 1;
    if (prev_trail_col >= 0) {
      for (int32_t i = blk_row_start; i < q_len; ++i) {
        uint16_t *row = out->data.data() + static_cast<size_t>(i) * cache_len;
        row[prev_trail_col] = mask_value_fp16;
      }
    }
  }

  return true;
}

}  // namespace locateanything_runtime
