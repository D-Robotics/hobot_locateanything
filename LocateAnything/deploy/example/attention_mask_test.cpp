// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Unit test for attention_mask builder. Verifies the three regions of a
// PBD decode mask against the upstream mask_sdpa_utils.py semantics:
//   1. active history is right-aligned immediately before the query window
//   2. query window itself is causal (i sees 0..i)
//   3. PBD: last block_size×block_size block is bidirectional (all allow)
//   4. PBD: last block_size rows mask the prev-round trailing token col
//
// Prints an ASCII grid for a small case so the pattern is eyeballable.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

#include "locateanything_runtime/attention_mask.hpp"

namespace rt = locateanything_runtime;

namespace {

float Fp16ToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  if (exp == 0) {
    if (mant == 0) return sign ? -0.0f : 0.0f;
    float val = (mant / 1024.0f) * std::ldexp(1.0f, -14);
    return sign ? -val : val;
  }
  if (exp == 31) return std::nanf("");
  float val = std::ldexp(1.0f + mant / 1024.0f, static_cast<int>(exp) - 15);
  return sign ? -val : val;
}

void PrintGrid(const rt::AttentionMask &m, const char *title) {
  std::printf("  %s  shape=[%d,%d,%d]  (0=allow, #=masked)\n", title,
              m.shape[0], m.shape[1], m.shape[2]);
  int32_t rows = m.shape[1];
  int32_t cols = m.shape[2];
  for (int32_t i = 0; i < rows; ++i) {
    std::printf("  q%-2d |", i);
    for (int32_t j = 0; j < cols; ++j) {
      uint16_t v = m.data[static_cast<size_t>(i) * cols + j];
      float f = Fp16ToFloat(v);
      // allow == 0.0f (or -0.0f); masked == mask_value (-32768)
      std::printf("%s", (f == 0.0f) ? " ." : " #");
    }
    std::printf("\n");
  }
}

}  // namespace

int main() {
  const uint16_t kMaskVal = rt::FloatToFp16Bits(-32768.0f);
  std::printf("mask_value -32768 -> fp16 bits = 0x%04x (float %.1f)\n",
              kMaskVal, Fp16ToFloat(kMaskVal));

  // Case A: cold-start prefill, q_len=8, cache_len=16, past_len=0, no PBD.
  // Expect: pure causal in cols 8..15; cols 0..7 stay masked.
  rt::AttentionMask prefill;
  rt::BuildAttentionMask(/*q_len*/ 8, /*cache_len*/ 16, /*past_len*/ 0,
                         /*block_size*/ 0, kMaskVal, /*causal*/ false, &prefill);
  PrintGrid(prefill, "A. prefill (cold, q=8, cache=16, no PBD):");

  bool ok = true;
  auto prefill_allow = [&](int32_t i, int32_t j) {
    return Fp16ToFloat(prefill.data[static_cast<size_t>(i) * 16 + j]) == 0.0f;
  };
  for (int32_t i = 0; i < 8; ++i) {
    for (int32_t j = 0; j < 16; ++j) {
      const bool expected = j >= 8 && j <= 8 + i;
      if (prefill_allow(i, j) != expected) {
        ok = false;
        std::printf("[FAIL] prefill (%d,%d) alignment mismatch\n", i, j);
      }
    }
  }

  // Case B: decode step with PBD. q_len=6 (block_size), cache_len=16,
  // past_len=10 (10 tokens already cached). Expect:
  //   - cols 0..9 all allow (history)
  //   - cols 10..15 the query window, causal + last 6x6 bidirectional block
  //   - col 9 (past_len-1) masked on last 6 rows (prev trailing token)
  rt::AttentionMask decode;
  rt::BuildAttentionMask(/*q_len*/ 6, /*cache_len*/ 16, /*past_len*/ 10,
                         /*block_size*/ 6, kMaskVal, /*causal*/ false, &decode);
  PrintGrid(decode, "B. decode PBD (q=6, cache=16, past=10, block=6):");

  // Verifications for case B.
  int32_t rows = decode.shape[1];
  int32_t cols = decode.shape[2];
  auto allow = [&](int32_t i, int32_t j) {
    return Fp16ToFloat(decode.data[static_cast<size_t>(i) * cols + j]) == 0.0f;
  };
  auto masked = [&](int32_t i, int32_t j) {
    return decode.data[static_cast<size_t>(i) * cols + j] == kMaskVal;
  };

  // 1. history [0, past_len-1) visible to all 6 rows. NOTE col past_len-1
  //    (==9 here) is the prev-round trailing token, which rule (2) masks —
  //    so we exclude it from the "all history visible" check.
  for (int32_t i = 0; i < rows; ++i) {
    for (int32_t j = 0; j < 9; ++j) {  // past_len - 1 = 9
      if (!allow(i, j)) { ok = false; std::printf("[FAIL] history (%d,%d) not allow\n", i, j); }
    }
  }

  // Case C exposes right alignment when history + query do not fill cache.
  // q=2, past=4 => history cols 10..13 and query cols 14..15.
  rt::AttentionMask ar;
  rt::BuildAttentionMask(/*q_len*/ 2, /*cache_len*/ 16, /*past_len*/ 4,
                         /*block_size*/ 0, kMaskVal, /*causal*/ true, &ar);
  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < 16; ++j) {
      const bool actual = Fp16ToFloat(ar.data[static_cast<size_t>(i) * 16 + j]) == 0.0f;
      const bool expected = (j >= 10 && j <= 13) || (j >= 14 && j <= 14 + i);
      if (actual != expected) {
        ok = false;
        std::printf("[FAIL] AR (%d,%d) alignment mismatch\n", i, j);
      }
    }
  }
  // 2. PBD prev-trailing: col 9 masked on last 6 rows
  for (int32_t i = 0; i < rows; ++i) {
    if (!masked(i, 9)) { ok = false; std::printf("[FAIL] prev-trailing (row %d, col 9) not masked\n", i); }
  }
  // 3. PBD bidirectional block: rows 0..5 × cols 10..15 all allow
  for (int32_t i = 0; i < rows; ++i) {
    for (int32_t j = 10; j < 16; ++j) {
      if (!allow(i, j)) { ok = false; std::printf("[FAIL] PBD block (%d,%d) not allow\n", i, j); }
    }
  }

  // Case D: fused PBD with two accepted prefix tokens. q=8 and block=6,
  // so rows 0..1 remain causal, rows 2..7 form the MTP window, and col 9
  // (cache_len-block_size-1) is the previous round's trailing prefix token.
  rt::AttentionMask fused;
  rt::BuildAttentionMask(/*q_len*/ 8, /*cache_len*/ 16, /*past_len*/ 8,
                         /*block_size*/ 6, kMaskVal, /*causal*/ false, &fused);
  auto fused_allow = [&](int32_t i, int32_t j) {
    return Fp16ToFloat(fused.data[static_cast<size_t>(i) * 16 + j]) == 0.0f;
  };
  auto fused_masked = [&](int32_t i, int32_t j) {
    return fused.data[static_cast<size_t>(i) * 16 + j] == kMaskVal;
  };
  if (!fused_allow(1, 9)) {
    ok = false;
    std::printf("[FAIL] fused causal prefix cannot see its own tail\n");
  }
  for (int32_t i = 2; i < 8; ++i) {
    if (!fused_masked(i, 9)) {
      ok = false;
      std::printf("[FAIL] fused previous tail (row %d, col 9) not masked\n", i);
    }
    for (int32_t j = 10; j < 16; ++j) {
      if (!fused_allow(i, j)) {
        ok = false;
        std::printf("[FAIL] fused PBD block (%d,%d) not allow\n", i, j);
      }
    }
  }

  std::printf("[verdict] attention_mask test %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
