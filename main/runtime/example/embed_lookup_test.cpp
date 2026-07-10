// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Unit test for EmbedLookup — opens embed_tokens.bin, gathers a few
// token IDs, prints the first few fp16 values of each row so we can
// eyeball correctness against a known reference (e.g. token 0 should
// be the <pad>/<bos> row, token 151665 is image_token_index, etc.).
//
// Build (on S600):
//   cd main/runtime/build && make embed_lookup_test
//   ./embed_lookup_test <path/to/embed_tokens.bin>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include "locateanything_runtime/embed_lookup.hpp"

namespace rt = locateanything_runtime;

namespace {

// Decode one fp16 bit pattern to float (same as vision_dummy_test).
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

void PrintRow(const char *label, const uint16_t *row, int32_t hidden_dim, int32_t show_n) {
  std::printf("  %-18s first %d:", label, show_n);
  for (int32_t i = 0; i < show_n; ++i) {
    std::printf(" %.4f", Fp16ToFloat(row[i]));
  }
  std::printf("\n");
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <embed_tokens.bin>\n", argv[0]);
    return 1;
  }
  constexpr int32_t kVocab = 152681;
  constexpr int32_t kHidden = 2048;

  rt::EmbedLookup embed;
  if (!embed.Open(argv[1], kVocab, kHidden)) {
    std::fprintf(stderr, "[FAIL] Open\n");
    return 2;
  }
  std::printf("[ok] Open: vocab=%d hidden=%d\n", embed.VocabSize(), embed.HiddenDim());

  // Gather a handful of interesting token IDs:
  //   0           — <pad>/<bos> row
  //   151643      — <bos> (Qwen2)
  //   151665      — image_token_index (LA special)
  //   151677      — coord_start_token_id (first coord <0>)
  //   152678      — null_token_id
  //   152679      — switch_token_id
  //   152680      — last valid (vocab-1)
  const int32_t ids[] = {0, 151643, 151665, 151677, 152678, 152679, 152680};
  const int32_t n = sizeof(ids) / sizeof(ids[0]);
  std::vector<uint16_t> rows(static_cast<size_t>(n) * kHidden);
  embed.Gather(ids, n, rows.data());

  const char *labels[] = {"tok_0_pad", "tok_151643_bos", "tok_151665_img",
                          "tok_151677_coord0", "tok_152678_null",
                          "tok_152679_switch", "tok_152680_last"};
  for (int32_t i = 0; i < n; ++i) {
    const uint16_t *row = rows.data() + static_cast<size_t>(i) * kHidden;
    PrintRow(labels[i], row, kHidden, 6);
  }

  // Out-of-range fallback: id -1 and id 999999 should both map to row 0.
  const int32_t oob_ids[] = {-1, 999999};
  std::vector<uint16_t> oob_rows(2 * kHidden);
  embed.Gather(oob_ids, 2, oob_rows.data());
  const uint16_t *row0 = rows.data();
  bool ok = true;
  for (int32_t i = 0; i < 2; ++i) {
    const uint16_t *r = oob_rows.data() + static_cast<size_t>(i) * kHidden;
    if (std::memcmp(r, row0, kHidden * sizeof(uint16_t)) != 0) {
      ok = false;
      std::printf("[FAIL] OOB id %d did not fall back to row 0\n", oob_ids[i]);
    }
  }
  if (ok) {
    std::printf("[ok] OOB fallback -> row 0 verified\n");
  }

  std::printf("[verdict] embed_lookup test %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
