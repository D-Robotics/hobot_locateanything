// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Unit test for position_ids builder.

#include <cstdint>
#include <cstdio>
#include "locateanything_runtime/position_ids.hpp"

namespace rt = locateanything_runtime;

namespace {

void PrintIds(const rt::PositionIds &p, const char *title) {
  std::printf("  %s  shape=[%d,%d,%d]  vals=", p.shape[0], p.shape[1],
              p.shape[2], title);
  int32_t n = p.shape[2];
  int32_t show = n <= 12 ? n : 12;
  for (int32_t i = 0; i < show; ++i) {
    std::printf("%d ", p.data[i]);
  }
  if (n > 12) std::printf("... (last 6: %d %d %d %d %d %d)",
      p.data[n-6], p.data[n-5], p.data[n-4], p.data[n-3], p.data[n-2], p.data[n-1]);
  std::printf("\n");
}

}  // namespace

int main() {
  bool ok = true;

  // Case A: cold prefill, q=256, past=0, block_size=0 (no PBD).
  // Expect: 0..255 strictly increasing.
  rt::PositionIds prefill;
  rt::BuildPositionIds(256, 0, 0, false, &prefill);
  PrintIds(prefill, "A. prefill (q=256, past=0, no PBD):");
  for (int32_t i = 0; i < 256; ++i) {
    if (prefill.data[i] != i) { ok = false; std::printf("[FAIL] prefill[%d]=%d\n", i, prefill.data[i]); }
  }

  // Case B: PBD decode, q=6, past=256, block=6.
  // base would be [256,257,258,259,260,261]; after [-6:]-=1 → [255,256,257,258,259,260].
  // Wait — that shifts ALL 6. Upstream: position_ids[-n_future:]-=1 where
  // n_future=block_size=6 and q_len=6, so all 6 get -=1.
  // Result: [255,256,257,258,259,260].
  rt::PositionIds decode;
  rt::BuildPositionIds(6, 256, 6, true, &decode);
  PrintIds(decode, "B. decode PBD (q=6, past=256, block=6):");
  int32_t expect_b[] = {255, 256, 257, 258, 259, 260};
  for (int i = 0; i < 6; ++i) {
    if (decode.data[i] != expect_b[i]) { ok = false; std::printf("[FAIL] decode[%d]=%d expect %d\n", i, decode.data[i], expect_b[i]); }
  }

  // Case C: prefill with prior history, q=256, past=100, no PBD.
  // Expect: 100..355.
  rt::PositionIds prefill2;
  rt::BuildPositionIds(256, 100, 0, false, &prefill2);
  PrintIds(prefill2, "C. prefill (q=256, past=100):");
  for (int32_t i = 0; i < 256; ++i) {
    if (prefill2.data[i] != 100 + i) { ok = false; std::printf("[FAIL] prefill2[%d]\n", i); }
  }

  // Case D: plain AR decode (no PBD), q=1, past=300, block=0.
  // Expect: [300].
  rt::PositionIds ar;
  rt::BuildPositionIds(1, 300, 0, false, &ar);
  PrintIds(ar, "D. AR decode (q=1, past=300):");
  if (ar.data[0] != 300) { ok = false; std::printf("[FAIL] ar[0]=%d\n", ar.data[0]); }

  std::printf("[verdict] position_ids test %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
