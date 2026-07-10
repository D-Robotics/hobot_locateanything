// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Prefill output layout 深查: dump logits output buffer 所有非0区段的
// 起止 offset + 字节范围, 找 BPU 真实写数据规律. 不假设任何 stride.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <utility>

#include "locateanything_runtime/hbm_session.hpp"
#include "locateanything_runtime/embed_lookup.hpp"
#include "locateanything_runtime/attention_mask.hpp"
#include "locateanything_runtime/position_ids.hpp"

namespace rt = locateanything_runtime;

static float Fp16ToFloat(uint16_t h) {
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

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s <language.hbm> <embed_tokens.bin>\n", argv[0]); return 1; }
  rt::HbmSession session;
  auto r = session.Load(argv[1]);
  if (!r.ok()) { std::fprintf(stderr, "[FAIL] Load\n"); return 2; }
  rt::Graph *prefill = session.GetGraph("prefill");
  if (!prefill) return 3;

  rt::EmbedLookup embed;
  if (!embed.Open(argv[2], 152681, 2048)) return 4;
  std::vector<int32_t> tids(256);
  for (int i = 0; i < 256; ++i) tids[i] = i;
  std::vector<uint8_t> eb(256 * 2048 * 2, 0);
  embed.Gather(tids.data(), 256, eb.data());
  rt::Tensor in_e; in_e.shape = {1,256,2048}; in_e.dtype = 4; in_e.data = eb;

  rt::PositionIds pos; rt::BuildPositionIds(256, 0, 0, false, &pos);
  rt::Tensor in_p; in_p.shape = {1,1,256}; in_p.dtype = 8; in_p.data.resize(256*4);
  std::memcpy(in_p.data.data(), pos.data.data(), 256*4);

  rt::AttentionMask mask;
  rt::BuildAttentionMask(256, 1024, 0, 0, rt::FloatToFp16Bits(-32768.0f), false, &mask);
  rt::Tensor in_m; in_m.shape = {1,256,1024}; in_m.dtype = 4; in_m.data.resize(256*1024*2);
  std::memcpy(in_m.data.data(), mask.data.data(), 256*1024*2);

  std::vector<rt::Tensor> inputs;
  inputs.push_back(in_e); inputs.push_back(in_p); inputs.push_back(in_m);
  for (int i = 0; i < 72; ++i) {
    rt::Tensor kv; kv.shape = {1,1024,2,128}; kv.dtype = 2; kv.data.assign(1024*2*128, 0);
    inputs.push_back(kv);
  }

  std::vector<rt::Tensor> outputs;
  r = session.ExecuteGraphByName("prefill", inputs, &outputs);
  if (!r.ok()) { std::fprintf(stderr, "[FAIL] Execute\n"); return 5; }

  auto &logits = outputs[0];
  const uint8_t *raw = logits.data.data();
  int64_t total = (int64_t)logits.data.size();
  std::printf("logits out bytes=%zu\n", logits.data.size());

  // 找所有非0 byte 的区段 (连续非0段)
  int64_t seg_start = -1;
  int seg_count = 0;
  for (int64_t i = 0; i < total; ++i) {
    if (raw[i] != 0) {
      if (seg_start < 0) { seg_start = i; }
    } else {
      if (seg_start >= 0) {
        int64_t len = i - seg_start;
        std::printf("seg[%d] offset=%lld len=%lld bytes (%lld fp16) | as 152681-row: row=%lld col=%lld\n",
                    seg_count, (long long)seg_start, (long long)len, (long long)len/2,
                    (long long)(seg_start/2/152681), (long long)((seg_start/2)%152681));
        // 算 152704-row
        std::printf("         as 152704-row: row=%lld col=%lld\n",
                    (long long)(seg_start/2/152704), (long long)((seg_start/2)%152704));
        seg_count++; seg_start = -1;
      }
    }
  }
  if (seg_start >= 0) {
    std::printf("seg[%d] offset=%lld len=%lld (to end)\n", seg_count, (long long)seg_start, (long long)(total-seg_start));
    seg_count++;
  }
  std::printf("total non-zero byte segments: %d\n", seg_count);

  return 0;
}
