// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Prefill verify: 跑 language.hbm 的 prefill graph 一次, 验证 hbm 能跑 +
// logits 合理 (无 NaN/Inf, shape 对, argmax 在合理范围).
//
// 输入构造 (按你"真实 prompt + dummy vision"决策, 但简化: 不 tokenize,
// 直接用 embed_tokens.bin 前 256 个真实 token 的 embed 当输入):
//   input_0 (1,256,2048)fp16 = embed_lookup(token_ids=arange(0,256))
//   input_1 (1,1,256)int32   = position_ids arange(0,256)  [纯 causal, 无 PBD]
//   input_2 (1,256,1024)fp16 = causal mask (下三角 allow, 上三角 mask=-32768)
//   input_3..74 (72个) (1,1024,2,128)int8 = 全 0  [KV cache 冷启动]
//
// 预期: logits (1,256,152681)fp16, 无 NaN, argmax(token 255) 在合理 id 范围.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include "locateanything_runtime/hbm_session.hpp"
#include "locateanything_runtime/embed_lookup.hpp"
#include "locateanything_runtime/attention_mask.hpp"
#include "locateanything_runtime/position_ids.hpp"

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

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <language.hbm> <embed_tokens.bin>\n", argv[0]);
    return 1;
  }
  const char *hbm_path = argv[1];
  const char *embed_path = argv[2];

  // 1. 加载 language hbm
  rt::HbmSession session;
  auto r = session.Load(hbm_path);
  if (!r.ok()) { std::fprintf(stderr, "[FAIL] Load: %s\n", r.message.c_str()); return 2; }
  std::printf("[ok] Load. graphs: ");
  for (auto &n : session.GetGraphNames()) std::printf("%s ", n.c_str());
  std::printf("\n");

  rt::Graph *prefill = session.GetGraph("prefill");
  if (!prefill) { std::fprintf(stderr, "[FAIL] prefill graph not found\n"); return 3; }
  std::printf("[ok] prefill: %d in %d out\n",
              (int)prefill->GetInputNames().size(), (int)prefill->GetOutputNames().size());

  // 2. embed_lookup: 前 256 个 token
  rt::EmbedLookup embed;
  if (!embed.Open(embed_path, 152681, 2048)) { std::fprintf(stderr, "[FAIL] embed open\n"); return 4; }
  std::printf("[ok] embed open: vocab=%d hidden=%d\n", embed.VocabSize(), embed.HiddenDim());

  std::vector<int32_t> token_ids(256);
  for (int i = 0; i < 256; ++i) token_ids[i] = i;  // token 0..255
  std::vector<uint8_t> embed_bytes(256 * 2048 * 2, 0);
  embed.Gather(token_ids.data(), 256, embed_bytes.data());
  rt::Tensor in_embeds;
  in_embeds.shape = {1, 256, 2048};
  in_embeds.dtype = 4;  // F16
  in_embeds.data = embed_bytes;
  std::printf("[ok] embeds built (token 0..255)\n");

  // 3. position_ids: 0..255 (纯 causal, 无 PBD)
  rt::PositionIds pos;
  rt::BuildPositionIds(256, 0, 0, false, &pos);
  rt::Tensor in_pos;
  in_pos.shape = {1, 1, 256};
  in_pos.dtype = 8;  // S32
  in_pos.data.resize(256 * 4);
  std::memcpy(in_pos.data.data(), pos.data.data(), 256 * 4);

  // 4. attention_mask: causal (prefill, block_size=0)
  rt::AttentionMask mask;
  uint16_t mask_val = rt::FloatToFp16Bits(-32768.0f);
  rt::BuildAttentionMask(256, 1024, 0, 0, mask_val, false, &mask);
  rt::Tensor in_mask;
  in_mask.shape = {1, 256, 1024};
  in_mask.dtype = 4;  // F16
  in_mask.data.resize(256 * 1024 * 2);
  std::memcpy(in_mask.data.data(), mask.data.data(), 256 * 1024 * 2);
  std::printf("[ok] pos + mask built (causal, 256x1024)\n");

  // 5. KV cache: 72 个 (1,1024,2,128) int8 全 0
  std::vector<rt::Tensor> inputs;
  inputs.push_back(in_embeds);   // idx 0
  inputs.push_back(in_pos);      // idx 1
  inputs.push_back(in_mask);      // idx 2
  for (int i = 0; i < 72; ++i) {
    rt::Tensor kv;
    kv.shape = {1, 1024, 2, 128};
    kv.dtype = 2;  // S8
    kv.data.assign(1024 * 2 * 128, 0);  // 全 0
    inputs.push_back(kv);
  }
  std::printf("[ok] KV cache: 72 x (1,1024,2,128) int8 zeros, total inputs=%d\n",
              (int)inputs.size());

  // 6. execute prefill
  std::vector<rt::Tensor> outputs;
  r = session.ExecuteGraphByName("prefill", inputs, &outputs);
  if (!r.ok()) { std::fprintf(stderr, "[FAIL] Execute: code=%d %s\n", r.code, r.message.c_str()); return 5; }
  std::printf("[ok] Execute returned %d outputs\n", (int)outputs.size());

  // 7. 验证 logits (output[0])
  auto &logits = outputs[0];
  std::printf("  out[0] shape=[");
  for (size_t i = 0; i < logits.shape.size(); ++i)
    std::printf("%d%s", logits.shape[i], i+1==logits.shape.size()?"":",");
  std::printf("] dtype=%s bytes=%zu\n", rt::DtypeName(logits.dtype), logits.data.size());

  if (logits.shape != std::vector<int32_t>{1, 256, 152681}) {
    std::fprintf(stderr, "[FAIL] logits shape mismatch\n");
    return 6;
  }

  // DEBUG: dump 原始字节找非0区段 (不假设layout)
  const uint16_t *raw = reinterpret_cast<const uint16_t*>(logits.data.data());
  int64_t total_fp16 = (int64_t)logits.data.size() / 2;
  // 找前 3 个非0 fp16 的位置
  int found = 0;
  for (int64_t i = 0; i < total_fp16 && found < 5; ++i) {
    if (raw[i] != 0) {
      float f = Fp16ToFloat(raw[i]);
      int64_t row = i / 152704;
      int64_t col = i % 152704;
      std::printf("  raw[%lld] (row=%lld col=%lld) = 0x%04x -> %.4f\n",
                  (long long)i, (long long)row, (long long)col, raw[i], f);
      found++;
    }
  }
  // 直接扫最后 152704 个 fp16 (row255 if stride 152704) 看非0数
  int nz_last = 0;
  for (int64_t i = (total_fp16 - 152704); i < total_fp16; ++i) {
    if (raw[i] != 0) nz_last++;
  }
  std::printf("  last 152704 fp16: non_zero=%d\n", nz_last);

  // DEBUG: 每行 (按 152704 fp16) 第一个非0的位置, 看数据真实分布
  int64_t per_row_stride = 305408 / 2;
  std::printf("  -> per-row first non-zero col (stride=%lld fp16):\n", (long long)per_row_stride);
  for (int r = 0; r < 256; ++r) {
    const uint16_t *row = raw + r * per_row_stride;
    int64_t first_nz = -1;
    for (int c = 0; c < 152681; ++c) { if (row[c] != 0) { first_nz = c; break; } }
    if (first_nz >= 0) {
      std::printf("    row[%d] first_nz col=%lld val=%.4f\n", r, (long long)first_nz, Fp16ToFloat(row[first_nz]));
    }
  }
  std::printf("  (rows without any non-zero not shown)\n");

  // 解 fp16, 检查 NaN + argmax(最后一位)
  // NOTE: logits output stride=[78184448, 305408, 2] — 每行 152681 fp16 后有
  // 46 bytes padding (152704 fp16 per row). 必须按 stride[1]/2 跳行, 不能按
  // 152681 紧密读, 否则错位.
  const uint16_t *lp = reinterpret_cast<const uint16_t*>(logits.data.data());
  int64_t row_stride_fp16 = 305408 / 2;  // = 152704
  int64_t n = 256LL * 152681;
  int nan_count = 0;
  // argmax 末位 (token 255 的预测)
  int32_t best_id = 0; float best_val = -1e30f;
  const uint16_t *row255 = lp + 255 * row_stride_fp16;
  for (int32_t v = 0; v < 152681; ++v) {
    float f = Fp16ToFloat(row255[v]);
    if (std::isnan(f)) { nan_count++; continue; }
    if (f > best_val) { best_val = f; best_id = v; }
  }
  // 全量 NaN 抽样
  for (int64_t i = 0; i < n; i += 10007) {
    if (std::isnan(Fp16ToFloat(lp[i]))) nan_count++;
  }
  std::printf("  -> token255 argmax: id=%d val=%.4f (NaN sample hits=%d)\n",
              best_id, best_val, nan_count);

  // argmax 前 5 位 (token 0..4) 看趋势
  std::printf("  -> argmax per position (first 5): ");
  for (int p = 0; p < 5; ++p) {
    int32_t b = 0; float bv = -1e30f;
    const uint16_t *row = lp + p * row_stride_fp16;
    for (int32_t v = 0; v < 152681; ++v) {
      float f = Fp16ToFloat(row[v]);
      if (!std::isnan(f) && f > bv) { bv = f; b = v; }
    }
    std::printf("[%d]=%d ", p, b);
  }
  std::printf("\n");

  // DEBUG: KV output[1] 非0 才说明 BPU 真写了
  if (outputs.size() > 1) {
    const auto &kvout = outputs[1];
    const int8_t *kp = reinterpret_cast<const int8_t*>(kvout.data.data());
    int nz = 0, sn = 0;
    for (size_t i = 0; i < kvout.data.size(); i += 97) { sn++; if (kp[i] != 0) nz++; }
    std::printf("  -> KV out[1] bytes=%zu non_zero=%d/%d sampled\n", kvout.data.size(), nz, sn);
  }
  // logits 整体 min/max/mean
  float lmin = 1e30f, lmax = -1e30f; double lsum = 0; int64_t lcnt = 0;
  for (int64_t i = 0; i < 256LL * 152681; i += 1009) {
    float f = Fp16ToFloat(lp[i]);
    if (std::isnan(f)) continue;
    if (f < lmin) lmin = f;
    if (f > lmax) lmax = f;
    lsum += f; lcnt++;
  }
  std::printf("  -> logits sample min=%.4f max=%.4f mean=%.4f (n=%lld)\n", lmin, lmax, lsum / lcnt, (long long)lcnt);

  bool ok = (nan_count == 0) && (best_id >= 0) && (best_id < 152681);
  std::printf("[verdict] prefill verify %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
