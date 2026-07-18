// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Prefill verify v3: 用真实 LA prompt 构造 (925 个 151665 占位符 + system +
// query "cat", 对齐 4090 PyTorch dump ground truth), chunk_1024 hbm.
//
// 真实 prompt (4090 dump):
//   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
//   <|im_start|>user\n<image>cat<|im_end|>\n<|im_start|>assistant\n
//   processor 把 <image> 展开成 <|vision_start|> + 925×<IMG_CONTEXT>(151665) +
//   <|vision_end|>
//   总 input_ids len ~970, pad 到 1024
//
// embed: 真实 token embed, 151665 位替换成 dummy vision embed (256 个 0.1 fp16)
// pos: 0..1023
// mask: causal (1,1024,2048) — 注意 cache_len=2048 后 mask 列维度=2048
// KV: 72×(1,1024,2,128) int8 全 0
//
// 验证: logits (1,1024,152681) 行 0..969 应该非0 (真实 prompt 长度), argmax 合理

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

static float Fp16ToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
  if (exp == 0) { if (mant == 0) return sign ? -0.0f : 0.0f; float v = (mant/1024.0f)*std::ldexp(1.0f,-14); return sign?-v:v; }
  if (exp == 31) return std::nanf("");
  float v = std::ldexp(1.0f + mant/1024.0f, (int)exp - 15);
  return sign ? -v : v;
}

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s <chunk_1024.hbm> <embed_tokens.bin>\n", argv[0]); return 1; }
  rt::HbmSession session;
  auto r = session.Load(argv[1]);
  if (!r.ok()) { std::fprintf(stderr, "[FAIL] Load: %s\n", r.message.c_str()); return 2; }
  rt::Graph *prefill = session.GetGraph("prefill");
  if (!prefill) { std::fprintf(stderr, "[FAIL] prefill graph\n"); return 3; }
  // 确认 chunk_1024 shape
  auto &is = prefill->GetInputShapes();
  std::printf("[check] prefill input_0 shape = [%d,%d,%d]\n", is[0][0], is[0][1], is[0][2]);
  if (is[0][1] != 1024) { std::fprintf(stderr, "[FAIL] expected q_len=1024, got %d\n", is[0][1]); return 4; }

  rt::EmbedLookup embed;
  if (!embed.Open(argv[2], 152681, 2048)) return 5;

  // 构造真实 LA prompt ids (对齐 4090 dump)
  // system: <|im_start|>(151644) system(8948) \n(198) You(2610) are(525) a(264) helpful(10950) assistant(17847) .(13) <|im_end|>(151645) \n(198)
  // user: <|im_start|>(151644) user(872) \n(198) <|vision_start|>(151652) [IMG_CONTEXT×925](151665) <|vision_end|>(151653) cat(4616) <|im_end|>(151645) \n(198)
  // assistant: <|im_start|>(151644) assistant(77091) \n(198)
  constexpr int32_t IM_S=151644, IM_E=151645, NL=198, VIS_S=151652, VIS_E=151653, IMG=151665, CAT=4616;
  std::vector<int32_t> tids = {
    IM_S, 8948, NL, 2610, 525, 264, 10950, 17847, 13, IM_E, NL,  // system (11)
    IM_S, 872, NL, VIS_S,  // user start + vision_start (4)
  };
  for (int i = 0; i < 925; ++i) tids.push_back(IMG);  // 925 IMG_CONTEXT
  tids.push_back(VIS_E);  // vision_end
  tids.push_back(CAT);  // cat
  tids.push_back(IM_E); tids.push_back(NL);  // <|im_end|>\n
  tids.push_back(IM_S); tids.push_back(77091); tids.push_back(NL);  // assistant\n
  // tids 长度 = 11+4+925+1+1+1+1+3 = 947? 算下: 11+4=15, +925=940, +1(vis_e)=941, +1(cat)=942, +1(im_e)=943, +1(nl)=944, +3(assistant)=947
  int real_len = (int)tids.size();
  std::printf("[ok] real prompt len=%d, pad to 1024\n", real_len);
  // pad 到 1024 (用 IM_E 填充, 反正在 mask 里被屏蔽)
  while ((int)tids.size() < 1024) tids.push_back(IM_E);

  // embed_lookup 真实 embed, 151665 位替换 dummy vision
  std::vector<uint8_t> eb(1024 * 2048 * 2, 0);
  embed.Gather(tids.data(), 1024, eb.data());
  uint16_t *e16 = reinterpret_cast<uint16_t*>(eb.data());
  for (size_t i = 0; i < tids.size(); ++i) {
    if (tids[i] == IMG) {
      uint16_t *row = e16 + i * 2048;
      for (int d = 0; d < 2048; ++d) row[d] = rt::FloatToFp16Bits(0.1f);  // dummy vision
    }
  }
  rt::Tensor in_e; in_e.shape = {1,1024,2048}; in_e.dtype = 4; in_e.data = eb;

  // pos 0..1023
  rt::PositionIds pos; rt::BuildPositionIds(1024, 0, 0, false, &pos);
  rt::Tensor in_p; in_p.shape = {1,1,1024}; in_p.dtype = 8; in_p.data.resize(1024*4);
  std::memcpy(in_p.data.data(), pos.data.data(), 1024*4);

  // mask causal (1,1024,2048) — cache_len=2048
  rt::AttentionMask mask;
  rt::BuildAttentionMask(1024, 2048, 0, 0, rt::FloatToFp16Bits(-32768.0f), false, &mask);
  rt::Tensor in_m; in_m.shape = {1,1024,2048}; in_m.dtype = 4; in_m.data.resize(1024*2048*2);
  std::memcpy(in_m.data.data(), mask.data.data(), 1024*2048*2);

  std::vector<rt::Tensor> inputs;
  inputs.push_back(in_e); inputs.push_back(in_p); inputs.push_back(in_m);
  for (int i = 0; i < 72; ++i) {
    rt::Tensor kv; kv.shape = {1,1024,2,128}; kv.dtype = 2; kv.data.assign(1024*2*128,0);
    inputs.push_back(kv);
  }
  std::printf("[ok] inputs ready: embeds+pos+mask+72×KV (all 1024-len)\n");

  std::vector<rt::Tensor> outputs;
  r = session.ExecuteGraphByName("prefill", inputs, &outputs);
  if (!r.ok()) { std::fprintf(stderr, "[FAIL] Execute: %s\n", r.message.c_str()); return 6; }
  auto &logits = outputs[0];
  const uint16_t *raw = reinterpret_cast<const uint16_t*>(logits.data.data());
  int64_t per_row = 305408 / 2;  // stride[1]/2, 可能 chunk_1024 stride 不同, 先查

  // 每 row 第一个非0, 重点看 row 0..real_len
  std::printf("per-row first non-zero (real_len=%d):\n", real_len);
  int rows_with_data = 0;
  for (int rr = 0; rr <= real_len + 5; ++rr) {
    const uint16_t *row = raw + (int64_t)rr * per_row;
    int64_t first_nz = -1;
    for (int c = 0; c < 152681; ++c) { if (row[c] != 0) { first_nz = c; break; } }
    if (first_nz >= 0) {
      if (rows_with_data < 5 || rr >= real_len - 3)
        std::printf("  row[%d] first_nz col=%lld val=%.4f\n", rr, (long long)first_nz, Fp16ToFloat(row[first_nz]));
      rows_with_data++;
    }
  }
  std::printf("rows with data (0..%d): %d\n", real_len + 5, rows_with_data);

  // argmax 最后一个真实 token (row real_len-1 = assistant\n 后的预测)
  int32_t best_id = 0; float best_val = -1e30f;
  const uint16_t *row_last = raw + (int64_t)(real_len - 1) * per_row;
  for (int v = 0; v < 152681; ++v) {
    float f = Fp16ToFloat(row_last[v]);
    if (!std::isnan(f) && f > best_val) { best_val = f; best_id = v; }
  }
  std::printf("row[%d] (last real) argmax: id=%d val=%.4f\n", real_len - 1, best_id, best_val);
  return 0;
}
