# LocateAnything C++ 自研推理流程设计 (S600)

> 决策背景: libxlm.so 能加载 LA 全部 hbm (prefill+decode+visual) + xlm_infer ret=0 +
> 真实性能 (vit 30ms / prefill 6426 tps / decode 57 tps / tpot 17.4ms), 但输出乱码——
> libxlm 走 Qwen2.5-VL 分支按 Qwen2.5-VL 构造 image embed 插入/position_ids/mask,
> 跟 LA 的 image_token_index 单占位符替换 + vanilla 1D position + PBD block_size=6
> bidirectional mask 不匹配 (KNOWN_ISSUES #020). 用户拍板: 纯 C++ 自研, 借 libxlm
> 的 tokenizers_* C API 做 tokenizer, 其余全用我们的 hbm_session + embed_lookup +
> attention_mask + position_ids.

## 复用 (来自 libxlm / 已写模块)

- **hbm 加载 + execute**: `hbm_session.cpp` (KNOWN_ISSUES #015-#018, 已对齐 HB_HBMRuntime.cc)
  - vision.hbm graph "visual" (1 in 1 out)
  - language.hbm graph "prefill" (75 in 73 out) + "decode" (75 in 73 out)
  - L2M env var `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6`
- **embed_lookup**: `embed_lookup.cpp` (mmap embed_tokens.bin 597MB, gather by token id)
- **attention_mask**: `attention_mask.cpp` (causal + PBD block_size=6 bidirectional + prev-trailing mask)
- **position_ids**: `position_ids.cpp` (vanilla 1D + PBD pos[-6:]-=1)
- **tokenizer**: libxlm `tokenizers_*` C API (tokenizers_new_from_str / encode / decode /
  id_to_token / token_to_id / get_vocab_size), 加载 main/language/tokenizer/tokenizer.json
  (152681 vocab, 1001 坐标 token <0>~<1000> id 151677~152677)
- **image preprocess**: OpenCV (libopencv_world, S600 自带) — resize 448×448 + BGR2RGB +
  归一化(mean0.5/std0.5) + patchify(14×14×3=588) → (1024, 588) fp32

## 自研推理 loop (端到端数据流)

```
1. image_preprocess(image.jpg)
   → cv::imread → cv::resize(448,448) → cv::cvtColor(BGR2RGB)
   → 归一化 (pix/255 - 0.5)/0.5 → patchify 32×32 patch × 14×14×3 → (1024, 588) fp32

2. vision.hbm.execute(visual, {(1024,588) fp32})
   → (1, 256, 2048) fp16  [256 vision tokens × 2048 LM hidden]

3. tokenizer.encode(chat_template(query))  [query="cat"]
   → token_ids 含 image_token_index=151665 占位符 (单数)
   → e.g. [<|im_start|>, user, ..., 151665, cat, ..., <|im_end|>, <|im_start|>, assistant, \n]

4. embed_lookup.Gather(token_ids)  [除了 151665]
   → text_embeds fp16 (1, L_text, 2048), 151665 位留空

5. 拼接: 把 vision_embeds (256, 2048) 替换进 151665 占位符位
   → full_embeds (1, L_text + 255, 2048) fp16  [因为 1 个占位符变 256, 净增 255]

6. attention_mask.BuildAttentionMask(q_len=L, cache_len=1024, past_len=0,
                                     block_size=0, causal=false)  [prefill 纯 causal]
   → (1, L, 1024) fp16

7. position_ids.BuildPositionIds(q_len=L, past_len=0, block_size=0, is_pbd=false)
   → (1, 1, L) int32  [0..L-1]

8. kv_cache: 36 层 × 2 (K/V) × (1, 1024, 2, 128) int8, 全 0 冷启动

9. language.hbm.execute(prefill, {embeds, pos_ids, mask, 72×kv_cache_in})
   → logits (1, L, 152681) fp16 + 72×kv_cache_out (写到 cache 的前 L 位)

10. sample: argmax(logits[0, L-1]) → next_token  [第一个 decode 输入]

11. PBD decode loop (每轮 6 tokens):
    while not (null_token or im_end or max_steps):
      a. embed_lookup.Gather(last 6 tokens + 5 mask tokens)  [PBD: 1 实+5 mask]
         → (1, 6, 2048) fp16
      b. position_ids.BuildPositionIds(q_len=6, past_len=cur, block_size=6, is_pbd=true)
         → (1, 1, 6) int32  [pos[-6:]-=1]
      c. attention_mask.BuildAttentionMask(q_len=6, cache_len=1024, past_len=cur,
                                           block_size=6, causal=false)
         → (1, 6, 1024) fp16  [PBD bidirectional block + prev-trailing mask]
      d. language.hbm.execute(decode, {embeds, pos_ids, mask, 72×kv_cache_in})
         → logits (1, 6, 152681) + 72×kv_cache_out (写到 cache 的 cur..cur+6 位)
      e. sample 6 tokens (greedy/temperature)
      f. handle_pattern(6 tokens, token_ids) → type (coord_box/point_box/empty/im_end/...)
      g. decode_bbox_avg → 坐标 token id → float/1000 归一化 bbox
      h. cur += 6

12. 输出: ref + box 列表 + 性能 (prefill tps / decode tps / e2e)
```

## 待写模块 (S600 上写, 逐模块测试 push)

- [x] hbm_session (Phase 1, vision PASS)
- [x] embed_lookup (Phase 2, PASS)
- [x] attention_mask (Phase 2, PASS)
- [x] position_ids (Phase 2, 已写未测)
- [ ] image_preprocess (OpenCV 版, 替换 Phase1 的 dummy 版)
- [ ] tokenizer_wrapper (封装 libxlm tokenizers_* C API)
- [ ] vision_text_concat (image_token_index 占位符替换)
- [ ] kv_cache (36 层 ring buffer)
- [ ] pbd_generate (sample + handle_pattern + decode_bbox_avg, 移植 generate_utils.py)
- [ ] locateanything_infer (顶层调度 + perf 计时)
- [ ] run_locateanything_infer.sh

## 关键 special token id (config.json)

```
image_token_index    = 151665  (vision embed 占位符)
box_start_token_id   = 151668  <box>
box_end_token_id     = 151669  </box>
coord_start_token_id = 151677  <0>   (坐标 token 范围 [151677, 152677] 共 1001)
coord_end_token_id   = 152677  <1000>
ref_start_token_id   = 151672  <ref>
ref_end_token_id     = 151673  </ref>
text_mask_token_id   = 151676  <text_mask>  (PBD 窗口的 mask 位)
null_token_id        = 152678  <null>       (终止信号)
switch_token_id      = 152679  <switch>     (PBD→AR 切换)
im_end_token_id      = 151645  <|im_end|>   (对话终止)
none_token_id        = 4064    (none, Qwen 原生 subword, "no object")
```
