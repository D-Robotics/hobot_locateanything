# LocateAnything C++ 自研推理流程设计 (S600) v2

> v2 修正: 通读 upstream modeling_locateanything.py (lines 304-537) +
> generate_utils.py (lines 15-503) + modeling_qwen2.py prepare_inputs_for_generation
> (1551-1606) 后, 纠正 v1 的 3 处错误:
> 1. prefill 不是独立一步, 是 generate loop 的 iter_round==1 (MTP 路径喂 L+6, KV 回退到 L)
> 2. 每轮 token 数可变 (handle_pattern 返回 1/3/4/6), 不是固定 6
> 3. 坐标 token 不靠 argmax, 靠 decode_bbox_avg 从 logits top-k 算 + hybrid 异常过滤
>
> 决策背景见 docs/KNOWN_ISSUES.md #020: libxlm 乱码放弃, 纯 C++ 自研, 借 libxlm
> tokenizers_* C API 做 tokenizer.

## 复用 (来自 libxlm / 已写模块)

- **hbm 加载 + execute**: `hbm_session.cpp` (KNOWN_ISSUES #015-#018, 对齐 HB_HBMRuntime.cc)
  - vision.hbm graph "visual": (1,1024,588)fp32 → (1,256,2048)fp16
  - language.hbm graph "prefill": 75 in (embeds(1,L,2048)fp16 + pos(1,1,L)int32 + mask(1,L,1024)fp16 + 72×kv(1,1024,2,128)int8) → 73 out (logits(1,L,152681)fp16 + 72×kv)
  - language.hbm graph "decode": 同 prefill 但 q_len=6
  - env: HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
- **embed_lookup**: mmap embed_tokens.bin 597MB, gather by token id
- **attention_mask**: causal + PBD block_size=6 bidirectional + prev-trailing mask
- **position_ids**: vanilla 1D + PBD pos[-6:]-=1
- **tokenizer**: libxlm tokenizers_* C API (encode/decode/id_to_token), 加载
  main/language/tokenizer/tokenizer.json (152681 vocab, 1001 坐标 token)
- **image_preprocess**: OpenCV (libopencv_world, S600 自带) — resize 448×448 +
  BGR2RGB + 归一化(0.5/0.5) + patchify(14×14×3=588) → (1024,588) fp32

## 关键 special token id (config.json + get_token_ids_from_config)

```
image_token_index    = 151665  (vision embed 占位符, 单个)
box_start_token_id   = 151668  <box>
box_end_token_id     = 151669  </box>
coord_start_token_id = 151677  <0>   坐标范围 [151677, 152677] 共 1001
coord_end_token_id   = 152677  <1000>
ref_start_token_id   = 151672  <ref>
ref_end_token_id     = 151673  </ref>
text_mask_token_id   = 151676  <text_mask>  (PBD pre_mask_tokens 用, =default_mask_token_id)
null_token_id        = 152678  <null>       (终止信号)
switch_token_id      = 152679  <switch>     (config 有, 但 handle_pattern 实际没用)
im_end_token_id      = 151645  <|im_end|>   (eos, 对话终止)
none_token_id        = 4064    (Qwen 原生 subword, "no object")
block_size / n_future_tokens = 6
```

## 端到端推理 loop (对齐 upstream generate, modeling_locateanything.py:464-510)

### Step A: 前处理 (loop 之前)

```
A1. image_preprocess(image.jpg) → (1024, 588) fp32
    cv::imread → resize(448,448) → BGR2RGB → 归一化 (pix/255-0.5)/0.5
    → patchify 32×32 patch × 14×14×3 = 1024 patches × 588

A2. vision.hbm.execute(visual, {(1024,588)fp32}) → vit_embeds (1,256,2048)fp16
    (MoonViT patch_embed + 27 blocks + merger + mlp1 全在 hbm 里)

A3. tokenizer.encode(chat_template(query="cat"))
    套 chat_template: <|im_start|>system\nYou are...<|im_end|>\n<|im_start|>user\n<image>cat<|im_end|>\n<|im_start|>assistant\n
    → input_ids (含 1 个 image_token_index=151665 占位符)
    (注: chat_template 用 <image 1> 占位符, tokenizer 把它编成 151665? 要 verify)

A4. generated = input_ids  (累积序列, 初始 = prompt)
    past_key_values = 36 层 × 2 (K/V) × (1, 1024, 2, 128) int8, 全 0
    total_gen_length = min(max_pos, len(generated) + max_new_tokens)
    full_position_ids = arange(0, total_gen_length + 6)  # +6 防 OOB
    use_mtp = (generation_mode == 'hybrid' or 'fast')  # 默认 hybrid
    iter_round = 0
```

### Step B: generate loop (while len(generated) < total_gen_length)

```
iter_round += 1

B1. 构造本轮输入 (MTP 路径, use_mtp=True 时):
    generated_with_mask = generated + [generated[-1]] + [text_mask]*5
                         # [generated(L), 上轮尾重复(1), 5 mask] = L+6
    start_idx = past_kv 已写入的 token 数  # iter1=0, 之后=len(generated)
    position_ids = full_position_ids[start_idx : L+6]
    position_ids[-6:] -= 1   # 重复 token + 5 mask 共 6 个位置回退 1
    # iter1: 模型吃全 L+6 (past_kv 空); iter2+: 模型只吃 6 (1 重复+5 mask),
    #        历史从 KV cache 读

    AR 路径 (use_mtp=False 时):
    position_ids = full_position_ids[start_idx : len(generated)]
    # 模型只吃 1 个 token (上轮尾), 历史从 KV cache

B2. iter_round==1 时附加 vision:
    把 generated_with_mask 里的 image_token_index=151665 位
    替换成 vit_embeds (256, 2048) → embeds 序列净增 255
    (仅 iter1 有 image_token_index 占位符, 之后轮次没有)

B3. attention_mask:
    MTP: BuildAttentionMask(q_len=6, cache_len=1024, past_len=start_idx,
                            block_size=6, causal=false)
         → (1, 6, 1024) fp16  [最后6×6 bidirectional + prev-trailing mask]
    AR:  BuildAttentionMask(q_len=1, cache_len=1024, past_len=start_idx,
                            block_size=0, causal=true)
         → (1, 1, 1024) fp16  [纯 causal]
    iter1 (MTP, prefill): q_len=L+6, past_len=0, block_size=6
         → (1, L+6, 1024) fp16

B4. language.hbm.execute(prefill or decode, {embeds, pos_ids, mask, 72×kv_in})
    iter1: 用 prefill graph (q_len=L+6)
    iter2+: 用 decode graph (q_len=6 或 1)
    → logits (1, q_len, 152681) fp16 + 72×kv_out (写到 cache 的 start_idx.. 位)

B5. KV cache 截断回退:
    把 KV cache 截到 len(generated) (丢弃本轮多写的 6 个 mask 位)
    # upstream lines 483-486, 每轮都做

B6. 采样 (MTP 路径):
    next_logits = logits[:, -6:, :]   # 取最后 6 位
    x0 = argmax/sample 6 个 token
    box_avg = decode_bbox_avg(next_logits, probs, token_ids, keep_k_avg=4)
              # 从 4 个坐标位 top-k 算坐标 token id + hybrid 异常过滤
              # 返回 [box_start, x1, x2, y1, y2, box_end] 或 None
    is_box_empty = (box_avg 全 0)
    new_tokens = x0 if is_box_empty else box_avg
    pattern = handle_pattern(new_tokens, token_ids, mode='hybrid')
              # 返回 {type, tokens(变长1/3/4/6), need_switch_to_ar, is_terminal}
    out_token = pattern['tokens']   # 变长!
    out_type = pattern['type']

    采样 (AR 路径):
    next_logits = logits[:, -1:, :]
    out_token = argmax 1 个 token
    out_type = 'box_end_ar' if token==box_end else 'coord_ar' if coord/none
               else 'im_end' (hybrid) / 'continue_ar' (slow)

B7. generated += out_token   # 追加变长 tokens
    iter_round==1 时: 还要把 image_token_index 替换的 255 个 vision token
    计入 generated 的长度 (上游 generated 是 input_ids, vision 是 forward 时
    内部替换, generated 长度仍按 input_ids 算 — 要 verify 这个长度语义)

B8. 终止 + 模式切换:
    if out_type == 'im_end': break
    if mode == 'hybrid':
        if out_type == 'error_box': use_mtp = False  # 切 AR
        elif out_type == 'box_end_ar': use_mtp = True  # 切回 MTP
```

### handle_pattern 全分支 (generate_utils.py:408-503)

```
x0 = [t0, t1, t2, t3, t4, t5]  (6 个 token)

1. t0 == null_token       → {type: im_end, tokens: [im_end], terminal}        # len 1
2. t0 == im_end_token     → {type: im_end, tokens: [im_end], terminal}        # len 1
3. [t0,t1] == [box,none]  → {type: empty_box, tokens: [box,none,box_end]}    # len 3
4. t0 == box_start:
   数 t1..t4 里连续 coord token 数 coord_ix (初始1, 遇 coord 就+1)
   - coord_ix==5 && t5==box_end → {type: coord_box, tokens: x0}              # len 6
   - coord_ix==3 && t3==box_end → {type: point_box, tokens: x0[:4]}          # len 4
   - else:
     - mode==fast  → {type: coord_box, tokens: x0}                           # len 6 (宽容)
     - else         → {type: error_box, tokens: x0[:coord_ix], switch_to_ar} # len 1-4
5. else (ref/text):
   遇 null 截断, 末尾双 ref_end 去重
   → {type: ref_object, tokens: x0 (变长 0-6)}
```

### decode_bbox_avg (generate_utils.py:276-361)

```
输入: logits (6,V), probs (6,V), token_ids, keep_k_avg=4 (注意不是 keep_k=5)
1. box_type = is_valid_box_frame(probs)  # empty/legal/illegal
2. empty_box → return [box,none,box_end,null,null,null]  (6 长度但只有前3有效)
3. illegal_box → return None
4. legal_box:
   for pos in 1..4: topk(probs[pos], k=4) → pos_ids, pos_probs
   mask = coord_start <= pos_ids <= coord_end
   任一位 top-k 没 coord → return None
   first_valid = 每位最高 prob 的 coord id
   if mode==hybrid: 异常过滤 (prob<0.9 && >1 coord && (max-min)>60 → 该位置置0)
   if mode==fast: 不过滤
   return [box_start, c0, c1, c2, c3, box_end]  (6 token id)
```

### is_valid_box_frame (generate_utils.py:246-273)

```
读 probs (6,V):
- p_start = probs[0, box_start]
- empty_box if p_start>=0.6 && probs[1,none]>0.2 && probs[2,box_end]>0.2
            && probs[3,null]>0.1 && probs[4,null]>0.1
- legal_box if probs[5,box_end]+probs[5,null]+probs[5,im_end] >= 0.2
- illegal_box else
```

## 待写模块 (S600 上写, 逐模块测试 push)

- [x] hbm_session (Phase 1, vision PASS)
- [x] embed_lookup (Phase 2, PASS)
- [x] attention_mask (Phase 2, PASS)
- [x] position_ids (Phase 2, 已写待测)
- [ ] image_preprocess (OpenCV 版, 替换 Phase1 dummy)
- [ ] tokenizer_wrapper (封装 libxlm tokenizers_* C API + chat_template 渲染)
- [ ] vision_text_concat (image_token_index 占位符→256 vision embed 替换, iter1 only)
- [ ] kv_cache (36 层 × 2 × (1,1024,2,128) int8 ring buffer + 每轮截断回退)
- [ ] sampling (argmax/temperature/top-p + apply_repetition_penalty)
- [ ] decode_bbox_avg + is_valid_box_frame (移植 generate_utils.py)
- [ ] handle_pattern (移植, 全分支)
- [ ] pbd_generate (顶层 generate loop, 整合 B1-B8)
- [ ] locateanything_infer (CLI 入口 + perf 计时)
- [ ] run_locateanything_infer.sh

## 关键 gotcha (subagent 提醒, C++ 移植要小心)

1. keep_k vs keep_k_avg: _sample_token_in_mtp 传 keep_k=5 是顶层 kwarg,
   decode_bbox_avg 内部用 keep_k_avg (默认4). 不要混淆.
2. full_position_ids 上界 = total_gen_length + 6, 防 OOB.
3. KV 截断每轮都做 (AR 路径是 no-op, 安全).
4. visual_features/image_token_index 仅 iter1 附加; iter1 language_model 吃的
   是已 mlp1 投影的 embed (mlp1 在 vision.hbm 里, A2 已含).
5. pre_mask_tokens 用 text_mask_token_id=151676, 不是 null(152678)/switch(152679).
6. switch_token_id config 有但 handle_pattern 没用, 别照它的存在推断行为.
7. iter1 position_ids: 全 L+6, 最后 6 减 1 (重复 token 在 L-1 位, 5 mask 在 L..L+4).
8. generated 长度语义: 上游 generated=input_ids, vision 在 forward 内部替换,
   generated 长度按 input_ids 算 (image_token_index 算 1 个). 要 verify 我们
   hbm 的 prefill graph 期望的 q_len 是 input_ids 长度还是 +255 后的.
   # 这是最大的不确定点, 写 vision_text_concat 前要 verify hbm prefill input_0
   # 的 shape 是 (1, L_input, 2048) 还是 (1, L_input+255, 2048).
