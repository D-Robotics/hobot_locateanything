# LocateAnything on S600 — Runtime Architecture

> **Design decision (2026-07-10)**: LocateAnything-3B on D-Robotics S600 uses a **split HBM + host runtime** deployment rather than a monolithic unified HBM. The vision tower (MoonViT-SO-400M) and language tower (Qwen2.5-3B + PBD) are compiled to two independent HBMs, and a Python host runtime orchestrates the end-to-end forward pass, KV cache management, and PBD generation loop.

## 1. Why split-HBM (not monolithic unified HBM)

hbdk4 4.10.2's `link_models()` merges multiple graphs into one file — it does **not** perform cross-modality kernel fusion. Vision output tensors and language input tensors still round-trip through DDR between graph executions regardless of whether they live in one HBM or two. Therefore:

- **A unified HBM buys nothing at inference time** — same execute cost, plus 3.5 GB extra memory footprint on-chip.
- **A unified HBM costs at compile time** — wall-clock ≈ M2 (2h20m) + M3 (75m) ≈ 3.5h vs. running them independently and reusing artifacts.
- **A unified HBM hurts debuggability** — a shape error inside a merged graph gives no hint whether the offender is on the vision or language side.

Split-HBM wins on every axis except "one file to ship", which is a minor convenience compared to the above.

## 2. Physical artifact layout

```
main/
├── vision/outputs/locateanything-vit-3b_nash-p_w4/
│   └── LocateAnything-3B_vision_448x448_w8_nash-p_corenum_4.hbm    (463 MB)
│       └── graph "visual": (1, 1024, 588) fp32 -> (1, 256, 2048) fp16
│
├── language/outputs/locateanything-lm-3b_nash-p_w4/
│   ├── LocateAnything-3B_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm    (1.6 GB)
│   │   ├── graph "prefill": q_len=256   (per-session, run ~1-4 times)
│   │   └── graph "decode":  q_len=6     (per-token-block, run N times, PBD)
│   └── LocateAnything-3B_embed_tokens.bin    (597 MB, 152681 x 2048 fp16)
│
└── runtime/
    ├── locateanything_runtime.py       (top-level API)
    ├── image_preprocess.py             (448x448 patchify -> (1024, 588))
    ├── text_preprocess.py              (LocateAnythingProcessor logic ported to numpy)
    ├── embed_lookup.py                 (mmap embed_tokens.bin for gather)
    ├── kv_cache.py                     (36-layer circular ring buffer, cache_len=1024)
    ├── attention_mask.py               (causal + PBD block-bidirectional mask builder)
    ├── position_ids.py                 (1D vanilla RoPE indices + PBD shared-window offset)
    ├── pbd_generate.py                 (PBD 6-token block decode loop + <switch> / <box> parsing)
    ├── coord_decode.py                 (<0>..<1000> token IDs -> normalized bbox)
    └── cli.py                          (locateanything-run <image> <prompt> -> bbox + text)
```

## 3. Host vs. BPU responsibility split

| Component | Location | Notes |
|---|---|---|
| Tokenize prompt | Host | `LocateAnythingProcessor` port, tokenizer.json |
| Image resize/patchify | Host | 448x448 fixed for now; patch_size=14 -> 1024 patches x (3x14x14=588) |
| Embed token lookup | Host | `embed_tokens.bin` mmap + `np.take` by IDs |
| Vision tower forward | **BPU** | `vision.hbm` graph "visual" |
| Vision -> text embed concat | Host | Replace `image_token_index=151665` placeholder with 256 vision embeds |
| Attention mask construction | Host | Causal + last-`block_size` diagonal block bidirectional (PBD) |
| Position IDs construction | Host | `np.arange` + PBD shared-window offset `pos_ids[-6:] -= 1` |
| KV cache circular buffer | Host | 36 layers x 2 (K+V) x (1, 1024, 2, 128) int8, write ptr, prefill init |
| **LM 36-layer attn+mlp** | **BPU** | `language.hbm` graphs "prefill" and "decode" |
| **LM head (tied w/ embed)** | **BPU** | Baked into HBM at compile time; output is already fp16 logits `(1, q_len, 152681)` |
| Sampling (argmax/temperature/top-k) | Host | Standard nucleus sampling on fp16 logits |
| PBD `<switch>` / `<box>` decision | Host | After each 6-token block, decide whether to stop mask-generation mode |
| `<0>` .. `<1000>` coord decode | Host | Coord token IDs -> `float / 1000` normalized coordinates |

## 4. HBM graph I/O reference

### 4.1 `language.hbm::prefill` (called 1-4 times per session)

- **Inputs (75)**:
  - `_input_0` : `(1, 256, 2048)` fp16 — input embeds (from host embed lookup + vision concat)
  - `_input_1` : `(1, 1, 256)` int32 — position IDs
  - `_input_2` : `(1, 256, 1024)` fp16 — attention mask (0 for allowed, `mask_value=-32768` for blocked)
  - `_input_3` .. `_input_74` : 72 x `(1, 1024, 2, 128)` int8 — KV cache in (36 layers x K/V)
- **Outputs (73)**:
  - `_output_0` : `(1, 256, 152681)` fp16 — logits for all 256 positions
  - `_output_1` .. `_output_72` : 72 x `(1, 256, 2, 128)` int8 — KV cache out (36 layers x K/V, first-256-position writes)

### 4.2 `language.hbm::decode` (called N times per session, PBD block)

Same I/O shape as prefill except `q_len` = 6 instead of 256 on the `_input_0` / `_output_0` / KV-cache-out axes. `_input_2` becomes `(1, 6, 1024)` and `_input_1` becomes `(1, 1, 6)`.

### 4.3 `vision.hbm::visual` (called 1 time per session)

- **Input**: `(1, 1024, 588)` fp32 — patchified 448x448 RGB image, 1024 patches of 3x14x14
- **Output**: `(1, 256, 2048)` fp16 — 256 vision tokens (patch_merger 2x2 downsample) at LM hidden size

## 5. Inference timeline (single query, one image + prompt)

```
[T=0]      user_input = (image, prompt)
[T=0+t1]   host: image_preprocess(image) -> (1024, 588) fp32
[T=0+t2]   host: tokenize(prompt) -> token_ids  (with image placeholder markers)
[T=0+t3]   host: embed_lookup(token_ids) -> text_embeds fp16
[T=0+t4]   BPU:  vision.hbm.execute() -> vision_embeds (1, 256, 2048) fp16     [~50-100 ms]
[T=0+t5]   host: replace image placeholder with vision_embeds
[T=0+t6]   host: build prefill inputs (mask + pos_ids + zero KV cache)
[T=0+t7]   BPU:  language.hbm.prefill.execute() -> logits + KV cache            [~200-500 ms per chunk of 256]
[T=0+t8]   host: sample first token, append to KV cache, loop:
              for step in 1..max_new_blocks:
                host: build decode inputs (single-6-block mask + pos_ids + KV cache)
                BPU:  language.hbm.decode.execute() -> logits (1,6,152681)      [~50-100 ms per block]
                host: parse 6 tokens
                host: if <switch> encountered: exit
                host: if <eos> encountered: exit
[T=end]    host: coord_decode(token_stream) -> bbox list + descriptive text
```

End-to-end wall-clock estimate: **1-4 seconds per query** on S600. Actual numbers are pending sanity + benchmark (M6, M7).

## 6. Weight file transport plan

**On the 4090 (build machine, 10.112.20.45)**:
- Compile artifacts under `main/{vision,language}/outputs/`
- Push code (this repo) to GitHub via HTTPS + PAT credential helper (see KNOWN_ISSUES #011)

**On the S600 (deploy machine, 10.112.133.20)**:
- `git clone https://github.com/LiuAnclouds/oe_locateanything.git` on S600
- User `scp` the following manually from 4090 to S600:
  - `main/vision/outputs/locateanything-vit-3b_nash-p_w4/LocateAnything-3B_vision_448x448_w8_nash-p_corenum_4.hbm` (463 MB)
  - `main/language/outputs/locateanything-lm-3b_nash-p_w4/LocateAnything-3B_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm` (1.6 GB)
  - `main/language/outputs/locateanything-lm-3b_nash-p_w4/LocateAnything-3B_embed_tokens.bin` (597 MB)
- Total transfer: ~2.7 GB one-time per model version
- S600 stores under identical path structure so runtime code is env-agnostic
- Weights are `.gitignore`-d in this repo (see `.gitignore`), never `git commit`-d

## 7. Deploy-time invariants

1. **Weights source is `eagle/Embodied/LocateAnything-3B`** (LA-fine-tuned Qwen2.5+MoonViT), not the raw Qwen2.5-VL baseline
2. **Vocab is 152681** (with coord `<0>..<1000>`, ref, box, switch, null tokens) — not 151936
3. **PBD block_size = decode_seq_len = 6** — do not compile with `q_len != 6`
4. **MoonViT vision tower** is not swappable with Qwen2.5-VL / Qwen3-VL ViT
5. **cache_len = 1024** implies max total sequence (prompt + generated) = 1024 tokens

## 8. Future extensions

- Native-resolution vision (bicubic pos-emb interpolation) once single-image OOM is resolved (M5)
- Multi-batch inference (currently batch_size=1 hard-coded due to PBD constraints)
- INT4 language weight compression (currently INT8 quant; INT4 experimental)
- On-chip vision output to language input direct DMA (requires hbdk4 cross-graph fusion support, not available in 4.10.2)
