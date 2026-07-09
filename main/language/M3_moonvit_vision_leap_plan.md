# M3 · MoonViT vision leap DSL 独立翻译 —— 开发 checklist

参考 M2 已成功的 language 编译路径，把 vision 侧从 PyTorch 版翻成 leap DSL 版并接入 oellm_build。

## 目标产物

```
LocateAnything-3B_vision_448x448_w8_nash-p_corenum_*.hbm
```

命名对齐 qwen2_5-vl-3b baseline 的 `_vision_448x448_w8_nash-p_corenum_4.hbm` 命名规范。

---

## 已就绪的基础（M2 完成时顺带做完）

| 模块 | PyTorch 参考 | leap DSL | 备注 |
|---|---|---|---|
| `blocks/vision_patch.py` | ✅ static pos_emb + Conv2d | ⬜ 待加 build() | 数值等价 0 diff |
| `blocks/vision_attention.py` | ✅ packed wqkv + SDPA | ⬜ 待加 build() | 数值等价 2.61e-8 |
| `blocks/vision_block.py` | ✅ 27 层 encoder block | ⬜ 待加 build() | 数值等价 2.15e-6 |
| `blocks/vision_patch_merger.py` | ✅ 2×2 merger + mlp1 | ⬜ 待加 build() | 数值等价 0 diff |
| `utils/rope_2d.py` | ✅ 实数展开 apply_rope_real | ⬜ 待加 apply_rope_leap_2d | 数值等价 4.77e-7 |

---

## 需要新增的文件

### `blocks/vision_*_leap.py` (4 个，参照 M2 的 `text_*_leap.py`)

- `vision_patch_leap.py` — Conv2d + 静态 pos_emb 的 leap DSL 版
- `vision_attention_leap.py` — MoonViT attention（**关键**：2D rope 用 `leap.mul/leap.slice/leap.concat` 展开）
- `vision_block_leap.py` — LayerNorm + attention + MLP2 (GELU tanh) 残差
- `vision_patch_merger_leap.py` — reshape + LayerNorm(4608) + Linear + GELU + Linear

### `utils/rope_2d_leap.py`（可选，或直接内嵌到 vision_attention_leap.py）

- `apply_rope_leap_2d(query, key, freqs_cos_sin)` — 展开成 leap 原语

### `vision_model_leap.py`

- `LocateAnythingVisionModel(Model)` — 顶层 leap Module
- `build(pixel_patches)` → visual_embeds `[N/4, 2048]`
- `get_leap_input_types()` → `[TensorType(1, 1024, 3*14*14, fp16)]` 
  格式参照 qwen2_5_vl `Qwen2_5_VLVisionModel.get_leap_input_types`（那边是 `patch_size * patch_size * in_channels`）

### `apis/model/locateanything_vision.py`

- `LocateAnythingVisionApi` — 参照 `locateanything_language.py` 结构
  - `__init__`: load LocateAnything checkpoint 的 vision_model.* + mlp1.*
  - `save_embed_tokens()`: skip（vision 不需要）
  - `compile(vit_kwargs, llm_kwargs)`: 一个 stage（`visual`），走 export_module + convert_mlir + compile_hbo + link_models

### `model_factory.py` 注册

```python
@register_model("locateanything-vit-3b", ["nash-p"])
def _build_locateanything_vit_3b(args):
    ...
```

---

## 关键坑（M2 已踩过 + 报告 pit 5.4/#9）

1. **build() 返回值必须扁平** —— 官方 vision 返回单个 tensor 就够（vision 无 KV cache），不涉及 `*list` unpacking
2. **`gelu(approximate="tanh")`** —— MoonViT MLP 用 GELU tanh，不是默认 erf；leap DSL 里对应 `leap.gelu` 是否有 `approximate` 参数需要 check
3. **2D rope 展开** —— `apply_rope_leap_2d` 需要把 `(cos, sin)` pair 拆到 `leap.slice(..., last_dim, [0, 1])` 分开取，然后 `leap.mul + leap.sub + leap.add` 重组
4. **Learnable2DInterpPosEmb bicubic 静态化** —— `MoonVisionPatchEmbedStatic.pos_emb_static` 是常量 buffer，编译时能直接 baked in ✓（P1 已解决）
5. **Conv2d in leap** —— `nn.Conv2d(3, 1152, kernel=14, stride=14)`，看 qwen2_5_vl vision `Qwen2_5_VisionPatchEmbed` 用的什么原语（`leap.conv2d` 或 `leap.matmul`）

---

## 执行顺序（预计 4-6 小时）

1. **copy + rename** qwen2_5_vl vision 的 leap DSL 骨架到 locateanything (2h)
   - `qwen2_5_vl/blocks/attention.py::Qwen2_5_VLVisionAttention.build` → `vision_attention_leap.py`
   - `qwen2_5_vl/blocks/transformer_block.py::Qwen2_5_VLVisionBlock.build` → `vision_block_leap.py`
   - `qwen2_5_vl/blocks/mlp.py::Qwen2_5_VLPatchMergerMLP.build` → 合到 `vision_patch_merger_leap.py`
   - `qwen2_5_vl/model.py::Qwen2_5_VLVisionModel.build` → `vision_model_leap.py`
2. **改造** MoonViT 特有点 (1-2h)
   - packed wqkv (不是 Q/K/V 独立)
   - 2D rope apply (不是 mrope 也不是 1D)
   - 无 window attention（全局）
   - Learnable2DInterpPosEmb 静态化
3. **Api + registry** (1h)
   - `LocateAnythingVisionApi`
   - `@register_model("locateanything-vit-3b", ["nash-p"])`
4. **编译验证** (3-4h wall time，跟 language 类似)
   - `oellm_build --model_name locateanything-vit-3b --march nash-p ...`
   - watchdog + cron 复用现有基建

---

## 待用户拍板

- 现在开始动手？还是等 M2 language 编译完 (预计 3 小时后 = ~17:30) 一起讨论？

推荐：**等 M2 编译完成 + 精度验证初步 OK 后再动 M3**，避免同时编两个 hbm 导致 4090 内存 / GPU 资源撞车。M2 用 GPU calibration 阶段已过，现在纯 CPU compile_hbo。M3 起来会再用 GPU 15-30 分钟做 calibration，可能跟 M2 有资源冲突。
