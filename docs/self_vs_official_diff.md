# Qwen2.5-VL-3B 自编译 vs 官方预编译 差异对比报告

## 1. 编译环境

| 项目 | 值 |
|------|-----|
| 主机 | 4090 服务器 (10.112.20.45) |
| GPU | NVIDIA GeForce RTX 4090 |
| Driver | 535.309.01 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| leap-llm | 1.0.5 |
| hbdk4-compiler | 4.10.2a2.dev202603180400+4c23b55.develop |
| Conda | oellm_clean (Python 3.10) |

## 2. 编译命令

```bash
oellm_build \
  --model_name qwen2_5-vl-3b \
  --march nash-p \
  --input_model_path /home/kangjie.xu/oe_locateanything/main/language/baseline_weights/Qwen2.5-VL-3B-Instruct \
  --output_model_path /home/kangjie.xu/oellm_clean/output \
  --w_bits 4 \
  --chunk_size 256 \
  --cache_len 1024 \
  --device cuda:0 \
  --vit_core_num 4 \
  --prefill_core_num 4 \
  --decode_core_num 4 \
  --jobs 16
```

注：`--calib_json_path` 未指定，走 SDK 默认的 mmstar/conversation.json（120 条图文校准数据）。

## 3. 输入权重

| 文件 | 来源 | MD5 |
|------|------|-----|
| model-00001-of-00002.safetensors (3.8G) | modelscope: Qwen/Qwen2.5-VL-3B-Instruct | `b0f76da6be4bd3d5135ec107bf28a224` |
| model-00002-of-00002.safetensors (3.3G) | modelscope: Qwen/Qwen2.5-VL-3B-Instruct | `0fc7e1a32d6f1e4657dcffa7477084fc` |

上述 MD5 与 HuggingFace 官方 `Qwen/Qwen2.5-VL-3B-Instruct` 完全一致，权重来源无问题。

## 4. 编译产物对比

| 文件 | 自编译大小 | 官方大小 | 差异 | 自编译 MD5 | 官方 MD5 |
|------|-----------|---------|------|-----------|---------|
| language hbm | 1,830,617,080 | 1,825,574,136 | +5,042,944 (+0.28%) | `3b6e633a30c4ef84ab0b17b5b198d0fe` | `867cf00ce9f9e181443685502adfd1b1` |
| vision hbm | 762,029,104 | 762,028,080 | +1,024 (+0.0001%) | `3bd2a975b0bbe36397823eed83fd8fdf` | `2336e2fa39db22e5a9d11e1361354b1d` |
| embed_tokens | 622,329,856 | 622,329,856 | 0 | `db2b8cac8590332654c72cbab881b48d` | `78da6d69f0dc5b10c27378aee8605db1` |

## 5. embed_tokens.bin 数值差异

- 自编译 embed 与 raw safetensors 中 `model.embed_tokens.weight` 提取的 fp16 数组**完全一致**（逐元素 max diff = 0.0）
- 官方预编译 embed 与 raw safetensors 不一致（max diff = 0.26757812）
- 说明官方预编译的 embed 经过了某种预处理，而非直接从 HF 权重提取

## 6. 推理结果对比

| 模型来源 | vlm_demo 输出 |
|----------|--------------|
| 官方预编译全套 | ✅ 正常：`"The image shows a close-up of a cat lying down..."` |
| 自编译全套 | ❌ 乱码：`"and cat a哭了、1 -2 -2 - (2 - 2.0 -2..."` |

推理环境：S600 BPU，同一台设备，同一套 lib（`oellm_runtime/lib/libhbrt4.so`），同一个 `vlm` 二进制。

## 7. 编译日志关键信息

```
miss_key: ['model.visual.patch_embed.proj_2d.weight', 'model.language_model.cache_cos', 'model.language_model.cache_sin']
unexpected_key: []
```

- `proj_2d.weight` miss 为正常行为（`Qwen2_5_VisionPatchEmbed.forward()` 中从 `proj.weight` 自动计算）
- `cache_cos`/`cache_sin` miss 为正常行为（运行时从 rope_theta 计算）

编译全过程无 ERROR、无 WARN（除 `torch_dtype` deprecation 外）。

## 8. 编译各阶段耗时

| 阶段 | 耗时 |
|------|------|
| build (加载权重) | 7.8s |
| export visual | 1.6s |
| export prefill | 7.9s |
| export decode | 7.0s |
| convert_mlir visual | 7.7s |
| compile_hbo visual | 9273s (2.6h) |
| convert_mlir prefill | 23.9s |
| compile_hbo prefill | 4122s (1.1h) |
| convert_mlir decode | 23.7s |
| compile_hbo decode | 4599s (1.3h) |
| link_models | 74.6s + 25.5s |
| **总计** | **~5.1 小时** |

## 9. 待确认问题

1. 官方编译 Qwen2.5-VL-3B 时使用的 `oellm_build` 完整参数是什么？
2. 官方预编译的 embed_tokens.bin 为什么和 HF 公开权重不一致（max diff 0.27）？是否经过了额外预处理？
3. 自编译的 language hbm 比官方大 5MB，vision hbm 大 1KB，这是编译器的非确定性行为还是参数差异？
4. 上述编译命令和参数是否正确？是否有遗漏的必需参数？