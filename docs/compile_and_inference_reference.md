# Qwen2.5-VL-3B 编译 & 推理 & LA 适配记录

## 1. 编译参数（4090 x86 主机）

### 1.1 编译命令

```bash
source /home/kangjie.xu/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

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

### 1.2 后台编译（防 SSH 断开）

```bash
nohup oellm_build \
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
  --jobs 16 \
  > /home/kangjie.xu/oellm_clean/compile.log 2>&1 &
```

### 1.3 编译产物

| 文件 | 说明 |
|------|------|
| `Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm` | 语言模型 HBM |
| `Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm` | 视觉模型 HBM |
| `Qwen2.5-VL-3B-Instruct_embed_tokens.bin` | 词表嵌入权重 |

### 1.4 编译日志关键位置

```
miss_key: ['model.visual.patch_embed.proj_2d.weight', 'model.language_model.cache_cos', 'model.language_model.cache_sin']
```

- `proj_2d.weight` miss 是正常的（`forward()` 里从 `proj.weight` 自动计算）
- `cache_cos` / `cache_sin` miss 是正常的（运行时从 `rope_theta` 计算）

### 1.5 编译脚本位置

- 我们自写的编译脚本：`/home/kangjie.xu/oe_locateanything/main/scripts/compile_baseline_qwen2_5-vl-3b.sh`
- SDK 提供的编译工具：`oellm_build`（`pip install leap_llm-1.0.5-py3-none-any.whl` 后可用）
- 交叉编译 C++ demo 脚本（官方）：`oellm_runtime/examples/vlm_demo/build_vlm.sh`

### 1.6 已知问题

- 官方文档写明："If you need to quantize and compile large models yourself, please contact D-Robotics technical support"
- 官方 SDK 未提供 `oellm_build` 编译 Qwen2.5-VL-3B 的官方脚本/参数
- 自编译的 embed 和 raw safetensors 权重完全一致（MD5: `db2b8cac...`）
- 官方预编译的 embed 和 raw safetensors 权重不一致（MD5: `78da6d69...`, max diff=0.27）
- 自编译的 hbm 在 S600 上跑 vlm_demo 输出乱码，官方预编译 hbm 正常

---

## 2. 推理命令（S600 端侧）

### 2.1 官方 Qwen3-0.6B LLM（纯文本）

```bash
cd ~/oe_locateanything/oellm_runtime/examples/llm_demo
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
echo "你是谁？" | ./llm -c qwen3_0.6b_config.json
```

模型文件：`../../model/Qwen3_0.6B/w8/Qwen3-0.6B_language_chunk_256_cache_2048_w8_nash-p_corenum_4_4.hbm`
（从 OSS 下载：`https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/models/Qwen3-0.6B/w8/...`）

### 2.2 官方 Qwen2.5-VL-3B VLM（图文）

```bash
cd ~/oe_locateanything/oellm_runtime/examples/vlm_demo
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6

# 纯文本
echo "what is a cat?" | ./vlm -c qwen2.5vl_3b_config.json

# 带图片
./vlm -c qwen2.5vl_3b_config.json -i /tmp/test-cat.jpg
```

模型文件（官方预编译，从 OSS 下载）：
```bash
cd ~/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/w4/
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/models/Qwen2.5-VL-3B-Instruct/w4/Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/models/Qwen2.5-VL-3B-Instruct/w4/Qwen2.5-VL-3B-Instruct_vision_448x448_w8-4_nash-p_corenum_4.hbm
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/models/Qwen2.5-VL-3B-Instruct/w4/Qwen2.5-VL-3B-Instruct_embed_tokens_w4_fp16.bin
```

### 2.3 Config 文件

```json
{
  "model_type": "Qwen2.5-VL",
  "model_dir": "../../model/Qwen2.5-VL-3B-Instruct/w4/",
  "vit_model_file": "Qwen2.5-VL-3B-Instruct_vision_448x448_w8-4_nash-p_corenum_4.hbm",
  "llm_model_file": "Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm",
  "embed_weight_file_path": "Qwen2.5-VL-3B-Instruct_embed_tokens_w4_fp16.bin",
  "vit_bpu_core": [0,1,2,3],
  "prefill_bpu_core": [0,1,2,3],
  "decode_bpu_core": [0,1,2,3],
  "vocabulary_path": "../../configs/Qwen2.5_VL_config",
  "vocab_size": 151936,
  "embed_dim": 2048,
  "image_height": 448,
  "image_width": 448,
  "mask_pad_value": -32768
}
```

---

## 3. LA 适配方案

### 3.1 现状

| 组件 | Qwen2.5-VL-3B | LA LocateAnything-3B | 差异 |
|------|---------------|---------------------|------|
| Language 架构 | 36层, hidden=2048, GQA 16/2 | 36层, hidden=2048, GQA 16/2 | 结构相同 |
| Vision 架构 | Qwen2.5-VL ViT | MoonViT | 完全不同 |
| Vocab | 151936 | 152681 | +745 个 special token |
| Embed | 151936×2048 | 152681×2048 | 多 745 行 |
| LM Head | 151936×2048 | 152681×2048 | 多 745 行 |
| 注意力 | MRoPE (3D) | Vanilla 1D RoPE | 位置编码不同 |
| PBD | 无 | block_size=6, causal_attn=False | 并行块解码 |

### 3.2 方案：官方预编译 hbm 做基准，替换 LA 权重

**核心思路**：LA 的 language 塔结构和 Qwen2.5-VL-3B 完全一样（36层, hidden=2048, GQA 16/2），可以复用官方预编译的 language hbm，只替换 embed 和 lm_head 权重。

#### Step 1：验证"官方 hbm + LA embed"纯文本

把官方预编译的 language hbm + LA 的 embed_tokens.bin 放 S600 上，跑纯文本推理。

预期：vocab 前 151936 个 token 应该能正常响应，后 745 个 LA 专属 token 会是 OOV。

#### Step 2：处理 vision 塔

LA 的 MoonViT 不能复用 Qwen2.5-VL 的 vision hbm。方案：
- **A**：编译 LA 的 MoonViT vision hbm
- **B**：S600 上 CPU/GPU 跑 PyTorch vision 前向

#### Step 3：处理 lm_head vocab 差异

vlm_demo 的 config 里 `vocab_size: 151936`，LA 需要 152681。需要改 config，同时确保 lm_head 覆盖 152681 个 token。

#### 待确认问题

- 官方 RoPE 是 MRoPE (3D position_ids)，LA 是 1D RoPE……这个差异是否影响 language hbm 的复用？
- 官方 hbm 的 decode 是 `decode_seq_len=1`，LA 需要 PBD `decode_seq_len=6`……这个差异怎么处理？
- LA 的 `causal_attn=False`（PBD 窗口内双向注意力）……官方 hbm 是标准 causal，这个差异怎么处理？

---

## 4. 已知问题索引

| 编号 | 问题 | 状态 |
|------|------|------|
| #001 | README 中文乱码（SSH heredoc） | ✅ 已修复 |
| #002 | LA lm_head DynamicQuantLinear 全 0 | ✅ 已定位 |
| #003 | LA quant_input_embeds 量化域不匹配 | ⚠️ Patch 无效 |
| #004 | Qwen2.5-VL 自编译 hbm 乱码 | ⚠️ 根因不明（权重一致但 embed 不同） |