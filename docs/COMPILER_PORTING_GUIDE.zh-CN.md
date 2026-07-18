# 从零构建 LocateAnything-3B S600 HBM

本文记录 LocateAnything-3B 从原始 checkpoint 到 D-Robotics S600 HBM 的完整
工程路径，包括编译链验证、模型适配、量化隐藏域、BC/HBO/HBM 构建和板端验证。

## 1. 目标与方法

LocateAnything-3B 的部署由四个相互依赖的部分组成：

1. MoonViT 将图像编码为视觉 token；
2. projector 将 MoonViT hidden 1152 映射到 Language hidden 2048；
3. Qwen2.5 decoder 执行 prefill、PBD decode 和 AR decode；
4. Host runtime 负责 tokenizer、视觉 token 插入、mask、position IDs、KV cache、
   Hybrid 采样与 `<ref>/<box>` 解析。

项目先使用 Qwen2.5-VL-3B 建立可工作的 S600 编译与运行参考链路。它覆盖 OELLM
模型加载、Vision/Language HBM、embedding table、HBRT 运行时和图文语义验证，
便于将编译器问题与 LocateAnything 特有的 MoonViT/PBD 适配问题分层处理。

## 2. 环境与目录

| 角色 | 地址 | 目录 |
|---|---|---|
| 4090 编译机 | `kangjie.xu@10.112.20.45` | `/home/kangjie.xu/oe_locateanything` |
| S600 部署机 | `sunrise@10.112.133.20` | `/home/sunrise/oe_locateanything` |
| 原始源码 | Windows workspace | `Eagle/Embodied` |
| OELLM 环境 | 4090 | Conda `oellm_clean`, Python 3.10 |

推荐目录职责：

```text
oe_locateanything/
├── baselines/qwen2_5_vl/       编译链验证材料
├── docs/                       教程、架构、RCA、问题记录
├── main/scripts/               BC/HBM 构建与数值验证入口
├── main/runtime/               S600 Host runtime
├── main/outputs/               生成产物，不入 Git
└── toolchain/leap_llm/         SDK 源码和模型适配
```

## 3. 第一阶段：Qwen2.5-VL 编译链验证

### 3.1 验证范围

Qwen2.5-VL 验证链覆盖：

- Hugging Face checkpoint 到 Leap 模型的权重映射；
- Vision BC/HBM 与 Language BC/HBM 编译；
- embedding table 导出；
- S600 `libxlm/HBRT` 加载和执行；
- 纯文本与图像问答语义验证；
- 官方参考产物与自编译产物的单变量对比。

D-Robotics 开发者论坛的 Gemma4 部署文章提供了自定义模型构建的实践参考。本项目
在 Qwen2.5-VL 和 LocateAnything checkpoint 上重新完成权重、图、量化域与板端
行为验证，并将过程固化为脚本和 RCA 记录。

### 3.2 静态图像 Patch Embedding

Qwen2.5-VL 原始 Vision 使用 Conv3d patch embedding。静态图像在 temporal 维复制
两份后进入卷积，编译模型则使用 Conv2d 路径。因此等价权重应为：

```python
weight_2d = weight_5d.sum(dim=2)
```

对应修改位于：

```text
toolchain/leap_llm/models/qwen2_5_vl/model.py
toolchain/leap_llm/nn/modules/vision_embedding.py
```

权重在 checkpoint 加载阶段完成折叠，避免 calibration forward 再次覆盖已经映射的
`proj_2d.weight`。

### 3.3 2048 维隐藏域

参考 embedding 与原始 checkpoint embedding 的差异可由一个正交矩阵 `Q` 描述：

```text
E_reference ~= E_checkpoint @ Q
Q.T @ Q = I
```

该矩阵可以精确表示为 2048 阶归一化 Walsh-Hadamard 矩阵与确定性的行符号。
恢复出的精确矩阵用于完成 Vision、Language 和 embedding table 的统一隐藏域变换。

### 3.4 Vision 端折叠

对 row-vector 形式的 hidden state，Vision projector 原输出为：

```text
y = x @ W.T + b
```

目标输出为 `y @ Q`，可将变换折叠到最后一层：

```text
W' = Q.T @ W
b' = b @ Q
```

运行时图仍只包含原 projector，不增加 2048x2048 MatMul。

### 3.5 Language 端折叠

Language residual stream 统一进入 `Q` 域。对每层 RMSNorm 和线性层执行：

```text
Embedding:          E'     = E @ Q
Q/K/V input:        W_in' = W_in @ diag(gamma) @ Q
Attention output:   W_o'  = Q.T @ W_o
MLP gate/up input:  W_m'  = W_m @ diag(gamma) @ Q
MLP down output:    W_d'  = Q.T @ W_d
Final lm_head:      W_lm' = W_lm @ diag(gamma) @ Q
```

被折叠的 norm weight 设置为 1。Q/K/V 和 logits 的数学语义保持不变，residual hidden
则始终位于旋转域中。实现位于：

```text
toolchain/leap_llm/models/locateanything/hidden_rotation.py
```

Qwen2.5-VL 的板端最终验证使用自编译 Vision HBM、自编译 Language HBM 和自生成
embedding table，纯文本与图像语义均正常。完整实验记录见：

```text
docs/rca/sdk_compiler_rca_review.md
baselines/qwen2_5_vl/
```

## 4. 第二阶段：LocateAnything 源码与 Checkpoint 审计

### 4.1 Checkpoint 合同

| 组件 | 配置 |
|---|---|
| Language | Qwen2.5/Qwen2, 36 layers, hidden 2048, MLP 11008 |
| Attention | 16 Q heads, 2 KV heads, head dim 128 |
| RoPE | 1D, theta 1,000,000 |
| Vocabulary | 152,681, tied embeddings |
| PBD | block size 6, text-mask token 151676 |
| Vision | MoonViT, 27 layers, hidden 1152, patch 14 |
| Projector | 2x2 merge, 4608 -> 2048 -> 2048 |

模型配置与 checkpoint remote code 是编译适配的主要依据。上游
`eaglevl/utils/locany` 提供 Hybrid PBD 推理逻辑，`eaglevl/model/moon_vit` 提供
MoonViT 定义。详细审计见 [SOURCE_REVIEW.md](SOURCE_REVIEW.md)。

### 4.2 PBD 与 Hybrid 语义

MTP/PBD 每轮准备 6 个位置：一个真实尾 token 与 5 个 `<text_mask>` token，并对
最后 6 个 position IDs 执行 `-1` 偏移。Hybrid 模式根据 box pattern 在 PBD 与
AR 之间切换，因此 Language HBM 需要同时提供：

```text
prefill    q=1024
decode     q=6
decode_ar  q=1
```

## 5. 第三阶段：LocateAnything Leap 适配

### 5.1 模型注册与配置

新增模型入口：

```text
locateanything-lm-3b
locateanything-vit-3b
locateanything-3b
```

配置 dataclass 从 checkpoint `config.json` 读取 MoonViT、Qwen2、token IDs、PBD
block size 和 compile-time profile，避免将 Qwen2.5-VL 的 M-RoPE 配置混入 LA。

### 5.2 MoonViT Vision 图

Vision 适配包含：

1. Conv2d patch weight 展平为 `Linear(588, 1152)`；
2. 64x64 learnable position embedding 插值到固定 32x32 patch grid；
3. 27 层 MoonViT global attention 与 2D RoPE；
4. 2x2 patch merge，将 1024 token 合并为 256 token；
5. `4608 -> 2048 -> 2048` projector；
6. 在 projector 最后一层折叠公共隐藏域矩阵。

固定 448x448 profile 的图接口为：

```text
input  (1, 1024, 588) fp16
output (1, 256, 2048) fp16
```

### 5.3 Qwen2.5 Language 图

Language 适配保留 36 层 Qwen decoder，并针对 LA 调整：

- 使用 1D RoPE position IDs `(batch, 1, sequence)`；
- attention mask 由 Host runtime 输入，以支持 causal、PBD block 和 cache；
- vocabulary 固定为 152,681；
- tied embedding 同时生成 `embed_tokens.bin` 和 exportable lm_head；
- embedding、Attention/MLP、final norm/lm_head 应用公共隐藏域折叠；
- 输出 `prefill`、`decode` 和 `decode_ar` 三个图。

### 5.4 lm_head 的 Leap 导出

HBDK export 期间，hidden state 是 Leap `OpResult`。输出投影需要实现 `build()` 的
Leap 模块，因此 lm_head 使用 `DynamicQuantLinear`。Checkpoint 加载完成后，将
`embed_tokens.weight` 复制到 lm_head，保持 tied embedding 语义。

### 5.5 compile_mode 传播

SDK `Module.compile_mode()` 默认递归自定义 `Module` 与 `ModuleList`。MoonViT
projector 使用 `torch.nn.Sequential`，因此 `LocateAnythingVisionPatchMerger` 显式
将 compile/eager 模式传播给其中的 Leap Linear，确保 calibration 和 PyTorch 数值
验证调用 `forward()`，BC export 调用 `build()`。

### 5.6 关键修改对照

下表源码路径均相对仓库根目录：

| 文件 | 修改 | 原因 |
|---|---|---|
| `toolchain/leap_llm/models/qwen2_5_vl/model.py` | 在 checkpoint 加载阶段将 Conv3d temporal 权重求和到 Conv2d | 保持静态图像 temporal duplication 的等价语义 |
| `toolchain/leap_llm/nn/modules/vision_embedding.py` | 移除 forward 中对 `proj_2d.weight` 的重复覆盖 | 保证加载后的确定性权重贯穿 calibration 与 export |
| `toolchain/leap_llm/models/locateanything/config/locateanything_3b.py` | 从 LA checkpoint 解析 MoonViT、Qwen2、PBD 和 token IDs | 让编译配置与真实 checkpoint 合同一致 |
| `toolchain/leap_llm/models/locateanything/hidden_rotation.py` | 构造 signed Hadamard 并折叠 Language/Vision 权重 | 统一 embedding、residual stream 与 projector 的量化隐藏域 |
| `toolchain/leap_llm/models/locateanything/text_model_leap.py` | 使用 1D RoPE、Host mask、tied `DynamicQuantLinear` lm_head | 对齐 LA decoder 与 Leap export 接口 |
| `toolchain/leap_llm/models/locateanything/vision_model_leap.py` | 实现 MoonViT、2D RoPE、2x2 merge 与 projector | 生成 LA 原生视觉 token |
| `toolchain/leap_llm/models/locateanything/blocks/vision_patch_merger_leap.py` | 显式传播 `compile_mode()` 到 Sequential 子模块 | 统一 eager calibration 与 HBDK build 路径 |
| `toolchain/leap_llm/apis/model/locateanything_language.py` | 导出 `prefill`、`decode`、`decode_ar` 并原子写 embedding | 支持 PBD/Hybrid 图合同并避免旧 embedding 残留 |
| `toolchain/leap_llm/apis/model/locateanything_vision.py` | 加载 MoonViT 权重、插值 pos embedding、折叠输出隐藏域 | 形成可独立验证的固定分辨率 Vision 图 |
| `toolchain/leap_llm/apis/oellm_build.py` | 增加 hidden rotation 与 export-only 参数 | 支持 BC 预检和受控 RCA |
| `main/scripts/compile_*.sh` | 使用 `setsid + nohup`、独立日志和 PID | 让长时间 HBM 编译稳定脱离 SSH 会话 |

## 6. 从零构建

正式 HBM 构建前必须先完成 LocateAnything 专用校准。通用 VLM 问答数据不能自动
覆盖 grounding、坐标 token、PBD q=6 与 AR q=1 的激活分布；同时，日志中出现
`calib_json_path` 也不等于模型 API 已消费该数据。数据组成、隔离策略、scale audit
和验收门槛见 [LocateAnything Calibration Strategy](CALIBRATION.md)。

### 6.1 获取代码和权重

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
git clone https://github.com/NVlabs/Eagle.git eagle

hf download nvidia/LocateAnything-3B \
  --local-dir eagle/Embodied/LocateAnything-3B
```

### 6.2 安装 OELLM 编译环境

将 D-Robotics S600 OELLM 1.0.5 SDK 放到 `oellm/s600_sdk/`，安装 SDK wheel 和
依赖后执行：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

cd ~/oe_locateanything/toolchain
pip install -e . --no-deps

python -m leap_llm.apis.oellm_build --help | \
  grep -E 'locateanything|hidden_rotation|export_only'
```

### 6.3 验证隐藏域等价性

```bash
PYTHONPATH=~/oe_locateanything/toolchain \
python ~/oe_locateanything/main/scripts/validate_locateanything_rotation.py \
  --model-path ~/oe_locateanything/eagle/Embodied/LocateAnything-3B \
  --component all \
  --device cuda:0 \
  --dtype float32
```

参考验证结果：

```text
Language logits cosine: 0.999999999986
Language KV max diff:   6.109476e-05
Vision output cosine:   0.999999927
```

### 6.4 导出 Language BC

```bash
PYTHONPATH=~/oe_locateanything/toolchain \
python -m leap_llm.apis.oellm_build \
  --model_name locateanything-lm-3b \
  --march nash-p \
  --input_model_path ~/oe_locateanything/eagle/Embodied/LocateAnything-3B \
  --output_model_path ~/oellm_clean/output/la_language_export \
  --w_bits 4 \
  --chunk_size 1024 \
  --cache_len 2048 \
  --decode_seq_len 6 \
  --device cuda:0 \
  --prefill_core_num 4 \
  --decode_core_num 4 \
  --jobs 16 \
  --export_only
```

预期图接口：

| Graph | Input embeds | Logits output |
|---|---|---|
| `prefill` | `(1,1024,2048)` | `(1,1024,152681)` |
| `decode` | `(1,6,2048)` | `(1,6,152681)` |
| `decode_ar` | `(1,1,2048)` | `(1,1,152681)` |

每个 Language 图另有 position IDs、attention mask、72 个 KV 输入和 72 个 KV 输出。

### 6.5 导出 Vision BC

```bash
PYTHONPATH=~/oe_locateanything/toolchain \
python -m leap_llm.apis.oellm_build \
  --model_name locateanything-vit-3b \
  --march nash-p \
  --input_model_path ~/oe_locateanything/eagle/Embodied/LocateAnything-3B \
  --output_model_path ~/oellm_clean/output/la_vision_export \
  --w_bits 8 \
  --image_width 448 \
  --image_height 448 \
  --device cuda:0 \
  --vit_core_num 4 \
  --jobs 16 \
  --export_only
```

### 6.6 后台编译 HBM

```bash
cd ~/oe_locateanything
export REPO_ROOT=$PWD
export CONDA_ENV=oellm_clean

./main/scripts/compile_locateanything_language.sh
tail -f main/logs/locateanything_language_compile.log

# Language 完成后启动
./main/scripts/compile_locateanything_vit.sh
tail -f main/logs/locateanything_vit_compile.log
```

脚本使用 `setsid + nohup + stdin=/dev/null`，并设置 unbuffered Python 日志。受控
实验通过 `OUTPUT_MODEL_PATH` 指向独立目录，便于保存 BC/HBO/HBM 与 checksum：

```bash
OUTPUT_MODEL_PATH=~/oellm_clean/output/la_fix012_language \
  ./main/scripts/compile_locateanything_language.sh
```

## 7. S600 部署

### 7.1 记录和传输产物

```bash
sha256sum LocateAnything-3B_*.hbm LocateAnything-3B_embed_tokens.bin

scp LocateAnything-3B_*.hbm \
  LocateAnything-3B_embed_tokens.bin \
  sunrise@10.112.133.20:~/oe_locateanything/oellm_runtime/model/LocateAnything-3B/fix011/
```

在 S600 上重新执行 `sha256sum`，确认传输前后完全一致。

### 7.2 运行时合同

固定 448x448 profile 需要：

- 256 个视觉 token 插入到 token ID `151665` 对应位置；
- vocab 152,681，hidden 2048；
- prefill chunk 1024，cache 2048；
- PBD mask 与 q=6 position IDs；
- Hybrid 模式的 q=1 AR 图；
- `<ref>/<box>` 与坐标 token 解析。

板端运行旧 chunk-1024 图时使用：

```bash
export HB_DNN_USER_DEFINED_L2M_SIZES=8:8:8:8
```

## 8. 验证标准

按以下顺序推进，每一级都保留输入、输出、checksum 和日志：

1. **Checkpoint**：关键权重全部加载，missing/unexpected keys 在允许集合内；
2. **PyTorch**：隐藏域变换前后 logits、KV、Vision output 数值等价；
3. **BC**：图名、输入输出 shape、dtype、op 数正确；
4. **HBM**：HBM 可加载，图合同与 BC 一致；
5. **单图数值**：同输入下 HBM 与 PyTorch Vision/Language 对齐；
6. **板端语义**：文本、图像描述和 grounding box 输出正确；
7. **PBD/Hybrid**：q=6、q=1 切换及 box pattern fallback 正确；
8. **性能**：在固定模型、图片、prompt、采样参数和计时口径下统计 TPS 与 BPS。

## 9. 关键源码

| 路径 | 作用 |
|---|---|
| `toolchain/leap_llm/models/locateanything/hidden_rotation.py` | 公共隐藏域构造与权重折叠 |
| `toolchain/leap_llm/models/locateanything/text_model_leap.py` | Qwen2.5 Language 与 lm_head |
| `toolchain/leap_llm/models/locateanything/vision_model_leap.py` | MoonViT Vision 图 |
| `toolchain/leap_llm/apis/model/locateanything_language.py` | prefill/PBD/AR BC-HBM pipeline |
| `toolchain/leap_llm/apis/model/locateanything_vision.py` | Vision BC-HBM pipeline |
| `main/scripts/validate_locateanything_rotation.py` | 变换前后数值等价性测试 |
| `main/scripts/compile_locateanything_language.sh` | Language 后台编译 |
| `main/scripts/compile_locateanything_vit.sh` | Vision 后台编译 |

更细的实验过程和失败分析保存在 `docs/rca/` 与 `docs/KNOWN_ISSUES.md`。
