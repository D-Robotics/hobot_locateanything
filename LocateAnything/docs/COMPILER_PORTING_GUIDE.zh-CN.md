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
| 4090 编译机 | `kangjie.xu@10.112.20.45` | `/home/kangjie.xu/oe_locateanything/LocateAnything` |
| S600 部署机 | `sunrise@10.112.133.20` | `/home/sunrise/oe_locateanything/LocateAnything` |
| 原始源码 | Windows workspace | `Eagle/Embodied` |
| OELLM 环境 | 4090 | Conda `oellm_clean`, Python 3.10 |

推荐目录职责：

```text
LocateAnything/
├── docs/                       当前技术文档与问题记录
├── compiler/                   统一量化入口、OELLM 适配与内部脚本
├── deploy/                    S600 Host Runtime
├── workspace/                  模型、校准数据和生成产物
└── src/oe_locateanything/      共享路径与项目辅助代码
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
compiler/leap_llm/models/qwen2_5_vl/model.py
compiler/leap_llm/nn/modules/vision_embedding.py
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
compiler/leap_llm/models/locateanything/hidden_rotation.py
```

Qwen2.5-VL 的板端最终验证使用自编译 Vision HBM、自编译 Language HBM 和自生成
embedding table，纯文本与图像语义均正常。完整实验记录见：

```text
../Qwen-2.5-VL-3B/docs/QWEN2_5_VL_BASELINE.md
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
AR 之间切换。当前发布**编译合同**包含 13 张 Language 候选图：

```text
prefill    q=1024
decode     q=6
decode_ar  q=1
decode_pbd_q7 ... decode_pbd_q12
decode_ar_q2  ... decode_ar_q5
```

后三组融合图用于合并已接受 token 的 KV 提交与下一步 PBD/AR 计算；它们不改变
基础 PBD q=6 和 AR q=1 的模型语义。已验证的板端历史 HBM 只有
`prefill`、`decode`、`decode_ar` 三图；13 图候选必须完成完整 BC/HBM、S600 数值和
六任务验证后才能提升为部署版本。

## 5. 第三阶段：LocateAnything Leap 适配

### 5.1 模型注册与配置

新增模型入口：

```text
locateanything-lm-3b
locateanything-vit-3b
```

Language 与 Vision 独立编译并在 Runtime 中组合。早期统一入口只完成浮点 sanity，
不生成可部署 HBM，现已退出生产注册表。

配置 dataclass 从 checkpoint `config.json` 读取 MoonViT、Qwen2、token IDs、PBD
block size 和 compile-time profile，避免将 Qwen2.5-VL 的 M-RoPE 配置混入 LA。

### 5.2 MoonViT Vision 图

Vision 适配包含：

1. Conv2d patch weight 展平为 `Linear(588, 1152)`；
2. 64x64 learnable position embedding 插值到固定 48x48 patch grid；
3. 27 层 MoonViT global attention 与 2D RoPE；
4. 2x2 patch merge，将 2304 token 合并为 576 token；
5. `4608 -> 2048 -> 2048` projector；
6. 在 projector 最后一层折叠公共隐藏域矩阵。

固定 672x672 profile 的图接口为：

```text
input  (1, 2304, 588) fp16
output (1, 576, 2048) fp16
```

### 5.3 Qwen2.5 Language 图

Language 适配保留 36 层 Qwen decoder，并针对 LA 调整：

- 使用 1D RoPE position IDs `(batch, 1, sequence)`；
- attention mask 由 Host runtime 输入，以支持 causal、PBD block 和 cache；
- vocabulary 固定为 152,681；
- tied embedding 同时生成 `embed_tokens.bin` 和 exportable lm_head；
- embedding、Attention/MLP、final norm/lm_head 应用公共隐藏域折叠；
- 输出 `prefill`、`decode`、`decode_ar` 以及 10 张融合/bridge 图，共 13 张
  Language 候选图。

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
| `compiler/leap_llm/models/qwen2_5_vl/model.py` | 在 checkpoint 加载阶段将 Conv3d temporal 权重求和到 Conv2d | 保持静态图像 temporal duplication 的等价语义 |
| `compiler/leap_llm/nn/modules/vision_embedding.py` | 移除 forward 中对 `proj_2d.weight` 的重复覆盖 | 保证加载后的确定性权重贯穿 calibration 与 export |
| `compiler/leap_llm/models/locateanything/config/locateanything_3b.py` | 从 LA checkpoint 解析 MoonViT、Qwen2、PBD 和 token IDs | 让编译配置与真实 checkpoint 合同一致 |
| `compiler/leap_llm/models/locateanything/hidden_rotation.py` | 构造 signed Hadamard 并折叠 Language/Vision 权重 | 统一 embedding、residual stream 与 projector 的量化隐藏域 |
| `compiler/leap_llm/models/locateanything/text_model_leap.py` | 使用 1D RoPE、Host mask、tied `DynamicQuantLinear` lm_head | 对齐 LA decoder 与 Leap export 接口 |
| `compiler/leap_llm/models/locateanything/vision_model_leap.py` | 实现 MoonViT、2D RoPE、2x2 merge 与 projector | 生成 LA 原生视觉 token |
| `compiler/leap_llm/models/locateanything/blocks/vision_patch_merger_leap.py` | 显式传播 `compile_mode()` 到 Sequential 子模块 | 统一 eager calibration 与 HBDK build 路径 |
| `compiler/leap_llm/apis/model/locateanything_language.py` | 导出 `prefill`、`decode`、`decode_ar` 并原子写 embedding | 支持 PBD/Hybrid 图合同并避免旧 embedding 残留 |
| `compiler/leap_llm/apis/model/locateanything_vision.py` | 加载 MoonViT 权重、插值 pos embedding、折叠输出隐藏域 | 形成可独立验证的固定分辨率 Vision 图 |
| `compiler/leap_llm/apis/oellm_build.py` | 增加 hidden rotation 与 export-only 参数 | 支持 BC 预检和受控 RCA |
| `compiler/quantize.py` | 统一调度 prepare、calibrate、build 与 verify | 对外隐藏内部脚本和环境变量合同 |
| `compiler/scripts/build/language.sh`、`vision.sh` | 执行组件级 BC/HBM 构建并记录日志 | 作为统一入口的内部构建实现 |

## 6. 从零构建

正式 HBM 构建前必须先完成 LocateAnything 专用校准。通用 VLM 问答数据不能自动
覆盖 grounding、坐标 token、PBD q=6 与 AR q=1 的激活分布；同时，日志中出现
`calib_json_path` 也不等于模型 API 已消费该数据。数据组成、隔离策略、scale audit
和验收门槛见 [LocateAnything Calibration Strategy](CALIBRATION.md)。

当前发布校准集固定为 1200 条：Detection 620 条，其余 580 条覆盖 GUI、Referring、
OCR、Layout 和 Pointing。512 条仅作为 Scale 收敛检查点，最终 BC/HBM 使用完整
1200 条产生的 Scale。

### 6.1 获取代码和权重

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
git clone https://github.com/NVlabs/Eagle.git eagle

hf download nvidia/LocateAnything-3B \
  --local-dir eagle/Embodied/LocateAnything-3B
```

### 6.2 安装 OELLM 编译环境

安装 D-Robotics S600 OELLM 1.0.5 SDK wheel 和依赖后执行：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

cd ~/oe_locateanything/LocateAnything/compiler
pip install -e . --no-deps

cd ..
python compiler/quantize.py --help
```

### 6.3 准备校准输入并验证发布合同

```bash
cd ~/oe_locateanything/LocateAnything
python compiler/quantize.py prepare
python compiler/quantize.py calibrate
python compiler/quantize.py verify --component all --level contract
```

`verify --level contract` 检查数据清单、Scale、图覆盖和固定 profile，不等同于重新
执行隐藏域数值实验。内部 rotation validator 得到的参考结果为：

```text
Language logits cosine: 0.999999999986
Language KV max diff:   6.109476e-05
Vision output cosine:   0.999999927
```

### 6.4 导出 BC

```bash
python compiler/quantize.py build --component all --target bc
```

预期图接口：

| Graph | Input embeds | Logits output |
|---|---|---|
| `prefill` | `(1,1024,2048)` | `(1,1024,152681)` |
| `decode` | `(1,6,2048)` | `(1,6,152681)` |
| `decode_ar` | `(1,1,2048)` | `(1,1,152681)` |
| `decode_pbd_q7..q12` | `(1,7..12,2048)` | `(1,7..12,152681)` |
| `decode_ar_q2..q5` | `(1,2..5,2048)` | `(1,2..5,152681)` |

每个 Language 图另有 position IDs、attention mask、72 个 KV 输入和 72 个 KV 输出。

### 6.5 Vision BC 合同

Vision 使用固定 672x672 输入、W8 权重和四个 BPU core。`build --component all`
会在同一份配置下导出 Language 图族和 `visual` 图，避免两个组件使用不同 profile。

### 6.6 编译并验证 HBM

```bash
cd ~/oe_locateanything/LocateAnything
python compiler/quantize.py build --component all --target hbm --resume
python compiler/quantize.py verify --component all --level all
```

统一入口按照 `compiler/config.yaml` 顺序构建 Vision 与 Language，并将日志写入
`workspace/logs/`。受控实验通过 `--output-dir` 指向独立目录，便于保存
BC/HBO/HBM 与 checksum：

```bash
python compiler/quantize.py build --component language --target hbm \
  --output-dir ~/oellm_clean/output/la_fix012 --resume
```

## 7. S600 部署

### 7.1 记录和传输产物

```bash
sha256sum LocateAnything-3B_*.hbm LocateAnything-3B_embed_tokens.bin

# 4090 -> Windows relay -> S600；每一跳完成后重新计算 SHA256。
# 板端目标目录：
# 仅在候选通过全部发布检查后才提升到：
# ~/oe_locateanything/LocateAnything/workspace/artifacts/release/<release>/
```

在 S600 上重新执行 `sha256sum`，确认传输前后完全一致。

### 7.2 运行时合同

固定 672x672 profile 需要：

- 原图按比例 letterbox 到 672x672，并保留缩放与 padding 元数据；
- 576 个视觉 token 插入到 token ID `151665` 对应位置；
- `<box>` 坐标在送入校准前变换到 letterbox 坐标域，输出后再逆变换；
- vocab 152,681，hidden 2048；
- prefill chunk 1024，cache 4096；视觉 token 后最多剩余 448 个位置；
- PBD mask 与 q=6 position IDs；
- Hybrid 模式的 q=1 AR 图；
- `<ref>/<box>` 与坐标 token 解析。

板端运行当前四核图时使用：

```bash
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
```

## 8. 验证标准

按以下顺序推进，每一级都保留输入、输出、checksum 和日志：

1. **Checkpoint**：关键权重全部加载，missing/unexpected keys 在允许集合内；
2. **PyTorch**：隐藏域变换前后 logits、KV、Vision output 数值等价；
3. **BC**：`visual` 和 13 张 Language 图的名称、shape、dtype、op 数正确；
4. **HBM**：HBM 可加载，完整图目录与 BC 一致，不允许缺图后静默回退；
5. **单图数值**：同输入下 HBM 与 PyTorch Vision/Language 对齐；
6. **板端语义**：六任务结构合法，并按标签一对一匹配，在 IoU 0.90 下评估 box；
7. **PBD/Hybrid**：q=6、q=1 切换及 box pattern fallback 正确；
8. **性能**：在固定模型、图片、prompt、采样参数和计时口径下统计 TPS 与 BPS。

## 9. 关键源码

| 路径 | 作用 |
|---|---|
| `compiler/leap_llm/models/locateanything/hidden_rotation.py` | 公共隐藏域构造与权重折叠 |
| `compiler/leap_llm/models/locateanything/text_model_leap.py` | Qwen2.5 Language 与 lm_head |
| `compiler/leap_llm/models/locateanything/vision_model_leap.py` | MoonViT Vision 图 |
| `compiler/leap_llm/apis/model/locateanything_language.py` | prefill/PBD/AR BC-HBM pipeline |
| `compiler/leap_llm/apis/model/locateanything_vision.py` | Vision BC-HBM pipeline |
| `compiler/scripts/validate/rotation.py` | 变换前后数值等价性测试 |
| `compiler/scripts/build/language.sh` | Language 后台编译 |
| `compiler/scripts/build/vision.sh` | Vision 后台编译 |

失败原因、修复依据和复现边界统一记录在 `docs/KNOWN_ISSUES.md`。
