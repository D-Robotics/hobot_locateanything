# Qwen2.5-VL-3B 部署到 D-Robotics S600

## 前言

本文记录 Qwen2.5-VL-3B-Instruct 从模型权重、量化校准、BC/HBO/HBM 编译到
D-Robotics S600 板端运行的完整过程。部署使用 OELLM 1.0.5 与 HBDK 工具链，
Vision、Language 和 embedding 分别生成部署产物，板端使用 SDK 提供的
`vlm/libxlm/HBRT` 完成图文推理。

SDK 提供编译器、运行时和模型接口，但未给出该模型从原始 checkpoint 开始的完整
编译脚本。本文基于模型源码、SDK 接口、运行时张量和社区开发者实践完成适配，并以
PyTorch 数值等价性、S600 单模块输出和端到端语义结果作为验收依据。

## 1. 部署方案总览

| 模块 | 部署方式 | 主要职责 |
|---|---|---|
| Vision Module | Qwen2.5-VL Vision 编译为 Vision HBM | 将 448x448 图像编码为 256 个视觉 token |
| Language Module | Qwen2.5 Decoder 编译为 Language HBM | 执行 Prefill 和单 token Decode |
| Embedding | 导出 FP16 `embed_tokens.bin` | 在 Host 侧将 token ID 映射到 2048 维 embedding |
| Runtime Integration | 使用 SDK `vlm/libxlm` | 图像预处理、tokenizer、M-RoPE、mask、KV cache 和采样 |
| Validation | PyTorch、HBM、端到端分层验证 | 检查模型语义、张量接口和板端输出 |

最终部署产物为：

```text
Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm
Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm
Qwen2.5-VL-3B-Instruct_embed_tokens.bin
```

## 2. 推理流程

```text
JPEG / PNG
  -> libxlm 图像解码与预处理
  -> Vision 输入 [1, 1024, 588] FP16
  -> visual HBM
  -> Vision feature [1, 256, 2048] FP16

text prompt
  -> chat template
  -> tokenizer
  -> input_ids
  -> Host 查询 embed_tokens.bin

text embeddings + Vision feature
  -> 替换 256 个 <|image_pad|> 位置
  -> inputs_embeds / position_ids / attention_mask
  -> prefill HBM
  -> logits + KV cache
  -> decode HBM 循环
  -> sampler
  -> output tokens
```

视觉特征不是在序列尾部直接拼接。Host 先生成包含 256 个 `<|image_pad|>` 的输入
序列，再使用 Vision feature 替换这些位置的 embedding。序列长度在替换过程中保持
不变，后续通过 Language self-attention 完成图文融合。

## 3. 编译与部署环境

| 项目 | 配置 |
|---|---|
| 编译机 | `kangjie.xu@10.112.20.45` |
| GPU | NVIDIA RTX 4090 |
| 编译环境 | Conda `oellm_clean`，Python 3.10 |
| SDK | `D-Robotics_LLM_S600_1.0.5_SDK` |
| HBDK/HBRT 编译侧 | `libhbrt4.so 4.10.2a2` |
| 目标板 | `sunrise@10.112.133.20` |
| 板端运行时 | HBRT 4.9.6 |
| 目标架构 | `nash-p` |
| 推荐资源 | CPU 16 核以上、RAM 128 GB 以上、NVMe 可用空间 100 GB 以上 |

本文使用的主要目录：

```text
# 4090
/home/kangjie.xu/oe_locateanything/Qwen-2.5-VL-3B/

# S600
/opt/oellm_runtime/
```

# 一、模型架构

## 1. Vision Module

Qwen2.5-VL-3B 的视觉编码器使用 448x448 固定输入，主要参数如下。

| 项目 | 数值 |
|---|---|
| 输入分辨率 | 448x448 |
| Patch size | 14x14 |
| Temporal patch size | 2，静态部署时折叠为 1 |
| Patch 数量 | 32x32 = 1024 |
| Patch 输入维度 | 3x14x14 = 588 |
| Vision hidden size | 1280 |
| Transformer blocks | 32 |
| Attention | 28 个 Window Attention，4 个 Full Attention |
| Full Attention 层 | 7、15、23、31 |
| Spatial merge | 2x2 |
| 输出 token 数 | 1024 / 4 = 256 |
| Language hidden size | 2048 |

Vision HBM 内部流程：

```text
[1, 1024, 588]
  -> Patch Embedding: 588 -> 1280
  -> 32 x Vision Transformer Block
  -> 2x2 Spatial Merge: 4 x 1280 -> 5120
  -> MLP Projector: 5120 -> 5120 -> 2048
  -> [1, 256, 2048]
```

每个 Vision Transformer Block 使用视觉 self-attention，不包含独立的跨模态
cross-attention。Window Attention 降低大部分层的计算量，Full Attention 每隔八层
建立全局 patch 交互。

## 2. Language Module

| 项目 | 数值 |
|---|---|
| 架构 | Decoder-only Transformer |
| Hidden size | 2048 |
| Layers | 36 |
| Attention heads | 16 |
| KV heads | 2 |
| Head dimension | 128 |
| MLP intermediate size | 11008 |
| Vocabulary | 151936 |
| Position encoding | Qwen2.5-VL M-RoPE |
| Prefill chunk | 256 |
| KV cache length | 1024 |
| Decode query length | 1 |

Language HBM 包含 `prefill` 和 `decode` 两张图。Prefill 处理完整输入序列并建立
KV cache，Decode 每轮输入一个 token embedding，复用 KV cache 生成后续 token。

# 二、准备工作

## 1. 获取项目和模型

在 4090 编译机执行：

```bash
cd ~
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd ~/oe_locateanything

source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean
pip install -U modelscope

mkdir -p Qwen-2.5-VL-3B/checkpoint
modelscope download \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir Qwen-2.5-VL-3B/checkpoint/Qwen2.5-VL-3B-Instruct
```

模型目录至少应包含：

```text
config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
```

## 2. 下载 S600 SDK 与用户手册

在 4090 编译机下载 D-Robotics LLM S600 1.0.5 开发工具包和配套文档：

```bash
cd /opt/oellm-sdk
mkdir -p s600_sdk s600_doc

# D-Robotics_LLM_S600 开发工具包
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz

# D-Robotics_LLM_S600 用户手册
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_Doc.zip

tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz -C s600_sdk
unzip -q D-Robotics_LLM_S600_1.0.5_Doc.zip -d s600_doc

rm D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
rm D-Robotics_LLM_S600_1.0.5_Doc.zip
```

解压后检查目录：

```bash
ls -ld \
  /opt/D-Robotics_LLM_S600_1.0.5_SDK \
  /opt/D-Robotics_LLM_S600_1.0.5_Doc
```

预期目录：

```text
oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
oellm/s600_doc/D-Robotics_LLM_S600_1.0.5_Doc
```

## 3. 安装 S600 编译环境

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n oellm_clean python=3.10 -y
conda activate oellm_clean

cd /opt/D-Robotics_LLM_S600_1.0.5_SDK
pip install -r oellm_build/requirements.txt
pip install oellm_build/hbdk4_compiler-*.whl
pip install oellm_build/leap_llm-*.whl

cd /absolute/path/to/oellm-toolchain
pip install -e . --no-deps
```

验证环境：

```bash
python -c "import torch; import leap_llm; from hbdk4 import compiler; print(torch.cuda.is_available())"
```

输出应显示 CUDA 可用，且 `leap_llm`、`hbdk4.compiler` 能正常导入。

## 4. 准备校准数据

本次使用 SDK 内置 `mmstar` 图文数据，共 120 条样本：

```text
LocateAnything/compiler/leap_llm/apis/calibration/calibration_data/mmstar/conversation.json
```

校准时统一使用 448x448 图像预处理，并分别执行 Vision forward 和完整多模态
Language prefill。Language 校准输入包含实际的 image embedding、M-RoPE position
IDs、causal mask 和初始 KV cache，不能只用随机 embedding 代替。

# 三、编译适配

## 1. 图像预处理接口

板端 `libxlm` 按以下顺序生成 Vision 输入：

```text
JPEG/PNG 解码为 RGB
  -> resize 到 448x448
  -> pixel / 255
  -> 按通道执行 mean/std 归一化
  -> 14x14 patchify
  -> flatten
  -> FP16
```

归一化参数：

```text
mean = [0.48145466, 0.45782750, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

448/14=32，因此一张图得到 1024 个 patch；每个 patch 展平后包含
`3x14x14=588` 个数，最终输入为 `[1,1024,588] FP16`。

## 2. 静态图像 Temporal 权重折叠

checkpoint 中 Patch Embedding 是 Conv3D，权重形状为：

```text
[1280, 3, 2, 14, 14]
```

Qwen2.5-VL 对静态图片构造两个相同的 temporal slice。设两个 slice 都为 `x`，两个
temporal kernel 为 `W0`、`W1`，原始 Conv3D 结果为：

```text
y = x * W0 + x * W1
  = x * (W0 + W1)
```

因此部署图可以只保留一个图像 slice，并将 Conv3D 权重折叠为 Conv2D：

```python
weight_2d = weight_5d.sum(dim=2)
```

对应接口为：

```text
输入:  [1, 1024, 588]
权重:  [1280, 3, 14, 14]
输出:  [1, 1024, 1280]
```

权重折叠在 checkpoint 加载阶段完成，并保证 calibration 与 BC export 使用同一份
`proj_2d.weight`。适配位置：

```text
LocateAnything/compiler/leap_llm/models/qwen2_5_vl/model.py
LocateAnything/compiler/leap_llm/nn/modules/vision_embedding.py
```

## 3. Window Token 顺序

Vision 图内部按 `window_index` 重排 patch，以执行 Window Attention。Vision HBM 保留
该输出顺序，Host 在注入 Language 前执行：

```python
reverse_indices = torch.argsort(window_index)
image_embeds = image_embeds[:, reverse_indices, :]
```

随后使用 `<|image_pad|>` mask 替换对应 embedding：

```python
image_mask = input_ids == image_token_id
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

这一步保证 256 个视觉 token 与文本模板中的 256 个占位位置一一对应。

## 4. 统一隐藏域

通过 embedding 范数、token 间 Gram 矩阵和 Orthogonal Procrustes 检查，可确认
checkpoint 隐藏向量与 S600 运行接口之间由一个 2048 维正交矩阵 `Q` 关联。矩阵可
表示为归一化 signed Walsh-Hadamard 变换，并满足：

```text
Q.T @ Q = I
```

该变换全部离线折叠到已有权重，不向 HBM 增加额外的 2048x2048 MatMul。

Vision 侧将变换折叠到 merger 最后一层：

```text
W_new = Q.T @ W
b_new = b @ Q
```

Language 侧将整个 residual stream 置于同一隐藏域：

```text
Embedding:          E_new     = E @ Q
Q/K/V input:        W_in_new  = (W_in * gamma) @ Q
Attention output:   W_o_new   = Q.T @ W_o
MLP gate/up input:  W_m_new   = (W_m * gamma) @ Q
MLP down output:    W_d_new   = Q.T @ W_d
Final lm_head:      W_lm_new  = (W_lm * gamma_final) @ Q
```

被折叠的 RMSNorm 权重设置为 1。Q/K/V、RoPE、KV cache 和 logits 的语义保持不变，
Vision、embedding 和 Language residual 使用统一坐标域。

在 PyTorch 层验证同一输入的变换前后结果：

```text
logits cosine       = 0.999999999889
KV key cosine min   > 0.9999999998
KV value cosine min > 0.9999999994
argmax_equal        = True
```

# 四、量化与编译

## 1. 生成隐藏域矩阵

项目中的公共工具可生成经过 checksum 校验的 2048 维矩阵：

```bash
cd ~/oe_locateanything
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

PYTHONPATH=$(dirname "$OELLM_LEAP_ROOT") python - <<'PY'
import torch
from leap_llm.models.locateanything.hidden_rotation import build_signed_hadamard_rotation

rotation = build_signed_hadamard_rotation(2048)
torch.save(rotation, "/tmp/qwen2_5_vl_hidden_rotation.pt")
print(rotation.shape)
print((rotation.T @ rotation - torch.eye(2048)).abs().max().item())
PY
```

期望输出 shape 为 `[2048,2048]`，正交误差小于 `1e-5`。

## 2. Vision 编译

Vision 使用 W8 量化。完整流程为：

```text
加载 checkpoint
  -> temporal Conv3D 权重折叠
  -> Vision 输出隐藏域折叠
  -> 120 条图像校准
  -> export visual BC
  -> convert MLIR
  -> 移除边界 Quantize/Dequantize
  -> compile visual HBO
  -> link Vision HBM
```

后台编译命令：

```bash
cd ~/oe_locateanything
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

export OELLM_LEAP_ROOT=/absolute/path/to/leap_llm
export LEAP_ROOT=$OELLM_LEAP_ROOT
export MODEL_PATH=$PWD/Qwen-2.5-VL-3B/checkpoint/Qwen2.5-VL-3B-Instruct
export CALIBRATION_PATH=$LEAP_ROOT/apis/calibration/calibration_data/mmstar/conversation.json
export ROTATION_PATH=/tmp/qwen2_5_vl_hidden_rotation.pt
export OUTPUT_DIR=$PWD/Qwen-2.5-VL-3B/workspace/build/vision

mkdir -p "$OUTPUT_DIR"
nohup python -u Qwen-2.5-VL-3B/compiler/compile_vision.py \
  > "$OUTPUT_DIR/compile.log" 2>&1 < /dev/null &
echo $! > "$OUTPUT_DIR/compile.pid"
```

查看日志和进程：

```bash
tail -f ~/oellm_clean/output/qwen2_5_vl_vision/compile.log
ps -fp "$(cat ~/oellm_clean/output/qwen2_5_vl_vision/compile.pid)"
```

## 3. Language 编译

Language 使用 W4 量化，并导出 FP16 embedding。完整流程为：

```text
加载 checkpoint
  -> Language 全残差隐藏域折叠
  -> 导出 embed_tokens.bin
  -> 多模态 prefill 校准
  -> export prefill/decode BC
  -> convert MLIR
  -> 移除边界 Quantize/Dequantize
  -> compile prefill/decode HBO
  -> link Language HBM
```

Vision 编译完成后再启动 Language，避免两个 HBDK 作业争用 CPU 和内存：

```bash
cd ~/oe_locateanything
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

export OELLM_LEAP_ROOT=/absolute/path/to/leap_llm
export LEAP_ROOT=$OELLM_LEAP_ROOT
export MODEL_PATH=$PWD/Qwen-2.5-VL-3B/checkpoint/Qwen2.5-VL-3B-Instruct
export PROCESSOR_MODEL_PATH=$MODEL_PATH
export CALIBRATION_PATH=$LEAP_ROOT/apis/calibration/calibration_data/mmstar/conversation.json
export ROTATION_PATH=/tmp/qwen2_5_vl_hidden_rotation.pt
export QWEN_HELPER_ROOT=$PWD/Qwen-2.5-VL-3B/compiler
export OUTPUT_DIR=$PWD/Qwen-2.5-VL-3B/workspace/build/language

mkdir -p "$OUTPUT_DIR"
nohup python -u Qwen-2.5-VL-3B/compiler/compile_language.py \
  > "$OUTPUT_DIR/compile.log" 2>&1 < /dev/null &
echo $! > "$OUTPUT_DIR/compile.pid"
```

查看日志：

```bash
tail -f ~/oellm_clean/output/qwen2_5_vl_language/compile.log
```

## 4. 编译参数

Vision 和 Language 的 HBO 编译使用以下参数：

```text
march=nash-p
opt=2
jobs=16
core_num=4
input_no_padding=True
output_no_padding=True
enable_hpc=True
max_l2m_size=25165824
```

`jobs=16` 是编译并行度，不是 BPU 核数；实际图使用 4 个 BPU core。HBO 编译主要
消耗 CPU 和内存，GPU 主要用于模型加载、校准和 PyTorch forward。

## 5. 中间产物

Vision：

```text
vision.visual.bc
vision.visual_convert.bc
vision.visual.hbo
Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm
```

Language：

```text
language.prefill.bc
language.prefill_convert.bc
language.prefill.hbo
language.decode.bc
language.decode_convert.bc
language.decode.hbo
Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm
Qwen2.5-VL-3B-Instruct_embed_tokens.bin
```

本次验证产物：

| 产物 | 大小 | SHA256 |
|---|---:|---|
| Vision HBM | 762,029,104 bytes | `d4511b8f910c25d8111056ce4cddf7652c91b59e05c3d95fdabc4dead0e94df8` |
| Language HBM | 1,825,571,064 bytes | `05961201af02c22894f48a4c5d90f859878c473892f8c9e3c0012bbe7f7aabd0` |
| Embedding | 622,329,856 bytes | `f9efbe1d4905a581993e255de0e815f304c24a0bf00f5bee8f89f5e47e464caf` |

不同工具链构建时间或编译缓存可能影响 HBM 二进制内容。部署时应以当前编译机和板端
文件的 SHA256 一致为基本要求，不应只核对文件名。

# 五、部署到 S600

## 1. 已验证产物

经过语义验证的 S600 组合固定为：

```text
Vision:   model/Qwen2.5-VL-3B-Instruct/fix009_official_domain/
Language: model/Qwen2.5-VL-3B-Instruct/fix010_language_official_domain/
Config:   Qwen-2.5-VL-3B/deploy/qwen2_5_vl_3b_s600.json
```

Language HBM 和 embedding 的 SHA256 分别为：

```text
05961201af02c22894f48a4c5d90f859878c473892f8c9e3c0012bbe7f7aabd0
f9efbe1d4905a581993e255de0e815f304c24a0bf00f5bee8f89f5e47e464caf
```

旧的 `qwen2_5_vl_3b_self_compiled.json` 已归档；它指向
`self_compiled` 目录，不是最终通过语义验证的组合。

## 2. 配置运行时

将仓库中的运行配置放入板端 `vlm_demo`：

```text
Qwen-2.5-VL-3B/deploy/qwen2_5_vl_3b_s600.json
```

配置中的关键接口为：

```json
{
  "model_type": "Qwen2.5-VL",
  "model_dir": "../../model/Qwen2.5-VL-3B-Instruct/fix010_language_official_domain/",
  "vit_model_file": "../fix009_official_domain/Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm",
  "llm_model_file": "Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm.fresh",
  "embed_weight_file_path": "Qwen2.5-VL-3B-Instruct_embed_tokens.bin",
  "temporal_patch_size": 1,
  "vocab_size": 151936,
  "embed_dim": 2048,
  "image_height": 448,
  "image_width": 448
}
```

在 S600 复制配置：

```bash
cp ~/oe_locateanything/Qwen-2.5-VL-3B/deploy/qwen2_5_vl_3b_s600.json \
  /opt/oellm_runtime/examples/vlm_demo/
```

## 3. 启动推理

在 S600 执行：

```bash
cd /opt/oellm_runtime/examples/vlm_demo
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
```

纯文本：

```bash
./vlm -c qwen2_5_vl_3b_s600.json
```

带图片：

```bash
./vlm \
  -c qwen2_5_vl_3b_s600.json \
  -i /home/sunrise/oe_locateanything/oellm_runtime/examples/vlm_demo/image1.jpg
```

交互模式中重新加载图片时，路径不要加引号：

```text
/image /home/sunrise/oe_locateanything/oellm_runtime/examples/vlm_demo/image1.jpg
```

# 六、验证结果

## 1. 单模块数值验证

在相同运行时真实输入下，Vision 输出与 S600 参考链路的 cosine 为：

```text
cosine = 0.9879630
```

该结果用于确认 Vision 图的 patch embedding、window token 顺序、量化和 2048 维
输出接口处于可用状态。

## 2. 纯文本验证

输入：

```text
hi？
```

输出：

```text
Hello! How can I assist you today?
```

单次运行记录：

```text
prefill token num: 256
prefill cost: 46.254 ms
prefill speed: 5534.656 tokens/s
decode cost per token: 14.257 ms
decode speed: 70.143 tokens/s
```

## 3. 图文验证

对 `image1.jpg` 输入“描述一下图片”，模型识别出木质平台上的小熊猫，并描述了
红褐色毛发和白色面部标记。

单次运行记录：

```text
vit cost: 42.733 ms
prefill token num: 512
prefill cost: 86.972 ms
prefill speed: 5886.952 tokens/s
decode cost per token: 14.209 ms
decode speed: 70.380 tokens/s
```

以上数据是一次功能验证结果，不作为完整性能基准。正式性能测试需要固定图片、prompt、
采样参数、生成长度和计时口径，并进行多轮统计。

# 七、验收标准与常见问题

## 1. 验收顺序

1. checkpoint 关键权重全部加载，`missing/unexpected keys` 在允许范围内；
2. temporal 权重由 `[1280,3,2,14,14]` 折叠为 `[1280,3,14,14]`；
3. 隐藏域矩阵正交误差小于 `1e-5`；
4. Language 变换前后 logits、KV 和 argmax 保持等价；
5. BC 图名、shape 和 dtype 与运行时配置一致；
6. HBO 编译和 HBM link 正常完成；
7. 编译机与 S600 的 SHA256 完全一致；
8. S600 成功加载 `visual`、`prefill`、`decode` 三张图；
9. 纯文本和至少两张不同图片的语义输出正常。

## 2. 常见问题

### `libxlm.so` 找不到

```bash
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
```

必须在 `oellm_runtime/examples/vlm_demo` 目录执行，或改为绝对库路径。

### Vision HBM 文件不存在

检查 JSON 中 `model_dir` 与 `vit_model_file` 拼接后的完整路径。文件名中的
`w8`、`nash-p` 和 `corenum_4` 必须与实际产物一致。

### 图片读取失败

交互命令 `/image` 后直接填写路径，不要保留 shell 引号。先使用 `ls -l` 和
`file image.jpg` 检查文件存在且格式可读。

### 输出乱码

依次检查：

```text
Vision、Language、embedding 是否来自同一次隐藏域配置
embed 文件大小是否为 622329856 bytes
vocab_size 是否为 151936
Vision token 是否按 argsort(window_index) 恢复顺序
temporal_patch_size 是否为 1
三份产物 SHA256 是否在传输前后一致
```

### 日志长时间停留在 compile HBO

HBO 编译阶段主要在 CPU 上执行，进度条可能较长时间不刷新。使用 PID、CPU 时间和
子进程状态共同判断，不应仅根据最后一行日志判断卡死。建议单独运行一个 HBDK 编译
任务，并使用 `jobs=16`。

# 八、结论

本流程完成了 Qwen2.5-VL-3B-Instruct 从 checkpoint 到 S600 的完整部署：

```text
checkpoint
  -> 静态 Vision 适配
  -> 统一隐藏域权重折叠
  -> PTQ 校准
  -> BC / MLIR / HBO / HBM
  -> embedding 导出
  -> S600 runtime 配置
  -> 纯文本与图文验证
```

最终测试使用自编译 Vision HBM、自编译 Language HBM 和自行导出的 embedding；
运行时使用 S600 SDK 1.0.5 提供的 `vlm/libxlm/HBRT`。纯文本和图文主链路均已在
S600 上验证可用。

## 参考

- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [D-Robotics Developer](https://developer.d-robotics.cc/)
- [社区开发者 Gemma4 部署实践](https://forum.d-robotics.cc/t/topic/35332)
- `$OELLM_LEAP_ROOT/models/qwen2_5_vl/model.py`
- `$OELLM_LEAP_ROOT/apis/model/qwen2_5_vl.py`
- `Qwen-2.5-VL-3B/compiler/compile_vision.py`
- `Qwen-2.5-VL-3B/compiler/compile_language.py`
