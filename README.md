<div align="center">

# oe_locateanything

**LocateAnything-3B deployment on D-Robotics S600**

<p align="center">
  <img src="assets/LocateAnything.jpg" alt="LocateAnything" width="100%">
</p>

[![Platform](https://img.shields.io/badge/platform-D--Robotics%20S600-brightgreen)](#)
[![Runtime](https://img.shields.io/badge/runtime-OELLM%201.0.5-blue)](#)
[![Model](https://img.shields.io/badge/model-LocateAnything--3B-orange)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)

</div>

---

## 概述

本仓库提供 LocateAnything-3B 在 D-Robotics S600 平台上的端到端部署实现，包括：

- **Vision**  MoonViT + MLP Projector
- **Language**  LocateAnything Qwen2.5 Decoder
- **Generation**  Slow / Fast (PBD) / Hybrid 三种解码模式
- **Runtime**  Host 侧 visual embeddings 与 KV cache 调度
- **Validation**  PyTorch 与 HBM 输出一致性校验

前置准备、量化编译在 NVIDIA GPU 主机（x86）上完成，端侧运行在 S600 主机上。

---

## 架构

<details>
<summary>展开详细数据流</summary>

```text
image
  ↓
Vision HBM (MoonViT + MLP Projector)
  input  pixel_values [1656, 3, 14, 14]
  output visual_embeds [414, 2048]

text prompt
  ↓
Host tokenizer → input_ids

input_ids + visual_embeds
  ↓
Host: inputs_embeds / position_ids / attention_mask

Qwen Prefill HBM
  input  inputs_embeds [prefill_len, 2048], position_ids, attention_mask
  output logits, KV cache

PBD Decode HBM (q_len=6)
  input  block embeddings [6, 2048], KV cache, position_ids, attention_mask
  output logits [6, vocab], updated KV cache

AR Decode HBM (q_len=1)
  input  token embedding [1, 2048], KV cache, position_ids, attention_mask
  output logits [1, vocab], updated KV cache

Host: PBD / Hybrid sampling → fallback → box / coordinate post-processing
```

</details>

---

## 1. 基础环境准备（NVIDIA GPU 主机）

在 NVIDIA GPU 主机上生成 PyTorch 基线，供后续 S600 量化与端侧对齐使用。

### 1.1 拉取仓库

```bash
cd ~
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
git clone https://github.com/NVlabs/Eagle.git eagle
```

### 1.2 创建 Conda 环境

```bash
cd ~/oe_locateanything/eagle/Embodied

conda create -n locateanything python=3.10 -y
conda activate locateanything

python -m pip install -U pip huggingface_hub hf_transfer
```

### 1.3 下载模型权重

模型页面：<https://huggingface.co/nvidia/LocateAnything-3B>

```bash
cd ~/oe_locateanything/eagle/Embodied
rm -rf LocateAnything-3B

export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_ENABLE_HF_TRANSFER

hf download nvidia/LocateAnything-3B --local-dir LocateAnything-3B
```

权重存放路径：`~/oe_locateanything/eagle/Embodied/LocateAnything-3B`。

### 1.4 安装 LocateAnything 并跑基线

```bash
cd ~/oe_locateanything/eagle/Embodied
pip install -e .

PYTHONPATH=$PWD python ~/oe_locateanything/main/examples/demo_min.py
```

成功时输出：

```text
answer: <ref>cat</ref><box><...><...><...><...></box>
boxes: [{'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}]
```

同时在 `eagle/Embodied/deploy_s600/golden/` 下生成 `official_answer.txt` 与 `official_boxes.txt`，作为后续 HBM 输出对齐的基线。

---

## 2. OELLM S600 编译环境（NVIDIA GPU 主机）

OELLM S600 工具链与文档在同一台 NVIDIA GPU 主机上安装，用于将 LocateAnything 量化编译为 S600 上可执行的 HBM。

### 2.1 下载并解压 SDK 与文档

```bash
cd ~/oe_locateanything/oellm

mkdir -p s600_sdk s600_doc

wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_Doc.zip

tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz -C s600_sdk
unzip -q D-Robotics_LLM_S600_1.0.5_Doc.zip -d s600_doc

rm D-Robotics_LLM_S600_1.0.5_SDK.tar.gz D-Robotics_LLM_S600_1.0.5_Doc.zip
```

解压后目录：

```text
oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
oellm/s600_doc/D-Robotics_LLM_S600_1.0.5_Doc
```

### 2.2 创建 Conda 环境

```bash
conda create -n oellm python=3.10 -y
conda activate oellm

cd ~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
pip install -r oellm_build/requirements.txt
pip install oellm_build/hbdk4_compiler-*.whl
pip install oellm_build/hbdk4_runtime_aarch64_unknown_linux_gnu_nash-*.whl
pip install oellm_build/leap_llm-*.whl
```

### 2.3 交叉编译工具链

```bash
cd ~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
tar -xf arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
export LINARO_GCC_ROOT=$PWD/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu
```

### 2.4 验证

```bash
python -c "import leap_llm; print(leap_llm.__version__)"
python -c "from hbdk4.compiler import leap; print(leap)"
```

---

## 模型规格

| 模块 | 配置 |
|---|---|
| Vision Encoder | MoonViT-SO-400M（27 层，hidden=1152，patch=14） |
| Projector | 2-layer MLP，4608 → 2048 |
| Language Model | Qwen2.5-3B decoder（36 层，hidden=2048，KV heads=2） |
| Vocabulary | 152681（含 `<0>~<1000>`、`<ref>`、`<box>` 等坐标 token） |
| PBD Block | 6 tokens / block |
| Output Format | `<ref>label</ref><box>x1 y1 x2 y2</box>` |

| 参数量 | 值 | 占比 |
|---|---:|---:|
| Qwen2.5 language model | 3.400 B | 88.76% |
| MoonViT vision model | 0.417 B | 10.88% |
| MLP projector | 0.014 B | 0.36% |
| **Total** | **3.831 B** | 100% |

---

## 仓库结构

```text
oe_locateanything/
├── assets/                    静态资源
├── main/                      S600 部署工作目录
│   ├── examples/              基线与集成示例
│   ├── vision/                MoonViT + MLP Vision Module
│   ├── language/              Qwen Prefill / PBD Decode / AR Decode
│   ├── runtime/               Host 侧 runtime
│   ├── configs/               编译与运行配置
│   ├── scripts/               构建、验证、benchmark 脚本
│   ├── golden/                golden 数据
│   ├── benchmarks/            benchmark 输入与结果
│   ├── outputs/               编译产物（.gitignore）
│   └── logs/                  编译与验证日志
├── oellm/                     S600 SDK 与文档位置说明
├── eagle/                     LocateAnything / Eagle 源码（.gitignore）
└── README.md
```

---

## Roadmap

- [ ] S600 `qwen2_5_vl` / `qwen3_vl` 编译流程分析
- [ ] MoonViT + MLP 自定义 visual module 接入 leap_llm
- [ ] LocateAnything Qwen2.5 自定义 language module 接入 leap_llm
- [ ] Prefill / PBD Decode (q_len=6) / AR Decode HBM 编译
- [ ] Host runtime 集成 PBD / Hybrid 采样
- [ ] PyTorch ↔ HBM 单图与批量精度对齐
- [ ] S600 端侧性能与精度评估

---

## Citation

```bibtex
@misc{locateanything,
  title  = {LocateAnything},
  author = {NVIDIA},
  year   = {2025},
  url    = {https://huggingface.co/nvidia/LocateAnything-3B}
}
```

---

## References

- [LocateAnything / Eagle](https://github.com/NVlabs/Eagle)
- [D-Robotics OpenExplorer](https://developer.d-robotics.cc/)
- [Hugging Face: nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
