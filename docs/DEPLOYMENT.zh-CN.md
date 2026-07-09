# 部署指南 · LocateAnything-3B on D-Robotics S600

从零开始的分步手册，从裸的 NVIDIA GPU 主机到编译出可在 S600 上运行的 `LocateAnything-3B_*.hbm`。

主机分工：

- **NVIDIA GPU 主机 (x86)** — PyTorch baseline、OELLM S600 工具链、HBM 编译。
- **S600 端侧 (aarch64)** — 端侧执行 (runtime 层与 M5 一起交付)。

以下所有步骤在 NVIDIA GPU 主机上执行，除非明确标注 S600。

---

## 1. 基础环境 (NVIDIA GPU 主机)

### 1.1 克隆项目

```bash
cd ~
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
git clone https://github.com/NVlabs/Eagle.git eagle
```

### 1.2 创建 LocateAnything Conda 环境

```bash
cd ~/oe_locateanything/eagle/Embodied

conda create -n locateanything python=3.10 -y
conda activate locateanything

python -m pip install -U pip huggingface_hub hf_transfer
```

### 1.3 下载 LocateAnything-3B 权重

模型页面：<https://huggingface.co/nvidia/LocateAnything-3B>

```bash
cd ~/oe_locateanything/eagle/Embodied
rm -rf LocateAnything-3B

export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_ENABLE_HF_TRANSFER

hf download nvidia/LocateAnything-3B --local-dir LocateAnything-3B
```

权重路径：`~/oe_locateanything/eagle/Embodied/LocateAnything-3B`。

### 1.4 安装 LocateAnything 并运行 baseline

```bash
cd ~/oe_locateanything/eagle/Embodied
pip install -e .

PYTHONPATH=$PWD python ~/oe_locateanything/main/examples/demo_min.py
```

期望输出：

```text
answer: <ref>cat</ref><box><...><...><...><...></box>
boxes: [{'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}]
```

两个 golden 文件写入 `eagle/Embodied/deploy_s600/golden/` (`official_answer.txt`, `official_boxes.txt`)，作为后续 HBM ↔ PyTorch 对齐的参考。

---

## 2. OELLM S600 编译环境 (NVIDIA GPU 主机)

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

解压后布局：

```text
oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
oellm/s600_doc/D-Robotics_LLM_S600_1.0.5_Doc
```

### 2.2 创建 OELLM Conda 环境

```bash
conda create -n oellm python=3.10 -y
conda activate oellm

cd ~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
pip install -r oellm_build/requirements.txt
pip install oellm_build/hbdk4_compiler-*.whl
pip install oellm_build/hbdk4_runtime_aarch64_unknown_linux_gnu_nash-*.whl
```

### 2.3 以 editable 模式安装 vendored 的 `leap_llm`

本仓库将 `leap_llm` wheel 源码 vendored 到 `toolchain/`，这样每次改动都通过 `git diff` 跟踪。以 editable 模式安装：

```bash
cd ~/oe_locateanything/toolchain
pip install -e . --no-deps
```

验证：

```bash
python -c "import importlib.metadata as m; print('leap_llm', m.version('leap-llm'))"
python -c "from hbdk4.compiler import leap; print('hbdk4 leap OK:', leap)"
oellm_build --help | grep -E 'locateanything|qwen2_5-vl-3b'
```

期望：`leap_llm 1.0.5`，以及至少以下 builders：

```text
- qwen2_5-vl-3b: nash-p
- locateanything-3b: nash-p
- locateanything-lm-3b: nash-p
- locateanything-vit-3b: nash-p
```

### 2.4 交叉编译工具链 (供 S600 端侧 runtime 使用)

```bash
cd ~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
tar -xf arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
export LINARO_GCC_ROOT=$PWD/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu
```

---

## 3. 编译 LocateAnything HBM

### 3.1 Language HBM (Qwen2.5-3B + PBD)

```bash
conda activate oellm

oellm_build \
  --model_name locateanything-lm-3b \
  --march nash-p \
  --input_model_path ~/oe_locateanything/eagle/Embodied/LocateAnything-3B \
  --output_model_path ~/oe_locateanything/main/language/baseline_outputs/locateanything-lm-3b_nash-p_w4 \
  --w_bits 4 \
  --chunk_size 256 \
  --cache_len 1024 \
  --decode_seq_len 6 \
  --image_width 448 --image_height 448 \
  --device cuda:0 \
  --prefill_core_num 4 --decode_core_num 4 \
  --jobs 16
```

耗时：16 核 Xeon 级主机 + 单张 RTX 4090 校准，wall-clock ≈ 3–4 小时。

产出位于 `main/language/baseline_outputs/locateanything-lm-3b_nash-p_w4/`：

- `LocateAnything-3B_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm`
- `LocateAnything-3B_embed_tokens.bin` (fp16 embedding 表, ≈ 625 MB)

### 3.2 Vision HBM (MoonViT + projector, M3-β)

```bash
oellm_build \
  --model_name locateanything-vit-3b \
  --march nash-p \
  --input_model_path ~/oe_locateanything/eagle/Embodied/LocateAnything-3B \
  --output_model_path ~/oe_locateanything/main/vision/outputs/locateanything-vit-3b_nash-p_w8 \
  --w_bits 8 \
  --image_width 448 --image_height 448 \
  --device cuda:0 \
  --vit_core_num 4 \
  --jobs 16
```

`LocateAnythingVisionApi` 与 `locateanything-vit-3b` builder 已经注册；端到端 HBM 落盘随 M3-β 交付。

### 3.3 统一 builder (M4)

两个 stage 稳定后，`locateanything-3b` 会一次编译同时产出 vision + language。

---

## 4. Runtime 与精度对齐

Runtime 集成 (Host 侧视觉 embed 合并 + PBD 采样 + `<box>` 解析) 为 M5，精度对齐工具为 M6。

已入仓的骨架脚本：

- `main/examples/verify_lm_hbm.py` — HBM ↔ PyTorch logits 对齐。

---

## 编译主机推荐配置

| 组件 | 推荐配置 | 说明 |
|---|---|---|
| CPU 核数 | ≥ 16 | 直接决定 `compile_hbo` 耗时 |
| RAM | ≥ 64 GB | 3B 模型 + `--jobs 16` 峰值 ≈ 30 GB |
| 磁盘 (SSD) | ≥ 200 GB 可用 | 单 stage `.bc` / `.hbo` 中间产物 ≈ 10 GB |
| GPU | 1× RTX 4090 或 ≥ 24 GB VRAM | 只在校准阶段 (≈ 30 分钟) 使用 |
