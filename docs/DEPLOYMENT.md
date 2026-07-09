# Deployment Guide · LocateAnything-3B on D-Robotics S600

Step-by-step manual for going from a fresh NVIDIA GPU host to a compiled `LocateAnything-3B_*.hbm` ready to run on the S600.

Environment split:

- **NVIDIA GPU host (x86)** — PyTorch baseline, OELLM S600 toolchain, HBM compilation.
- **S600 device (aarch64)** — on-device execution (runtime layer lands with M5).

Every step below runs on the NVIDIA GPU host unless labelled S600.

---

## 1. Base Environment (NVIDIA GPU host)

### 1.1 Clone the project

```bash
cd ~
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
git clone https://github.com/NVlabs/Eagle.git eagle
```

### 1.2 Create the LocateAnything Conda environment

```bash
cd ~/oe_locateanything/eagle/Embodied

conda create -n locateanything python=3.10 -y
conda activate locateanything

python -m pip install -U pip huggingface_hub hf_transfer
```

### 1.3 Download LocateAnything-3B weights

Model page: <https://huggingface.co/nvidia/LocateAnything-3B>

```bash
cd ~/oe_locateanything/eagle/Embodied
rm -rf LocateAnything-3B

export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_ENABLE_HF_TRANSFER

hf download nvidia/LocateAnything-3B --local-dir LocateAnything-3B
```

Weights path: `~/oe_locateanything/eagle/Embodied/LocateAnything-3B`.

### 1.4 Install LocateAnything and run the baseline

```bash
cd ~/oe_locateanything/eagle/Embodied
pip install -e .

PYTHONPATH=$PWD python ~/oe_locateanything/main/examples/demo_min.py
```

Expected output:

```text
answer: <ref>cat</ref><box><...><...><...><...></box>
boxes: [{'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}]
```

Two golden files are written to `eagle/Embodied/deploy_s600/golden/` (`official_answer.txt`, `official_boxes.txt`) as reference for later HBM ↔ PyTorch alignment.

---

## 2. OELLM S600 Compile Environment (NVIDIA GPU host)

### 2.1 Download and extract the SDK and docs

```bash
cd ~/oe_locateanything/oellm

mkdir -p s600_sdk s600_doc

wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_Doc.zip

tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz -C s600_sdk
unzip -q D-Robotics_LLM_S600_1.0.5_Doc.zip -d s600_doc

rm D-Robotics_LLM_S600_1.0.5_SDK.tar.gz D-Robotics_LLM_S600_1.0.5_Doc.zip
```

Resulting layout:

```text
oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
oellm/s600_doc/D-Robotics_LLM_S600_1.0.5_Doc
```

### 2.2 Create the OELLM Conda environment

```bash
conda create -n oellm python=3.10 -y
conda activate oellm

cd ~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
pip install -r oellm_build/requirements.txt
pip install oellm_build/hbdk4_compiler-*.whl
pip install oellm_build/hbdk4_runtime_aarch64_unknown_linux_gnu_nash-*.whl
```

### 2.3 Install the vendored `leap_llm` in editable mode

This repository vendors the `leap_llm` wheel source at `toolchain/` so downstream changes track as `git diff`. Install as editable:

```bash
cd ~/oe_locateanything/toolchain
pip install -e . --no-deps
```

Verify:

```bash
python -c "import importlib.metadata as m; print('leap_llm', m.version('leap-llm'))"
python -c "from hbdk4.compiler import leap; print('hbdk4 leap OK:', leap)"
oellm_build --help | grep -E 'locateanything|qwen2_5-vl-3b'
```

Expected: `leap_llm 1.0.5` and at least the following builders:

```text
- qwen2_5-vl-3b: nash-p
- locateanything-3b: nash-p
- locateanything-lm-3b: nash-p
- locateanything-vit-3b: nash-p
```

### 2.4 Cross-compile toolchain (for S600-side runtime)

```bash
cd ~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
tar -xf arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
export LINARO_GCC_ROOT=$PWD/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu
```

---

## 3. Compile the LocateAnything HBMs

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

Wall-clock: ≈ 3–4 hours on a 16-core Xeon-class host with a single RTX 4090 for calibration.

Outputs under `main/language/baseline_outputs/locateanything-lm-3b_nash-p_w4/`:

- `LocateAnything-3B_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm`
- `LocateAnything-3B_embed_tokens.bin` (fp16 embedding table, ≈ 625 MB)

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

The `LocateAnythingVisionApi` and `locateanything-vit-3b` builder are registered; end-to-end HBM production lands with M3-β.

### 3.3 Unified builder (M4)

Once both stages are stable, `locateanything-3b` will run vision + language in one invocation.

---

## 4. Runtime and Precision Alignment

Runtime integration (host-side visual embed merge + PBD sampling + `<box>` parser) is M5. Precision alignment tooling is M6.

Skeleton scripts already checked in:

- `main/examples/verify_lm_hbm.py` — HBM ↔ PyTorch logits comparison.

---

## Recommended Compile-Host Hardware

| Component | Recommended | Notes |
|---|---|---|
| CPU cores | ≥ 16 | Directly gates `compile_hbo` wall-clock |
| RAM | ≥ 64 GB | Peak ≈ 30 GB with 3B model + `--jobs 16` |
| Disk (SSD) | ≥ 200 GB free | Intermediate `.bc` / `.hbo` per stage ≈ 10 GB |
| GPU | 1× RTX 4090 or ≥ 24 GB VRAM | Only used during calibration (≈ 30 min) |
