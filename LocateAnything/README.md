<div align="center">

<img src="assets/LocateAnything.jpg" alt="LocateAnything detection on D-Robotics S600" width="820">

# LocateAnything-3B on D-Robotics S600

Run LocateAnything-3B on the D-Robotics S600 BPU for open-vocabulary
detection, referring expression grounding, point grounding, OCR grounding,
and document layout grounding.

[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-D--Robotics%20S600-35a853)](https://developer.d-robotics.cc/)
[![SDK](https://img.shields.io/badge/OELLM-1.0.5-2563eb)](https://developer.d-robotics.cc/)
[![Model](https://img.shields.io/badge/model-LocateAnything--3B-f59e0b)](https://huggingface.co/nvidia/LocateAnything-3B)

**English** | [中文](README.zh-CN.md)

</div>

## Model Configuration

| Item | Configuration |
|---|---|
| Input | 672 x 672 letterbox |
| Vision | MoonViT, W8 |
| Language | Qwen2.5 decoder, W8 |
| LM Head | W8 |
| Prefill | 1024 tokens |
| KV cache | 4096 tokens |
| PBD | q=6 |
| AR | q=1 |
| BPU | 4 cores |

Two Language graph sets are available:

| Graph set | Purpose | Graphs |
|---|---|---|
| `standard` | Default deployment and accuracy validation | Prefill, PBD q6, AR q1 |
| `fused_decode` | Fused decoding | `standard` plus PBD q7-q12 and AR q2-q5 |

Their build configurations are `compiler/configs/standard.yaml` and
`compiler/configs/fused_decode.yaml`. Calibration, compilation, verification,
and runtime must use the same graph set.

## Deploy Prebuilt HBM

This path runs entirely on the S600. It downloads the source and prebuilt
Vision HBM, Language HBM, and embedding table without running CUDA calibration
or HBDK compilation.

### 1. Install the runtime

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything/LocateAnything

python3 -m venv --system-site-packages .venv-s600
source .venv-s600/bin/activate
python -m pip install -e ".[runtime]"
python -m pip install -U huggingface_hub
```

### 2. Download HBM assets

```bash
export LA_HF_REPO="YOUR_ACCOUNT/LocateAnything-3B-S600"

python scripts/hf_assets.py download \
  --repo-id "$LA_HF_REPO" \
  --kind hbm \
  --local-dir artifacts/huggingface
```

### 3. Build and install the S600 runtime

```bash
cmake -S deploy -B deploy/build -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/build \
  --target vision_hbm_runner language_hbm_runner \
  -j4

sh deploy/install_locateanything_cli.sh
export PATH="$HOME/.local/bin:$PATH"
```

### 4. Run inference

```bash
export LA_RELEASE_ROOT="$PWD/artifacts/huggingface/hbm"
export LA_TOKENIZER_DIR="$PWD/deploy/tokenizer"

LocateAnything \
  -i /path/to/image.jpg \
  -p '/detect person,motorcycle' \
  --output-dir artifacts/runs/predict/demo
```

The CLI keeps both HBM models resident, reports the Vision, Language, and
Postprocess stage timings, and automatically saves an annotated image when a
detection or point result is parsed.

## Prediction Outputs

Each request has one self-contained output directory:

```text
artifacts/runs/predict/demo/
└── request_0001/
    ├── prediction.json
    ├── annotated.png
    ├── timings.json
    └── logs/
        └── runtime.log
```

Each request in a CLI session gets `request_0001`, `request_0002`, and so on.
Without `--output-dir`, the CLI creates a timestamped run directory under
`artifacts/runs/predict/`. Results are never written next to the input image.

## Build from Source

This path requires an x86_64 CUDA host, the D-Robotics S600 OELLM 1.0.5 SDK,
the original model, and the 1,200-sample calibration set.

### 1. Download the SDK

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
```

Create the compiler environment using the SDK instructions. The commands below
assume that environment is named `oellm_clean`.

### 2. Install compiler dependencies

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything/LocateAnything

git clone https://github.com/NVlabs/Eagle.git artifacts/upstream/Eagle

source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean
python -m pip install -r compiler/requirements-host.txt
python -m pip install -e compiler --no-deps
```

### 3. Download the model and calibration data

```bash
hf download nvidia/LocateAnything-3B \
  --local-dir artifacts/models/LocateAnything-3B

export LA_HF_REPO="YOUR_ACCOUNT/LocateAnything-3B-S600"
python scripts/hf_assets.py download \
  --repo-id "$LA_HF_REPO" \
  --kind calibration \
  --local-dir artifacts/huggingface \
  --verify-images

export LA_CALIBRATION_ROOT="$PWD/artifacts/huggingface/calibration"
export LA_UPSTREAM_SOURCE="$PWD/artifacts/upstream/Eagle/Embodied"
```

### 4. Calibrate, compile, and verify `standard`

```bash
CONFIG=compiler/configs/standard.yaml

python compiler/quantize.py --config "$CONFIG" prepare --preflight-only
python compiler/quantize.py --config "$CONFIG" prepare --resume
python compiler/quantize.py --config "$CONFIG" calibrate --component all --resume
python compiler/quantize.py --config "$CONFIG" build --component all --target bc --resume
python compiler/quantize.py --config "$CONFIG" build --component all --target hbm --resume
python compiler/quantize.py --config "$CONFIG" verify --component all --stage specification
```

Builds, logs, and evaluations are stored separately:

```text
artifacts/calibration/current/statistics/standard/
artifacts/builds/locateanything-3b/standard/
artifacts/logs/locateanything-3b/standard/
artifacts/evaluation/locateanything-3b/standard/
```

Use `compiler/configs/fused_decode.yaml` for fused decoding. It writes to an
independent output tree.

Long-running commands report the current stage, completed work, elapsed time,
and an ETA when one can be computed. `--resume` only reuses complete outputs
whose configuration and input hashes still match.

The `pipeline` and `task` verification stages require Float/BC/HBM intermediate
results, ground-truth records, and S600 predictions generated from the same
inputs. Run `verify --stage pipeline` and `verify --stage task` only after those
inputs exist; the primary build flow does not substitute empty or unrelated
results.

### 5. Deploy a compiled build

```bash
BUILD_DIR="$PWD/artifacts/builds/locateanything-3b/standard"

bash deploy/deploy_locateanything_s600.sh \
  --release locateanything-s600-standard-v1 \
  --vision-hbm "$BUILD_DIR/vision/LocateAnything-3B_vision_672x672_w8_nash-p_corenum_4.hbm" \
  --language-hbm "$BUILD_DIR/language/LocateAnything-3B_language_chunk_1024_cache_4096_decoder_w8_lmhead_w8_nash-p_corenum_4_4.hbm" \
  --embed-bin "$BUILD_DIR/language/LocateAnything-3B_embed_tokens.bin" \
  --runtime-config deploy/runtime_config.json \
  --tokenizer-dir deploy/tokenizer \
  --ssh-target sunrise@S600_IP \
  --execute
```

Run the deployed release on the S600:

```bash
cd /home/sunrise/oe_locateanything/LocateAnything/artifacts/releases/locateanything-s600-standard-v1
sh deploy/install_locateanything_cli.sh
export PATH="$HOME/.local/bin:$PATH"

LocateAnything \
  -c config/locateanything_3b_config.json \
  -i /path/to/image.jpg \
  -p '/detect person,motorcycle' \
  --output-dir artifacts/runs/predict/demo
```

The deployment tool stages a new immutable directory, checks byte counts and
SHA256 values on the S600, builds the ARM64 runtime, and only then publishes
the final release directory. Existing release directories are not overwritten.

## Upload Calibration and HBM Assets

The Hugging Face upload contains only calibration data and deployable HBM
assets:

```bash
python scripts/hf_assets.py checksums --root hf_assets/calibration
python scripts/hf_assets.py checksums --root hf_assets/hbm
python scripts/hf_assets.py validate --kind all --local-dir hf_assets --verify-images

hf auth login
hf upload "$LA_HF_REPO" hf_assets . --repo-type model
```

## Repository Layout

```text
LocateAnything/
├── compiler/              calibration and BC/HBO/HBM compilation
│   ├── configs/           `standard` and `fused_decode` build configurations
│   ├── leap_llm/          OELLM model adapter
│   ├── scripts/           build and verification implementation
│   └── quantize.py        unified compiler entry point
├── deploy/                S600 C++ runners, CLI, and deployment tool
├── scripts/               Hugging Face asset utility
├── src/                   shared Python modules
├── tests/                 regression tests
└── artifacts/             local models, calibration, builds, runs, and releases
```

`artifacts/` is excluded from Git. Source directories do not contain model
weights, calibration tensors, compiler outputs, runtime logs, or report images.

## License

This project uses [CC BY-NC 4.0](LICENSE). Model weights, the D-Robotics SDK,
and upstream components retain their respective licenses.
