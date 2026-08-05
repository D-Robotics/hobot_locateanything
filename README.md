<div align="center">

<img src="LocateAnything/assets/LocateAnything.jpg" alt="LocateAnything 在 D-Robotics S600 上的检测效果" width="820">

# LocateAnything-3B on D-Robotics S600

在 D-Robotics S600 上运行 LocateAnything-3B，支持开放词汇检测、指代表达定位、点定位、OCR 和版面定位。

</div>

## 项目布局

```text
LocateAnything/
├── compiler/                    # 校准、量化和 BC/HBO/HBM 编译
├── deploy/                      # S600 运行时、CLI 和部署脚本
├── models/
│   └── LocateAnything-3B/       # 用户下载的模型文件
├── datasets/
│   └── calibration/             # 用户准备或下载的校准数据
├── outputs/                     # 编译和推理时自动生成
└── README.md
```

`models/`、`datasets/` 和 `outputs/` 的内容不提交到 Git。

## 模型配置

| 项目 | 配置 |
|---|---|
| 输入尺寸 | 672 x 672，保持宽高比并填充 |
| Vision | MoonViT，W8 |
| Language | Qwen2.5 decoder，W8 |
| LM Head | W8 |
| Prefill | 1024 tokens |
| KV Cache | 4096 tokens |
| PBD | q=6 |
| AR | q=1 |
| BPU | 4 cores |

Language 图集合由一个配置项选择：

- `standard`：Prefill、PBD q6、AR q1；
- `fused_decode`：在 standard 基础上增加融合解码图。

对应配置为 `compiler/configs/standard.yaml` 和 `compiler/configs/fused_decode.yaml`。

## 直接部署 HBM

### 1. 获取代码

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything/LocateAnything

sudo apt-get update
sudo apt-get install -y cmake ffmpeg
python3 -m venv --system-site-packages .venv-s600
source .venv-s600/bin/activate
python -m pip install -r deploy/requirements.txt
```

### 2. 下载 HBM

将项目提供的 Vision HBM、Language HBM 和 Embedding 下载到固定目录：

```bash
python -m pip install -U huggingface_hub
mkdir -p models/LocateAnything-3B

export HF_MODEL_REPO="<项目提供的 HBM 模型仓库>"
hf download "$HF_MODEL_REPO" \
  --local-dir models/LocateAnything-3B
```

目录中应包含：

```text
models/LocateAnything-3B/
├── LocateAnything-3B_vision.hbm
├── LocateAnything-3B_language.hbm
└── LocateAnything-3B_embed_tokens.bin
```

### 3. 编译 S600 运行时

```bash
cmake -S deploy -B deploy/build -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/build \
  --target vision_hbm_runner language_hbm_runner \
  -j4
```

### 4. 运行

```bash
sh deploy/scripts/install.sh
export PATH="$HOME/.local/bin:$PATH"

LocateAnything \
  -i /path/to/image.jpg \
  -p '/detect person,motorcycle' \
  --output-dir outputs/predict/demo
```

输出保存在：

```text
outputs/predict/demo/
└── request_0001/
    ├── prediction.json
    ├── annotated.png
    ├── timings.json
    └── logs/runtime.log
```

视频推理：

```text
/video /path/to/video.mp4
/detect person,motorcycle
```

视频结果保存在 `outputs/video/`。每帧独立推理，不跨帧复用 KV cache。

## 从零校准和编译

从零编译需要 x86_64 CUDA 主机、D-Robotics S600 OELLM SDK、LocateAnything 原始模型和校准数据。

### 1. 安装 S600 编译环境

```bash
mkdir -p ~/oellm/s600_sdk
cd ~/oellm

wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz -C s600_sdk

source ~/miniforge3/etc/profile.d/conda.sh
conda create -n oellm_clean python=3.10 -y
conda activate oellm_clean

cd ~/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
python -m pip install -r oellm_build/requirements.txt
python -m pip install oellm_build/hbdk4_compiler-*.whl
python -m pip install oellm_build/hbdk4_runtime_aarch64_unknown_linux_gnu_nash-*.whl
```

### 2. 准备源码和模型

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything/LocateAnything

mkdir -p models/LocateAnything-3B datasets/upstream
hf download nvidia/LocateAnything-3B \
  --local-dir models/LocateAnything-3B
git clone https://github.com/NVlabs/Eagle.git \
  datasets/upstream/Eagle
export LA_UPSTREAM_SOURCE="$PWD/datasets/upstream/Eagle/Embodied"
```

### 3. 准备校准数据

```bash
mkdir -p datasets/calibration/download
hf download xkj521999/OE_LA_Calibration_data source.zip \
  --repo-type dataset \
  --local-dir datasets/calibration/download

mkdir -p datasets/calibration/locateanything/source
python -m zipfile -e datasets/calibration/download/source.zip \
  datasets/calibration/locateanything/source
```

校准源数据应位于：

```text
datasets/calibration/locateanything/source/
├── selected.jsonl
└── images/
```

也可以使用自有校准数据，只需提供相同的 `selected.jsonl` 记录格式。

### 4. 执行 standard 编译

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean
python -m pip install -r compiler/requirements-host.txt
python -m pip install -e compiler --no-deps

CONFIG=compiler/configs/standard.yaml
python compiler/quantize.py --config "$CONFIG" prepare --resume
python compiler/quantize.py --config "$CONFIG" calibrate --component all --resume
python compiler/quantize.py --config "$CONFIG" build --component all --target bc --resume
python compiler/quantize.py --config "$CONFIG" build --component all --target hbm --resume
python compiler/quantize.py --config "$CONFIG" verify --component all --stage specification
```

生成内容统一在 `outputs/`：

```text
outputs/
├── calibration/locateanything/
├── builds/locateanything-3b/standard/
└── logs/locateanything-3b/standard/
```

### 5. 编译 fused_decode

```bash
CONFIG=compiler/configs/fused_decode.yaml
python compiler/quantize.py --config "$CONFIG" calibrate --component language --resume
python compiler/quantize.py --config "$CONFIG" build --component language --target bc --resume
python compiler/quantize.py --config "$CONFIG" build --component language --target hbm --resume
python compiler/quantize.py --config "$CONFIG" verify --component language --stage specification
```

### 6. 将 HBM 部署到 S600

```bash
BUILD_DIR=outputs/builds/locateanything-3b/standard
VISION_HBM=$(find "$BUILD_DIR/vision" -maxdepth 1 -name '*_vision_*.hbm' -print -quit)
LANGUAGE_HBM=$(find "$BUILD_DIR/language" -maxdepth 1 -name '*_language_*.hbm' -print -quit)
EMBED_BIN=$(find "$BUILD_DIR/language" -maxdepth 1 -name '*_embed_tokens.bin' -print -quit)

bash deploy/scripts/deploy.sh \
  --vision-hbm "$VISION_HBM" \
  --language-hbm "$LANGUAGE_HBM" \
  --embed-bin "$EMBED_BIN" \
  --ssh-target sunrise@S600_IP \
  --execute
```

脚本将模型文件写入 S600 项目的 `models/LocateAnything-3B/`，然后重新构建 `deploy/build/`。

## Hugging Face 资源

- Dataset 仓库：保存原始校准数据 `source.zip`；
- Model 仓库：保存 Vision HBM、Language HBM 和 Embedding；
- `generated/`、统计数据、编译中间文件和推理结果均由用户在本地生成。

模型文件和校准数据不随 GitHub 源码发布。

## 任务命令

```text
/detect cat,dog
/ground <phrase>
/ground_single <phrase>
/gui <element>
/gui_box <element>
/text
/ground_text <text>
/layout title,table,figure
/point <target>
```

检测任务会自动保存标注图片和 JSON 结果。
