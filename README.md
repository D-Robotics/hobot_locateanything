<div align="center">

<img src="LocateAnything/assets/LocateAnything.jpg" alt="LocateAnything 在 D-Robotics S600 上的检测效果" width="820">

# LocateAnything-3B on D-Robotics S600

在 D-Robotics S600 BPU 上运行 LocateAnything-3B，支持开放词汇检测、指代表达定位、
点定位、OCR 定位和文档布局定位。

[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-D--Robotics%20S600-35a853)](https://developer.d-robotics.cc/)
[![SDK](https://img.shields.io/badge/OELLM-1.0.5-2563eb)](https://developer.d-robotics.cc/)
[![Model](https://img.shields.io/badge/model-LocateAnything--3B-f59e0b)](https://huggingface.co/nvidia/LocateAnything-3B)

</div>

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

Language 提供两套执行图集合：

| 图集合 | 用途 | 包含的图 |
|---|---|---|
| `standard` | 默认部署与精度验证 | Prefill、PBD q6、AR q1 |
| `fused_decode` | 融合解码 | 在 `standard` 基础上增加 PBD q7-q12 和 AR q2-q5 |

配置文件分别为 `compiler/configs/standard.yaml` 和
`compiler/configs/fused_decode.yaml`。校准、编译、验证和运行时必须选择同一套图集合。

## 方式一：直接部署已编译 HBM

该方式在 S600 上下载运行代码和已经编译好的 Vision HBM、Language HBM、Embedding，
不需要 CUDA 编译主机。

### 1. 获取代码并安装 Python 依赖

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything/LocateAnything

python3 -m venv --system-site-packages .venv-s600
source .venv-s600/bin/activate
python -m pip install -r deploy/requirements.txt
```

### 2. 下载 HBM

从 Hugging Face 模型仓库直接下载 Vision HBM、Language HBM 和 Embedding：

```bash
python -m pip install -U huggingface_hub
export LA_HF_REPO="YOUR_ACCOUNT/LocateAnything-3B-S600"
export LA_RELEASE_ROOT="$PWD/artifacts/releases/direct"
mkdir -p "$LA_RELEASE_ROOT"

hf download "$LA_HF_REPO" \
  --local-dir "$LA_RELEASE_ROOT"
```

确认以下三个文件位于 `LA_RELEASE_ROOT` 根目录：

```text
LocateAnything-3B_vision.hbm
LocateAnything-3B_language.hbm
LocateAnything-3B_embed_tokens.bin
```

### 3. 编译 S600 运行时

```bash
cmake -S deploy -B deploy/build -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/build \
  --target vision_hbm_runner language_hbm_runner \
  -j4
```

### 4. 安装并运行 CLI

```bash
sh deploy/scripts/install.sh
export PATH="$HOME/.local/bin:$PATH"
export LA_RELEASE_ROOT="$PWD/artifacts/releases/direct"
export LA_TOKENIZER_DIR="$PWD/deploy/tokenizer"

LocateAnything \
  -i /path/to/image.jpg \
  -p '/detect person,motorcycle' \
  --output-dir artifacts/runs/predict/demo
```

CLI 启动后会常驻加载 Vision 和 Language HBM。检测、定位和点选命令示例：

```text
/detect person,motorcycle
/ground the orange in the center
/point the person's head
/text
/layout title,table,figure
```

每次推理依次显示 Vision、Language、Postprocess 三个阶段及耗时。检测或点定位成功时，
CLI 自动保存标注图，无需额外开启画框参数。

## 推理结果

指定 `--output-dir` 时，单次推理的结果集中保存在同一目录：

```text
artifacts/runs/predict/demo/
└── request_0001/
    ├── prediction.json    模型文本、坐标、原始框和 NMS 后的框
    ├── annotated.png      在原图上绘制的框或点
    ├── timings.json       Vision、Language、Postprocess 和总耗时
    └── logs/
        └── runtime.log    Vision 与 Language runner 日志
```

CLI 会为会话中的每次请求创建 `request_0001`、`request_0002` 等独立目录。不指定
`--output-dir` 时，程序在 `artifacts/runs/predict/` 下创建带时间和图片名的运行目录，
不会把结果写到输入图片旁边。

## 方式二：从零校准和编译

该方式需要 x86_64 CUDA 主机、D-Robotics S600 OELLM 1.0.5 SDK、原始模型和
符合 LocateAnything 输入格式的校准数据。

### 1. 下载 SDK

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
```

按照 SDK 内的安装说明创建工具链环境。以下命令假设环境名为 `oellm_clean`。

### 2. 安装编译依赖

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything/LocateAnything

git clone https://github.com/NVlabs/Eagle.git artifacts/upstream/Eagle

source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

python -m pip install -r compiler/requirements-host.txt
python -m pip install -e compiler --no-deps
```

### 3. 准备模型和校准数据

```bash
hf download nvidia/LocateAnything-3B \
  --local-dir artifacts/models/LocateAnything-3B

export LA_UPSTREAM_SOURCE="$PWD/artifacts/upstream/Eagle/Embodied"
```

校准数据可以使用项目提供的数据，也可以使用自己的数据。下载项目数据：

```bash
mkdir -p artifacts/calibration/download
hf download xkj521999/OE_LA_Calibration_data source.zip \
  --repo-type dataset \
  --local-dir artifacts/calibration/download

mkdir -p artifacts/calibration/locateanything/source
python -m zipfile -e artifacts/calibration/download/source.zip \
  artifacts/calibration/locateanything/source
```

下载或准备完成后，校准数据目录应为：

```text
artifacts/calibration/locateanything/
└── source/
    ├── selected.jsonl
    └── images/
```

`selected.jsonl` 中每条记录只需包含 `bundle_id`、`task`、`image` 和 `prompt`。
`image` 可以是绝对路径，也可以是相对 `selected.jsonl` 的路径；发布共享数据时建议使用
相对路径。`source_width`、`source_height` 和 `target_response` 均可选。样本数量和
收敛检查点由程序根据实际数据自动确定，不要求使用项目数据的任务配额或 Prompt 模板。

使用默认项目目录时不需要设置环境变量。若希望将所有生成物放到其他磁盘，只需设置
`LA_ARTIFACTS_ROOT`，目录内部仍保持 `calibration/locateanything/source/` 结构。

### 4. 使用 `standard` 图集合校准和编译

`compiler/quantize.py` 是统一入口。长任务显示当前阶段、完成数、已用时间和可计算的
预计剩余时间；`--resume` 只复用完整且配置一致的产物。

```bash
CONFIG=compiler/configs/standard.yaml

python compiler/quantize.py --config "$CONFIG" prepare --preflight-only
python compiler/quantize.py --config "$CONFIG" prepare --resume
python compiler/quantize.py --config "$CONFIG" calibrate --component all --resume
python compiler/quantize.py --config "$CONFIG" build --component all --target bc --resume
python compiler/quantize.py --config "$CONFIG" build --component all --target hbm --resume
python compiler/quantize.py --config "$CONFIG" verify --component all --stage specification
```

构建产物和日志分别保存到：

```text
artifacts/calibration/locateanything/generated/
artifacts/calibration/locateanything/statistics/standard/
artifacts/builds/locateanything-3b/standard/
artifacts/logs/locateanything-3b/standard/
```

### 5. 使用 `fused_decode` 图集合

融合解码使用独立配置和独立输出目录：

```bash
CONFIG=compiler/configs/fused_decode.yaml

python compiler/quantize.py --config "$CONFIG" calibrate --component language --resume
python compiler/quantize.py --config "$CONFIG" build --component language --target bc --resume
python compiler/quantize.py --config "$CONFIG" build --component language --target hbm --resume
python compiler/quantize.py --config "$CONFIG" verify --component language --stage specification
```

### 6. 部署编译产物

部署脚本先在本地检查必需文件和文件大小，再上传到 S600 的临时目录；S600 校验完成并
编译运行时后，才生成最终目录。同名目录不会被覆盖。

```bash
BUILD_DIR="$PWD/artifacts/builds/locateanything-3b/standard"

bash deploy/scripts/deploy.sh \
  --release locateanything-s600-standard-v1 \
  --vision-hbm "$BUILD_DIR/vision/LocateAnything-3B_vision_672x672_w8_nash-p_corenum_4.hbm" \
  --language-hbm "$BUILD_DIR/language/LocateAnything-3B_language_chunk_1024_cache_4096_decoder_w8_lmhead_w8_nash-p_corenum_4_4.hbm" \
  --embed-bin "$BUILD_DIR/language/LocateAnything-3B_embed_tokens.bin" \
  --runtime-config deploy/config/runtime.json \
  --tokenizer-dir deploy/tokenizer \
  --ssh-target sunrise@S600_IP \
  --execute
```

部署完成后在 S600 上运行：

```bash
cd /home/sunrise/oe_locateanything/LocateAnything/artifacts/releases/locateanything-s600-standard-v1
sh deploy/scripts/install.sh
export PATH="$HOME/.local/bin:$PATH"

LocateAnything \
  -c config/locateanything_3b_config.json \
  -i /path/to/image.jpg \
  -p '/detect person,motorcycle' \
  --output-dir artifacts/runs/predict/demo
```

## Hugging Face 资源

Hugging Face 只保存两类可复用资源：

- Dataset 仓库保存原始校准数据 `source.zip`；
- Model 仓库保存 Vision HBM、Language HBM 和 Embedding。

`generated/` 和 `statistics/` 由用户在本地执行 Prepare 与 Calibrate 后生成，不上传。
源码、运行日志、评测结果和报告图也不放入 Hugging Face 仓库。

## 项目结构

```text
LocateAnything/
├── compiler/              量化、校准、BC/HBO/HBM 编译
│   ├── configs/           `standard` 与 `fused_decode` 构建配置
│   ├── leap_llm/          OELLM 模型适配
│   ├── scripts/           构建和验证实现
│   └── quantize.py        统一编译入口
├── deploy/                S600 运行时、CLI 与部署
│   ├── apps/              Vision 与 Language HBM runner
│   ├── bin/               LocateAnything CLI 启动器
│   ├── config/            S600 运行配置
│   ├── include/           C++ 运行时头文件
│   ├── python/            CLI、推理流程与资源监控
│   ├── scripts/           安装与板端部署脚本
│   ├── src/               C++ 运行时实现
│   └── tokenizer/         LocateAnything tokenizer
└── artifacts/             本地模型、校准、构建、运行和发布产物
```

`artifacts/` 不提交到 Git。源码目录不保存模型、校准张量、编译产物、运行日志或报告图。

## License

本项目使用 [CC BY-NC 4.0](LICENSE)。模型权重、D-Robotics SDK 和上游组件遵循各自
许可证。
