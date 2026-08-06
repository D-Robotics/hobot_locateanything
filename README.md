<div align="center">

<img src="LocateAnything/assets/LocateAnything.jpg" alt="LocateAnything 在 D-Robotics S600 上的检测效果" width="820">

# LocateAnything-3B on D-Robotics S600

在 D-Robotics S600 上运行 LocateAnything-3B，支持开放词汇检测、指代表达定位、点定位、OCR 和版面定位。

</div>

## 项目布局

```text
LocateAnything/
├── compiler/                    # 校准、量化和 BC/HBO/HBM 编译
│   ├── config/                  # 编译配置
│   ├── datasets/                # 校准数据
│   ├── models/                  # Float 模型
│   └── outputs/                 # 编译产物和日志
├── inference/                   # S600 TROS C++ 推理包
│   ├── src/                     # 推理实现、运行器和 TROS 节点
│   ├── include/                 # C++ 推理模块接口
│   ├── config/                  # ROS 参数
│   ├── launch/                  # XML launch 文件
│   ├── models/                  # HBM、Embedding 和 Tokenizer
│   └── outputs/                 # 推理结果
└── README.md
```

编译和推理相互独立。推理端全部使用 C++，不依赖 Python 运行时。

## 默认配置

| 项目 | 配置 |
|---|---|
| 输入尺寸 | 672 x 672，保持宽高比并填充 |
| Vision | W8 |
| Language | W8 |
| LM Head | W8 |
| Prefill | 1024 tokens |
| KV Cache | 4096 tokens |
| PBD | q=6 |
| Language 图 | fused decode，Prefill + PBD q6-q12 + AR q1-q5 |
| 最大生成长度 | 4096 tokens |

默认配置提供完整的 fused decode 流程。常用编译参数集中在
`LocateAnything/compiler/config/quantization.yaml`，推理参数集中在
`LocateAnything/inference/config/locateanything.yaml`。运行时按 HBM 实际图接口执行，允许用户在同步修改编译图定义和 C++ 解码逻辑后扩展图集合。

## 直接部署 HBM

### 1. 获取代码和模型

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything/LocateAnything

mkdir -p inference/models
hf download <模型仓库> --local-dir inference/models
```

模型目录应包含：

```text
inference/models/
├── LocateAnything-3B_vision.hbm
├── LocateAnything-3B_language.hbm
└── LocateAnything-3B_embed_tokens.bin
```

### 2. 构建 TROS C++ 包

```bash
source /opt/tros/jazzy/setup.bash
cd inference
colcon build --merge-install --symlink-install
source install/setup.bash
```

### 3. 启动推理节点

```bash
ros2 launch locateanything locateanything.launch.xml
```

默认任务为 `/detect person`。运行时可通过 Prompt 话题切换任务：

```bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect person,motorcycle'}"
```

### 4. 输入图片、视频或 USB 相机

本地图片：

```bash
ros2 launch locateanything image.launch.xml \
  source:=/path/to/image.jpg
```

本地视频（AVI、MP4 等 OpenCV 支持的格式）：

```bash
ros2 launch locateanything video.launch.xml \
  source:=/path/to/video.mp4
```

USB 相机直接使用 TROS 的 `hobot_usb_cam`：

```bash
ros2 launch hobot_usb_cam hobot_usb_cam.launch.py \
  usb_video_device:=/dev/video0 usb_image_width:=1920 \
  usb_image_height:=1080 usb_framerate:=30 usb_zero_copy:=True
```

图片、视频和 USB 相机使用 TROS 节点发布共享内存 NV12 图像到
`/hbmem_img`。本地视频在上一帧完成后再发布下一帧，保证所有帧都经过模型；
USB 和 MIPI 相机保持实时输入，模型忙碌时只保留最新一帧，不累积过期画面。

### 5. 接入 MIPI 相机

MIPI 相机直接使用 TROS 的 `mipi_cam` 共享内存图像：

```bash
ros2 launch mipi_cam mipi_cam.launch.py \
  mipi_io_method:=shared_mem mipi_frame_ts_type:=realtime
```

图片、视频、USB 和 MIPI 使用同一推理节点，不跨帧复用 Language KV Cache。

### 6. 获取结果

| 话题 | 消息类型 | 内容 |
|---|---|---|
| `/locateanything/result` | `std_msgs/msg/String` | 检测框、点坐标和各阶段耗时 JSON |
| `/locateanything/annotated` | `sensor_msgs/msg/Image` | 已绘制预测结果的图像 |

结果同时保存到配置项 `output_directory` 指定的目录：

```text
inference/outputs/
├── predictions.jsonl
└── frames/
    ├── frame_000001.jpg
    └── ...
```

## 从零校准和编译

从零编译需要 CUDA 主机、D-Robotics S600 OE_LLM SDK、LocateAnything 原始模型和校准数据。

### 1. 安装编译环境

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
```

### 2. 准备模型和校准数据

```bash
cd /path/to/oe_locateanything/LocateAnything
mkdir -p compiler/models/LocateAnything-3B compiler/datasets/calibration/download
hf download nvidia/LocateAnything-3B \
  --local-dir compiler/models/LocateAnything-3B
hf download xkj521999/OE_LA_Calibration_data source.zip \
  --repo-type dataset --local-dir compiler/datasets/calibration/download
python -m zipfile -e compiler/datasets/calibration/download/source.zip \
  compiler/datasets/calibration/locateanything/source
```

也可以使用自有校准数据，并在 `compiler/config/quantization.yaml` 中修改数据路径。
默认只收集构建所需的量化 Scale 和校验信息；需要完整激活分布报告时，将
`calibration.detailed_statistics` 设为 `true`。

### 3. 校准并编译 HBM

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean
python -m pip install -r compiler/requirements-host.txt
python -m pip install -e compiler --no-deps

CONFIG=compiler/config/quantization.yaml
python compiler/quantize.py --config "$CONFIG" prepare --resume
python compiler/quantize.py --config "$CONFIG" calibrate --component all --resume
python compiler/quantize.py --config "$CONFIG" build --component all --target bc --resume
python compiler/quantize.py --config "$CONFIG" build --component all --target hbm --resume
python compiler/quantize.py --config "$CONFIG" verify --component all --stage specification
```

编译产物保存在 `compiler/outputs/`。将最终 Vision HBM、Language HBM 和 Embedding 放入 `inference/models/`，再按直接部署流程构建 TROS 包。

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
