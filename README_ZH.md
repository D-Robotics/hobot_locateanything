[English](./README.md) | 简体中文

# hobot_locateanything

![RDK S600](https://img.shields.io/badge/RDK-S600-2F6BFF)
![TROS Jazzy](https://img.shields.io/badge/TROS-Jazzy-00A6A6)
![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)
![LocateAnything-3B](https://img.shields.io/badge/model-LocateAnything--3B-4C8C4A)
![W8](https://img.shields.io/badge/quantization-W8-E67E22)

<p align="center">
  <img src="assets/LocateAnything.jpg" alt="LocateAnything 在 RDK S600 上运行" width="100%">
</p>

`hobot_locateanything` 是 LocateAnything-3B 的 RDK S600 TROS 推理功能包。Console 读取本地图片和视频，并保存标注结果；ROS 2 节点订阅 TROS 图像与 Prompt，发布 `ai_msgs/msg/PerceptionTargets`。两个入口共用同一套 C++ 推理核心。

## 功能介绍

### 支持任务

| 命令 | 任务 |
| --- | --- |
| `/detect person,car` | 开放词汇目标检测 |
| `/ground <phrase>` | 指代定位 |
| `/ground_single <phrase>` | 单目标指代定位 |
| `/gui <element>` | GUI 点定位 |
| `/gui_box <element>` | GUI 区域定位 |
| `/text` | OCR 文本与区域识别 |
| `/ground_text <text>` | 指定文本定位 |
| `/layout title,table,figure` | 文档版面定位 |
| `/point <target>` | 点定位 |

### 模型与量化

<p align="center">
  <img src="assets/LocateAnything_pipeline.png" alt="LocateAnything 推理流程" width="100%">
</p>

推理流程为 `图像 + Prompt -> 预处理 -> MoonViT -> Qwen2.5 Decoder -> 结构化结果解析`。

| 项目 | 配置 |
| --- | --- |
| Vision | MoonViT，27 个 Block，`672 x 672`，有符号 W8 权重 |
| Language | Qwen2.5 Decoder，36 层，Hidden Size 2048，有符号 W8 权重 |
| 激活 | 动态量化 |
| Visual Token | 576 |
| LM Head | W8，词表大小 152681 |
| Prefill / KV Cache | 1024 / 4096 Token |
| 解码 | PBD q=6、AR q=1、Host 采样 |
| 运行平台 | Nash-P，4 个 BPU 核，L2 `6:6:6:6` |

模型校准与 HBM 编译由 [Locateanything_PTQ](https://github.com/D-Robotics/Locateanything_PTQ) 维护。部署文件发布在 [D-Robotics/LocateAnything-3B-BPU](https://huggingface.co/D-Robotics/LocateAnything-3B-BPU)。

## 开发环境

| 项目 | 版本 |
| --- | --- |
| 硬件 | 地瓜机器人 RDK S600，AArch64 |
| 系统 | Ubuntu 24.04 LTS |
| TROS | Jazzy |
| 开发语言 | C++17 |
| 编译工具 | CMake、colcon |
| 依赖 | `rclcpp`、`sensor_msgs`、`std_msgs`、`hbm_img_msgs`、`ai_msgs`、`hobot_codec`、OpenCV、yaml-cpp |

## 使用介绍

### 1. 编译功能包

```bash
git clone https://github.com/D-Robotics/hobot_locateanything.git
cd hobot_locateanything

source /opt/tros/jazzy/setup.bash
colcon build --merge-install --packages-select hobot_locateanything
source install/setup.bash
```

### 2. 下载模型

```bash
mkdir -p install/lib/hobot_locateanything/models/tokenizer
cd install/lib/hobot_locateanything/models

wget -c https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_vision.hbm
wget -c https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_language.hbm
wget -c https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_embed_tokens.bin
wget -c -P tokenizer https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/tokenizer/vocab.json
wget -c -P tokenizer https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/tokenizer/merges.txt
wget -c -P tokenizer https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/tokenizer/added_tokens.json

cd ../../../..
```

运行时读取以下文件：

```text
install/lib/hobot_locateanything/models/
├── LocateAnything-3B_vision.hbm
├── LocateAnything-3B_language.hbm
├── LocateAnything-3B_embed_tokens.bin
└── tokenizer/
    ├── vocab.json
    ├── merges.txt
    └── added_tokens.json
```

### 3. 启动推理

#### 交互终端推理

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 run hobot_locateanything console
```

先加载图片，再输入任务命令：

```text
[User] <<< /image image/07_detection_multiclass.jpg
[User] <<< /detect person,bus,bicycle
```

视频输入使用：

```text
[User] <<< /video image/person_video.avi
[User] <<< /detect person
```

Console 以输入文件名创建输出目录。同一文件重复推理时覆盖原结果。

```text
outputs/07_detection_multiclass/
├── annotated.jpg
└── prediction.json

outputs/person_video/
├── annotated.mp4
├── predictions.jsonl
└── summary.json
```

系统缺少 MP4 编码器时，标注视频保存为 `annotated.avi`。

RDK S600 终端输出：

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
HBM loaded  [============================] 15.9 s
Ready  S600/Nash-P  |  hybrid  |  max tokens 4096
[Assistant] >>> /detect person,bus,bicycle
Performance
  Vision   252.5 ms
  Prefill  149.9 ms  620 tokens
  Decode   525.0 ms  47 tokens  89.5 tokens/s
  Host     41.6 ms
  Total    970.5 ms
Result
  Labels bicycle, bus, person  |  Boxes 6  |  Points 0  |  Stop im_end
```

#### ROS 2 节点推理

本地图片回灌和 USB 摄像头是两套独立流程。Launch 同时启动官方图像节点、Codec 和 LocateAnything 推理节点；收到有效 Prompt 前，推理节点不会处理图像。

##### 本地图片回灌

终端 1，启动本地图片回灌和推理节点并等待 `ready`：

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=fb
ros2 launch hobot_locateanything hobot_locateanything.launch.py
```

终端 2，持续订阅结构化结果：

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic echo \
  /perception/locateanything \
  ai_msgs/msg/PerceptionTargets
```

终端 3，发布 Prompt：

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic pub --once \
  /locateanything/prompt \
  std_msgs/msg/String \
  "{data: '/detect person'}"
```

有效 Prompt 会持续生效，直到新的有效 Prompt 覆盖或节点重启。

Launch 默认回灌安装目录中的 `07_detection_multiclass.jpg`，并以 2 FPS 持续发布。使用其他图片时传入 `publish_image_source:=/absolute/path/image.jpg`。

需要在同一张图片上切换任务时，保持终端 1、2 运行，直接在终端 3 发布新的 Prompt：

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic pub --once \
  /locateanything/prompt \
  std_msgs/msg/String \
  "{data: '/detect bus'}"
```

后续接收的同一张回灌图片使用 `/detect bus`，不需要重启图片发布节点或结果订阅。

##### USB 摄像头输入

终端 1，启动 USB 摄像头和推理节点并等待 `ready`：

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=usb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
  device:=/dev/video0 \
  locateanything_image_width:=1280 \
  locateanything_image_height:=720
```

终端 2，持续订阅结构化结果：

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic echo \
  /perception/locateanything \
  ai_msgs/msg/PerceptionTargets
```

终端 3，发布 Prompt：

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic pub --once \
  /locateanything/prompt \
  std_msgs/msg/String \
  "{data: '/ground cardboard box'}"
```

摄像头持续发布期间可直接在终端 3 发布新的 Prompt，后续新帧使用新 Prompt，无需重启摄像头节点或结果订阅。

ROS 节点只发布结果，不绘图、不编码、不保存文件。渲染和存储由下游 TROS 节点完成。

## 结果展示

### 推理效果

目标检测，Prompt：`/detect person,bus,bicycle`

<img src="assets/results/detection_multiclass.jpg" alt="目标检测" width="720">

GUI 定位，查询：`Go to file/function`、`Environment tab`、`Files tab`

<img src="assets/results/gui_rstudio.jpg" alt="GUI 定位" width="720">

指代定位，查询：`person wearing a graduation cap`、`woman in a black dress`、`clock tower`

<img src="assets/results/referring_graduation.jpg" alt="指代定位" width="520">

OCR，Prompt：`/text`

<img src="assets/results/ocr_scrapbook.jpg" alt="OCR" width="720">

指定文本定位，查询：`LIVE love LAUGH`、`laugh giggle be silly`、`Yes Virginia`

<img src="assets/results/ground_text_scrapbook.jpg" alt="指定文本定位" width="720">

版面定位，Prompt：`/layout plot,text`

<img src="assets/results/layout_plot.jpg" alt="版面定位" width="720">

点定位，Prompt：`/point succulent`

<img src="assets/results/point_succulent.jpg" alt="点定位" width="512">

### 性能

| 任务 | 输出 Token | Vision (ms) | Prefill (ms) | Decode (ms) | 总耗时 (ms) | Decode (Token/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 目标检测 | 47 | 252.5 | 149.9 | 525.0 | 970.5 | 89.5 |
| GUI 定位 | 14 | 253.2 | 149.7 | 266.0 | 720.7 | 52.6 |
| 指代定位 | 14 | 246.0 | 152.3 | 164.5 | 603.6 | 85.1 |
| OCR | 66 | 245.5 | 152.4 | 665.3 | 1148.3 | 99.2 |
| 指定文本定位 | 15 | 253.0 | 150.2 | 166.6 | 653.5 | 90.0 |
| 版面定位 | 43 | 245.4 | 151.8 | 448.1 | 904.7 | 96.0 |
| 点定位 | 37 | 246.0 | 152.2 | 480.5 | 923.5 | 77.0 |
