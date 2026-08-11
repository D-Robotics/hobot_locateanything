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

`hobot_locateanything` 是 LocateAnything-3B 在地瓜机器人 RDK S600 上的 TROS 推理功能包。项目提供读取本地媒体的交互终端，以及订阅 TROS 图像与 Prompt、发布 `ai_msgs/msg/PerceptionTargets` 的 ROS 2 节点。两个程序共用同一套 C++ 推理核心。

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

共用推理流程为 `图像 + Prompt -> 预处理 -> MoonViT -> Qwen2.5 Decoder -> 结构化结果解析`。

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

模型校准与 HBM 编译由 [Locateanything_PTQ](https://github.com/LiuAnclouds/Locateanything_PTQ) 维护。

## 开发环境

| 项目 | 版本 |
| --- | --- |
| 硬件 | 地瓜机器人 RDK S600，AArch64 |
| 系统 | Ubuntu 24.04 LTS |
| TROS | Jazzy |
| 开发语言 | C++17 |
| 编译工具 | CMake、colcon |
| 依赖 | `rclcpp`、`sensor_msgs`、`std_msgs`、`hbm_img_msgs`、`ai_msgs`、OpenCV、yaml-cpp |

## 使用介绍

### 1. 下载模型

```bash
mkdir -p "$HOME/tros_ws/src"
cd "$HOME/tros_ws/src"
git clone https://github.com/LiuAnclouds/hobot_locateanything.git
cd hobot_locateanything

python3 -m pip install -U huggingface_hub
hf download LiuAnclouds/LocateAnything-3B-S600 --local-dir models
```

运行时需要以下文件：

```text
models/
├── LocateAnything-3B_vision.hbm
├── LocateAnything-3B_language.hbm
├── LocateAnything-3B_embed_tokens.bin
└── tokenizer/
    ├── vocab.json
    ├── merges.txt
    └── added_tokens.json
```

### 2. 编译功能包

```bash
source /opt/tros/jazzy/setup.bash
cd "$HOME/tros_ws"
colcon build --packages-select hobot_locateanything
source install/setup.bash
```

### 3. 启动 TROS 推理

#### 交互终端推理

```bash
cd "$HOME/tros_ws/src/hobot_locateanything"
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 run hobot_locateanything console --config "$PWD/config.yaml"
```

RDK S600 终端输出：

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
HBM loaded  [============================] 15.9 s
Ready  S600/Nash-P  |  hybrid  |  max tokens 4096
[User] <<< /image image/07_detection_multiclass.jpg
Image loaded  image/07_detection_multiclass.jpg
[User] <<< /detect person,bus,bicycle
[Assistant] >>> /detect person,bus,bicycle
Performance
  Vision   251.8 ms
  Prefill  150.0 ms  620 tokens
  Decode   525.2 ms  47 tokens  89.5 tokens/s
  Total    969.5 ms
Result
  Labels bicycle, bus, person  |  Boxes 6  |  Points 0  |  Stop im_end
```

#### ROS 节点推理

订阅 `/hbmem_img` 共享内存图像：

```bash
cd "$HOME/tros_ws/src/hobot_locateanything"
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 run hobot_locateanything hobot_locateanything \
  --ros-args \
  --params-file "$PWD/config.yaml" \
  -p input_topic:=/hbmem_img \
  -p is_shared_mem_sub:=true
```

订阅 `/image` 普通 `sensor_msgs/msg/Image`：

```bash
ros2 run hobot_locateanything hobot_locateanything \
  --ros-args \
  --params-file "$PWD/config.yaml" \
  -p input_topic:=/image \
  -p is_shared_mem_sub:=false
```

节点运行期间可更新 Prompt：

```bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  '{data: "/detect kite"}'
```

RDK S600 节点输出：

```text
[INFO] [hobot_locateanything]: ready: input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything
[INFO] [hobot_locateanything]: prompt updated: /detect kite
[INFO] [hobot_locateanything]: frame_id=24643 prompt="/detect kite" output="<ref>kite</ref><box><403><458><832><999></box><|im_end|>" labels="kite" boxes=1 points=0 fps=2 stop_reason=im_end prompt_tokens=615 generated_tokens=11 pbd_calls=3 pbd_accepted_tokens=11 mode=hybrid preprocess_ms=15.850 vision_ms=247.862 language_ms=303.215 postprocess_ms=0.013 total_ms=566.942
```

MIPI、USB 摄像头和本地媒体发布由 TROS 节点负责。本节点只订阅 `/hbmem_img` 或 `/image`。

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

以下数据来自 RDK S600 上的稳定 W8 HBM。表中时间为单图推理耗时，不是摄像头输入帧率。

| 任务 | 输出 Token | Vision (ms) | Prefill (ms) | Decode (ms) | 总耗时 (ms) | Decode (Token/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 目标检测 | 47 | 251.8 | 150.0 | 525.2 | 969.5 | 89.5 |
| GUI 定位 | 14 | 253.2 | 149.7 | 266.0 | 720.7 | 52.6 |
| 指代定位 | 14 | 246.0 | 152.3 | 164.5 | 603.6 | 85.1 |
| OCR | 66 | 245.5 | 152.4 | 665.3 | 1148.3 | 99.2 |
| 指定文本定位 | 15 | 253.0 | 150.2 | 166.6 | 653.5 | 90.0 |
| 版面定位 | 43 | 245.4 | 151.8 | 448.1 | 904.7 | 96.0 |
| 点定位 | 37 | 246.0 | 152.2 | 480.5 | 923.5 | 77.0 |
