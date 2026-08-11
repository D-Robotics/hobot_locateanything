# LocateAnything 视觉定位模型

## 功能介绍

`hobot_locateanything` 在 RDK S600 上运行 LocateAnything-3B，可完成开放词汇目标检测、指代定位、GUI 定位、OCR、版面分析和点定位。项目包含两个入口：Console 读取本地图片或视频，ROS 2 节点订阅 TROS 图像和 Prompt，并发布 `ai_msgs/msg/PerceptionTargets`。

代码仓库：[https://github.com/LiuAnclouds/hobot_locateanything](https://github.com/LiuAnclouds/hobot_locateanything)

模型仓库：[https://huggingface.co/xkj521999/LocateAnything-3B-S600](https://huggingface.co/xkj521999/LocateAnything-3B-S600)

| Prompt | 功能 |
| --- | --- |
| `/detect person,car` | 开放词汇目标检测 |
| `/ground <phrase>`、`/ground_single <phrase>` | 指代定位 |
| `/gui <element>`、`/gui_box <element>` | GUI 点或区域定位 |
| `/text` | OCR 文本与区域识别 |
| `/ground_text <text>` | 指定文本定位 |
| `/layout title,table,figure` | 文档版面定位 |
| `/point <target>` | 点定位 |

## 支持平台

| 平台 | 系统 | 运行方式 |
| --- | --- | --- |
| RDK S600 | Ubuntu 24.04，TROS Jazzy | Console、ROS 2 |

## 算法信息

LocateAnything-3B 由 MoonViT 图像编码器和 Qwen2.5 Decoder 组成。模型使用 W8 权重，在 RDK S600 Nash-P BPU 上运行。

| 项目 | 配置 |
| --- | --- |
| 模型 | LocateAnything-3B，约 3B 参数 |
| Vision | MoonViT，27 层，输入尺寸 `672 x 672` |
| Visual Token | 576 |
| Language | Qwen2.5 Decoder，36 层，Hidden Size 2048 |
| 量化 | Vision、Decoder 和 LM Head 使用有符号 W8 权重，激活动态量化 |
| LM Head | W8，词表大小 152681 |
| Prefill / KV Cache | 1024 / 4096 Token |
| 解码 | PBD q=6、AR q=1、Host 采样 |
| BPU | Nash-P，4 个 BPU 核，L2 `6:6:6:6` |

## 准备工作

RDK S600 需要安装 Ubuntu 24.04 系统镜像和 TogetheROS.Bot Jazzy。

### 下载源码和模型

```bash
mkdir -p "$HOME/tros_ws/src"
cd "$HOME/tros_ws/src"
git clone https://github.com/LiuAnclouds/hobot_locateanything.git
cd hobot_locateanything

python3 -m pip install -U huggingface_hub
export HF_ENDPOINT="https://hf-mirror.com"
hf download xkj521999/LocateAnything-3B-S600 --local-dir models
```

模型目录包含：

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

### 编译功能包

```bash
source /opt/tros/jazzy/setup.bash
cd "$HOME/tros_ws"
colcon build --merge-install --packages-select hobot_locateanything
source install/setup.bash
```

## 使用方式

### 本地图片和视频

Console 直接读取本地媒体，不启动 ROS 图像话题。

```bash
cd "$HOME/tros_ws/src/hobot_locateanything"
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 run hobot_locateanything console --config "$PWD/config.yaml"
```

图片推理：

```text
[User] <<< /image image/07_detection_multiclass.jpg
[User] <<< /detect person,bus,bicycle
```

视频推理：

```text
[User] <<< /video image/person_video.avi
[User] <<< /detect person
```

图片结果保存为 `outputs/<图片名>/annotated.jpg` 和 `prediction.json`。视频结果保存为标注视频、`predictions.jsonl` 和 `summary.json`；系统缺少 MP4 编码器时使用 AVI。同名媒体重复推理时覆盖原结果。

### ROS 2 节点

启动推理节点：

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

节点收到有效 Prompt 后开始处理图像。新开一个终端发布 Prompt：

```bash
source /opt/tros/jazzy/setup.bash

ros2 topic pub --once \
  /locateanything/prompt \
  std_msgs/msg/String \
  "{data: '/detect person,bus,bicycle'}"
```

本地图片单次回灌：

```bash
source /opt/tros/jazzy/setup.bash

ros2 launch hobot_image_publisher hobot_image_publisher.launch.py \
  publish_image_source:="$HOME/tros_ws/src/hobot_locateanything/image/07_detection_multiclass.jpg" \
  publish_image_format:=jpg \
  publish_message_topic_name:=/hbmem_img \
  publish_is_loop:=False \
  publish_is_shared_mem:=True \
  publish_encoding:=nv12
```

USB 摄像头实时输入：

```bash
source /opt/tros/jazzy/setup.bash

ros2 launch hobot_usb_cam hobot_usb_cam.launch.py \
  usb_video_device:=/dev/video0 \
  usb_image_width:=1280 \
  usb_image_height:=720 \
  usb_framerate:=30 \
  usb_pixel_format:=mjpeg \
  usb_io_method:=mmap \
  usb_zero_copy:=True
```

查看推理结果：

```bash
source /opt/tros/jazzy/setup.bash
ros2 topic echo /perception/locateanything
```

ROS 节点发布结构化结果，不绘图、不编码、不保存文件。结果渲染和存储由下游 TROS 节点完成。

## 运行结果

```text
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

<img src="assets/results/detection_multiclass.jpg" alt="LocateAnything 多目标检测结果" width="720">

| 任务 | 输出 Token | Vision（ms） | Prefill（ms） | Decode（ms） | 总耗时（ms） | Decode（Token/s） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 目标检测 | 47 | 252.5 | 149.9 | 525.0 | 970.5 | 89.5 |
| GUI 定位 | 14 | 253.2 | 149.7 | 266.0 | 720.7 | 52.6 |
| 指代定位 | 14 | 246.0 | 152.3 | 164.5 | 603.6 | 85.1 |
| OCR | 66 | 245.5 | 152.4 | 665.3 | 1148.3 | 99.2 |
| 指定文本定位 | 15 | 253.0 | 150.2 | 166.6 | 653.5 | 90.0 |
| 版面定位 | 43 | 245.4 | 151.8 | 448.1 | 904.7 | 96.0 |
| 点定位 | 37 | 246.0 | 152.2 | 480.5 | 923.5 | 77.0 |
