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

### 1. 交互式 Console

Console 直接读取本地图片或视频，不启动 ROS 图像、Prompt 和结果话题。

#### 启动 Console

```bash
cd "$HOME/tros_ws/src/hobot_locateanything"
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 run hobot_locateanything console --config "$PWD/config.yaml"
```

终端输出：

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
Loading Vision HBM...
Loading Language HBM...
HBM loaded  [============================] 16.7 s
Ready  S600/Nash-P  |  hybrid  |  max tokens 4096
```

#### 目标检测

```text
[User] <<< /image image/07_detection_multiclass.jpg
Image loaded  image/07_detection_multiclass.jpg
[User] <<< /detect person,bus,bicycle
[Assistant] >>> /detect person,bus,bicycle
Performance
  Vision   253.5 ms
  Prefill  153.0 ms  620 tokens
  Decode   526.8 ms  47 tokens  89.2 tokens/s
  Host     41.2 ms
  Total    976.7 ms
Result
  Labels bicycle, bus, person  |  Boxes 6  |  Points 0  |  Stop im_end
```

<img src="assets/results/detection_multiclass.jpg" alt="目标检测" width="720">

#### GUI 定位

```text
[User] <<< /image image/02_gui_rstudio.jpg
Image loaded  image/02_gui_rstudio.jpg
[User] <<< /gui_box Go to file/function
[Assistant] >>> /gui_box Go to file/function
Performance
  Vision   245.8 ms
  Prefill  155.5 ms  618 tokens
  Decode   266.0 ms  14 tokens  52.6 tokens/s
  Host     10.8 ms
  Total    719.5 ms
Result
  Labels Go to file/function  |  Boxes 1  |  Points 0  |  Stop im_end

[User] <<< /gui_box Environment tab
[Assistant] >>> /gui_box Environment tab
Performance
  Vision   246.0 ms
  Prefill  156.0 ms  615 tokens
  Decode   125.2 ms  11 tokens  87.8 tokens/s
  Host     9.4 ms
  Total    579.6 ms
Result
  Labels Environment tab  |  Boxes 1  |  Points 0  |  Stop im_end

[User] <<< /gui_box Files tab
[Assistant] >>> /gui_box Files tab
Performance
  Vision   245.9 ms
  Prefill  155.7 ms  615 tokens
  Decode   124.6 ms  11 tokens  88.3 tokens/s
  Host     8.8 ms
  Total    578.6 ms
Result
  Labels Files tab  |  Boxes 1  |  Points 0  |  Stop im_end
```

<img src="assets/results/gui_rstudio.jpg" alt="GUI 定位" width="720">

#### 指代定位

```text
[User] <<< /image image/03_referring_graduation.jpg
Image loaded  image/03_referring_graduation.jpg
[User] <<< /ground person wearing a graduation cap
[Assistant] >>> /ground person wearing a graduation cap
Performance
  Vision   245.8 ms
  Prefill  156.1 ms  619 tokens
  Decode   165.9 ms  14 tokens  84.4 tokens/s
  Host     10.7 ms
  Total    608.8 ms
Result
  Labels person wearing a graduation cap  |  Boxes 1  |  Points 0  |  Stop im_end

[User] <<< /ground woman in a black dress
[Assistant] >>> /ground woman in a black dress
Performance
  Vision   246.2 ms
  Prefill  156.1 ms  619 tokens
  Decode   164.5 ms  14 tokens  85.1 tokens/s
  Host     9.3 ms
  Total    608.1 ms
Result
  Labels woman in a black dress  |  Boxes 1  |  Points 0  |  Stop im_end

[User] <<< /ground clock tower
[Assistant] >>> /ground clock tower
Performance
  Vision   245.7 ms
  Prefill  155.9 ms  616 tokens
  Decode   124.5 ms  11 tokens  88.3 tokens/s
  Host     8.5 ms
  Total    567.1 ms
Result
  Labels clock tower  |  Boxes 1  |  Points 0  |  Stop im_end
```

<img src="assets/results/referring_graduation.jpg" alt="指代定位" width="520">

#### OCR

```text
[User] <<< /image image/04_ocr_scrapbook.jpg
Image loaded  image/04_ocr_scrapbook.jpg
[User] <<< /text
[Assistant] >>> /text
Performance
  Vision   246.2 ms
  Prefill  155.7 ms  610 tokens
  Decode   666.4 ms  66 tokens  99.0 tokens/s
  Host     63.0 ms
  Total    1153.9 ms
Result
  Labels LIVE love LAUGH, Yes, Virginiaina, [to-day]], laugh giggle be silly
  Boxes 5  |  Points 0  |  Stop im_end
```

<img src="assets/results/ocr_scrapbook.jpg" alt="OCR" width="720">

#### 指定文本定位

```text
[User] <<< /image image/04_ocr_scrapbook.jpg
Image loaded  image/04_ocr_scrapbook.jpg
[User] <<< /ground_text LIVE love LAUGH
[Assistant] >>> /ground_text LIVE love LAUGH
Performance
  Vision   245.3 ms
  Prefill  155.5 ms  613 tokens
  Decode   165.0 ms  15 tokens  90.9 tokens/s
  Host     10.3 ms
  Total    649.8 ms
Result
  Labels LIVE love LAUGH.  |  Boxes 1  |  Points 0  |  Stop im_end

[User] <<< /ground_text laugh giggle be silly
[Assistant] >>> /ground_text laugh giggle be silly
Performance
  Vision   245.6 ms
  Prefill  155.4 ms  614 tokens
  Decode   164.8 ms  16 tokens  97.1 tokens/s
  Host     10.6 ms
  Total    650.2 ms
Result
  Labels laugh giggle be silly.  |  Boxes 1  |  Points 0  |  Stop im_end

[User] <<< /ground_text Yes Virginia
[Assistant] >>> /ground_text Yes Virginia
Performance
  Vision   245.5 ms
  Prefill  155.3 ms  611 tokens
  Decode   124.9 ms  12 tokens  96.1 tokens/s
  Host     9.1 ms
  Total    610.1 ms
Result
  Labels Yes Virginia.  |  Boxes 1  |  Points 0  |  Stop im_end
```

<img src="assets/results/ground_text_scrapbook.jpg" alt="指定文本定位" width="720">

#### 版面定位

```text
[User] <<< /image image/05_layout_plot.jpg
Image loaded  image/05_layout_plot.jpg
[User] <<< /layout plot,text
[Assistant] >>> /layout plot,text
Performance
  Vision   245.6 ms
  Prefill  155.0 ms  620 tokens
  Decode   448.1 ms  43 tokens  96.0 tokens/s
  Host     37.2 ms
  Total    908.8 ms
Result
  Labels plot, text  |  Boxes 6  |  Points 0  |  Stop im_end
```

<img src="assets/results/layout_plot.jpg" alt="版面定位" width="720">

#### 点定位

```text
[User] <<< /image image/06_pointing_succulent.jpg
Image loaded  image/06_pointing_succulent.jpg
[User] <<< /point succulent
[Assistant] >>> /point succulent
Performance
  Vision   246.6 ms
  Prefill  155.6 ms  608 tokens
  Decode   481.8 ms  37 tokens  76.8 tokens/s
  Host     39.4 ms
  Total    929.3 ms
Result
  Labels succulent  |  Boxes 0  |  Points 8  |  Stop im_end
```

<img src="assets/results/point_succulent.jpg" alt="点定位" width="512">

Console 将图片结果保存到 `outputs/<图片名>/annotated.jpg` 和 `prediction.json`。视频使用 `/video` 加载，任务命令与图片一致。

### 2. ROS 2 节点

ROS 2 推理使用独立的图像、Prompt、推理和结果话题。以下命令分别在不同终端执行。

#### 启动推理节点

```bash
cd "$HOME/tros_ws/src/hobot_locateanything"
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/tros/jazzy/lib/hobot_shm/config/shm_fastdds.xml
export RMW_FASTRTPS_USE_QOS_FROM_XML=1

ros2 run hobot_locateanything hobot_locateanything \
  --ros-args \
  --params-file "$PWD/config.yaml" \
  -p input_topic:=/hbmem_img \
  -p is_shared_mem_sub:=true
```

推理节点输出：

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 16.5 s
[INFO] [hobot_locateanything]: ready: input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything
[WARN] [hobot_locateanything]: waiting for prompt on /locateanything/prompt; image frames are ignored until a valid prompt arrives
```

#### 本地图片回灌

结果订阅节点：

```bash
source /opt/tros/jazzy/setup.bash
ros2 topic echo --once /perception/locateanything
```

Prompt 节点：

```bash
source /opt/tros/jazzy/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect person,bus,bicycle'}"
```

Prompt 节点输出：

```text
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/detect person,bus,bicycle')
```

TROS 图片发布节点：

```bash
source /opt/tros/jazzy/setup.bash
ros2 launch hobot_image_publisher hobot_image_publisher.launch.py \
  publish_image_source:="$HOME/tros_ws/src/hobot_locateanything/image/07_detection_multiclass.jpg" \
  publish_image_format:=jpg \
  publish_message_topic_name:=/hbmem_img \
  publish_fps:=1 \
  publish_is_loop:=True \
  publish_is_shared_mem:=True \
  publish_encoding:=nv12
```

图片发布节点输出：

```text
[INFO] [hobot_image_pub-1]: process started
[image_pub_node]: parameter:
 image_source: image/07_detection_multiclass.jpg
 fps: 1
 is_shared_mem: 1
 is_loop: 1
 image_format: jpg
 pub_encoding: nv12
 msg_pub_topic_name: /hbmem_img
[hobot_image_pub]: Enabling zero-copy
```

推理节点收到图像后的输出：

```text
[INFO] [hobot_locateanything]: prompt updated: /detect person,bus,bicycle
[INFO] [hobot_locateanything]: frame_id=1 prompt="/detect person,bus,bicycle" output="<ref>person</ref><box><125><356><245><755></box><box><720><400><842><766></box><ref>bus</ref><box><0><184><623><705></box><ref>bicycle</ref><box><522><464><697><678></box><box><694><427><788><541></box><box><811><600><998><875></box><|im_end|>" labels="person | person | bus | bicycle | bicycle | bicycle" boxes=6 points=0 fps=1 stop_reason=im_end prompt_tokens=620 generated_tokens=47 pbd_calls=10 pbd_accepted_tokens=44 mode=hybrid preprocess_ms=17.760 vision_ms=251.024 language_ms=705.680 postprocess_ms=0.023 total_ms=974.489
```

结果话题输出：

```yaml
header:
  frame_id: '1'
fps: 1
perfs:
- type: preprocess
  time_ms_duration: 17.760361
- type: vision
  time_ms_duration: 251.024219
- type: language
  time_ms_duration: 705.679849
- type: postprocess
  time_ms_duration: 0.022526
targets:
- type: person
  rois:
  - type: person
    rect: {x_offset: 80, y_offset: 148, height: 255, width: 77}
    confidence: -1.0
- type: person
  rois:
  - type: person
    rect: {x_offset: 461, y_offset: 176, height: 234, width: 78}
    confidence: -1.0
- type: bus
  rois:
  - type: bus
    rect: {x_offset: 0, y_offset: 38, height: 333, width: 399}
    confidence: -1.0
- type: bicycle
  rois:
  - type: bicycle
    rect: {x_offset: 334, y_offset: 217, height: 137, width: 112}
    confidence: -1.0
- type: bicycle
  rois:
  - type: bicycle
    rect: {x_offset: 444, y_offset: 193, height: 73, width: 60}
    confidence: -1.0
- type: bicycle
  rois:
  - type: bicycle
    rect: {x_offset: 519, y_offset: 304, height: 176, width: 120}
    confidence: -1.0
```

#### USB 摄像头输入

重新启动结果订阅节点：

```bash
source /opt/tros/jazzy/setup.bash
ros2 topic echo --once /perception/locateanything
```

更新 Prompt：

```bash
source /opt/tros/jazzy/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/ground cardboard box'}"
```

Prompt 节点输出：

```text
Waiting for at least 1 matching subscription(s)...
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/ground cardboard box')
```

TROS USB 摄像头节点：

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

USB 摄像头节点输出：

```text
[INFO] [hobot_usb_cam-1]: process started
[hobot_usb_cam]: framerate: 30
[hobot_usb_cam]: pixel_format_name: mjpeg
[hobot_usb_cam]: Camera calibration file: [/opt/tros/jazzy/lib/hobot_usb_cam/config/usb_camera_calibration.yaml] does not exist!
[hobot_usb_cam]: Enabling zero-copy
```

推理节点收到摄像头图像后的输出：

```text
[INFO] [hobot_locateanything]: prompt updated: /ground cardboard box
[INFO] [hobot_locateanything]: frame_id=13 prompt="/ground cardboard box" output="<ref>cardboard box</ref><box><503><613><556><655></box><|im_end|>" labels="cardboard box" boxes=1 points=0 fps=2 stop_reason=im_end prompt_tokens=616 generated_tokens=12 pbd_calls=3 pbd_accepted_tokens=12 mode=hybrid preprocess_ms=27.256 vision_ms=253.491 language_ms=304.637 postprocess_ms=0.011 total_ms=585.397
```

结果话题输出：

```yaml
header:
  frame_id: '13'
fps: 2
perfs:
- type: preprocess
  time_ms_duration: 27.255518
- type: vision
  time_ms_duration: 253.491319
- type: language
  time_ms_duration: 304.637464
- type: postprocess
  time_ms_duration: 0.011100
targets:
- type: cardboard box
  rois:
  - type: cardboard box
    rect: {x_offset: 644, y_offset: 505, height: 53, width: 68}
    confidence: -1.0
```

ROS 2 节点只发布结构化结果，绘制、编码和保存由下游 TROS 节点完成。

## 性能

| 任务 | 输出 Token | Vision（ms） | Prefill（ms） | Decode（ms） | 总耗时（ms） | Decode（Token/s） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 目标检测 | 47 | 253.5 | 153.0 | 526.8 | 976.7 | 89.2 |
| GUI 定位 | 14 | 245.8 | 155.5 | 266.0 | 719.5 | 52.6 |
| 指代定位 | 14 | 245.8 | 156.1 | 165.9 | 608.8 | 84.4 |
| OCR | 66 | 246.2 | 155.7 | 666.4 | 1153.9 | 99.0 |
| 指定文本定位 | 15 | 245.3 | 155.5 | 165.0 | 649.8 | 90.9 |
| 版面定位 | 43 | 245.6 | 155.0 | 448.1 | 908.8 | 96.0 |
| 点定位 | 37 | 246.6 | 155.6 | 481.8 | 929.3 | 76.8 |
