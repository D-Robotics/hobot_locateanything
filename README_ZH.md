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

## 算法简介

[LocateAnything](https://github.com/NVlabs/Eagle/tree/main/Embodied) 是开放语义视觉定位模型，通过文本指令完成目标检测、指代定位、GUI 与文本定位、文档版面定位和点定位。PBD（Parallel Box Decoding）以并行方式生成边界框坐标。

模型仓库：[D-Robotics/LocateAnything-3B-BPU](https://huggingface.co/D-Robotics/LocateAnything-3B-BPU)

模型校准与 HBM 编译：[D-Robotics/Locateanything_PTQ](https://github.com/D-Robotics/Locateanything_PTQ)

## 推理性能

| Platform | 任务 | 输出 Token | Vision (ms) | Prefill (ms) | Decode (ms) | 总耗时 (ms) | Decode (Token/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RDK S600 | 目标检测 | 47 | 254.7 | 151.6 | 526.3 | 978.5 | 89.3 |
| RDK S600 | GUI 定位 | 14 | 253.2 | 149.7 | 266.0 | 720.7 | 52.6 |
| RDK S600 | 指代定位 | 14 | 246.0 | 152.3 | 164.5 | 603.6 | 85.1 |
| RDK S600 | OCR | 66 | 245.5 | 152.4 | 665.3 | 1148.3 | 99.2 |
| RDK S600 | 指定文本定位 | 15 | 253.0 | 150.2 | 166.6 | 653.5 | 90.0 |
| RDK S600 | 版面定位 | 43 | 245.4 | 151.8 | 448.1 | 904.7 | 96.0 |
| RDK S600 | 点定位 | 37 | 246.0 | 152.2 | 480.5 | 923.5 | 77.0 |

## 模型与量化

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

## 开发环境

| 项目 | 版本 |
| --- | --- |
| 硬件 | 地瓜机器人 RDK S600，AArch64 |
| 系统 | Ubuntu 24.04 LTS |
| TROS | Jazzy |
| 开发语言 | C++17 |
| 编译工具 | CMake、colcon |
| 依赖 | `rclcpp`、`sensor_msgs`、`std_msgs`、`hbm_img_msgs`、`ai_msgs`、`hobot_codec`、OpenCV、yaml-cpp |

## 准备工作

RDK S600 需安装 Ubuntu 24.04 和 TogetheROS.Bot Jazzy。

### 编译功能包

```bash
git clone https://github.com/D-Robotics/hobot_locateanything.git
cd hobot_locateanything

source /opt/tros/jazzy/setup.bash
colcon build --merge-install --packages-select hobot_locateanything
source install/setup.bash
```

### 下载模型

```bash
mkdir -p install/lib/hobot_locateanything/models
wget -c -P install/lib/hobot_locateanything/models \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_vision.hbm
wget -c -P install/lib/hobot_locateanything/models \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_language.hbm
wget -c -P install/lib/hobot_locateanything/models \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_embed_tokens.bin
```

运行时文件：

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

## 基础功能：目标检测

### Console 推理

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 run hobot_locateanything console --config config/config.yaml
```

终端输出：

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
Loading Vision HBM...
Loading Language HBM...
HBM loaded  [============================] 16.7 s
Ready  S600/Nash-P  |  hybrid  |  max tokens 4096
Tasks
  /detect cat,dog               目标检测
  /ground <query>[,<query>...]  指代表达，多查询
  /ground_single <query>[,...]  指代表达，单目标查询
  /gui <query>[,<query>...]     GUI 点定位
  /gui_box <query>[,<query>...] GUI 框定位
  /text                         文本 OCR
  /ground_text <query>[,...]    指定文本定位
  /layout title,table,figure    文档版面分析
  /point <query>[,<query>...]   通用点定位
Session
  /image <image_path>           加载图片
  /video <video_path>           加载视频并处理全部帧
  regen                         重跑上次请求
  reset                         清除当前媒体
  exit                          退出程序
```

加载图片：

```text
/image image/07_detection_multiclass.jpg
```

图片加载结果：

```text
Image loaded  image/07_detection_multiclass.jpg
```

输入检测指令：

```text
/detect person,bus,bicycle
```

推理结果：

```text
[Assistant] >>> /detect person,bus,bicycle
Performance
  Vision   254.7 ms
  Prefill  151.6 ms  620 tokens
  Decode   526.3 ms  47 tokens  89.3 tokens/s
  Host     41.4 ms
  Total    978.5 ms
Result
  Labels bicycle, bus, person  |  Boxes 6  |  Points 0  |  Stop im_end
```

结果保存在 `outputs/07_detection_multiclass/annotated.jpg` 和 `prediction.json`。

<img src="assets/results/detection_multiclass.jpg" alt="开放词汇目标检测" width="720">

### ROS 2 推理

结果通过 `/perception/locateanything` 发布，Prompt 通过 `/locateanything/prompt` 更新。

#### 本地图片回灌

默认以 2 FPS 回灌 `image/07_detection_multiclass.jpg`。使用其他图片时修改 `publish_image_source`。

##### 启动命令

终端 1，启动图片回灌和推理节点：

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=fb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
  publish_image_source:=image/07_detection_multiclass.jpg
```

终端 2，订阅检测结果：

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic echo /perception/locateanything ai_msgs/msg/PerceptionTargets
```

终端 3，发布检测 Prompt：

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect person,bus,bicycle'}"
```

##### 运行结果

终端 1，图片发布和推理节点输出：

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 16.5 s
[INFO] [hobot_locateanything]: ready: input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything
[WARN] [hobot_locateanything]: waiting for prompt on /locateanything/prompt; image frames are ignored until a valid prompt arrives
[INFO] [hobot_image_pub-1]: process started
[image_pub_node]: parameter:
 image_source: image/07_detection_multiclass.jpg
 fps: 2
 is_shared_mem: 1
 is_loop: 1
 image_format: jpg
 pub_encoding: nv12
 msg_pub_topic_name: /hbmem_img
[hobot_image_pub]: Enabling zero-copy
[INFO] [hobot_locateanything]: prompt updated: /detect person,bus,bicycle
[INFO] [hobot_locateanything]: frame_id=38 prompt="/detect person,bus,bicycle" output="<ref>person</ref><box><220><392><312><690></box><box><666><424><758><701></box><ref>bus</ref><box><124><265><595><653></box><ref>bicycle</ref><box><514><465><646><618></box><box><735><575><878><782></box><|im_end|>" labels="person | person | bus | bicycle | bicycle" boxes=5 points=0 fps=1 stop_reason=im_end prompt_tokens=620 generated_tokens=41 pbd_calls=9 pbd_accepted_tokens=41 mode=hybrid preprocess_ms=43.935 vision_ms=250.393 language_ms=557.831 postprocess_ms=0.023 total_ms=852.182
```

终端 2，检测结果输出：

```yaml
header:
  frame_id: '38'
fps: 1
perfs:
  - type: preprocess
    time_ms_duration: 43.935122
  - type: vision
    time_ms_duration: 250.392581
  - type: language
    time_ms_duration: 557.831141
  - type: postprocess
    time_ms_duration: 0.022575
targets:
  - type: person
    rois:
      - type: person
        rect: {x_offset: 422, y_offset: 333, height: 572, width: 177}
        confidence: -1.0
  - type: person
    rois:
      - type: person
        rect: {x_offset: 1279, y_offset: 394, height: 532, width: 176}
        confidence: -1.0
  - type: bus
    rois:
      - type: bus
        rect: {x_offset: 238, y_offset: 89, height: 745, width: 904}
        confidence: -1.0
  - type: bicycle
    rois:
      - type: bicycle
        rect: {x_offset: 987, y_offset: 473, height: 294, width: 253}
        confidence: -1.0
  - type: bicycle
    rois:
      - type: bicycle
        rect: {x_offset: 1411, y_offset: 684, height: 396, width: 275}
        confidence: -1.0
```

图片发布节点以 2 FPS 输入，结果话题中的 `fps: 1` 是本次实际推理结果帧率。

终端 3，Prompt 发布输出：

```text
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/detect person,bus,bicycle')
```

终端 3，更新检测 Prompt：

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect bus'}"
```

Prompt 更新后的推理输出：

```text
[INFO] [hobot_locateanything]: prompt updated: /detect bus
[INFO] [hobot_locateanything]: frame_id=44 prompt="/detect bus" output="<ref>bus</ref><box><124><263><595><657></box><|im_end|>" labels="bus" boxes=1 points=0 fps=1 stop_reason=im_end prompt_tokens=615 generated_tokens=10 pbd_calls=3 pbd_accepted_tokens=10 mode=hybrid preprocess_ms=43.837 vision_ms=245.829 language_ms=304.013 postprocess_ms=0.013 total_ms=593.692
```

更新后的检测结果：

```yaml
header:
  frame_id: '48'
fps: 1
targets:
  - type: bus
    rois:
      - type: bus
        rect: {x_offset: 238, y_offset: 85, height: 756, width: 904}
        confidence: -1.0
```

发布新的有效 Prompt 后，后续图像使用新 Prompt，无需重启节点。已经进入推理的帧可能仍输出一次旧 Prompt 结果。

#### USB 摄像头

##### 启动命令

终端 1，启动 USB 摄像头和推理节点：

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=usb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
  device:=/dev/video0 \
  locateanything_image_width:=1280 \
  locateanything_image_height:=720
```

终端 2，订阅检测结果：

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic echo /perception/locateanything ai_msgs/msg/PerceptionTargets
```

终端 3，发布检测 Prompt：

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect cardboard box,person'}"
```

##### 运行结果

终端 1，USB 摄像头和推理节点输出：

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 16.5 s
[INFO] [hobot_locateanything]: ready: input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything
[WARN] [hobot_locateanything]: waiting for prompt on /locateanything/prompt; image frames are ignored until a valid prompt arrives
[INFO] [hobot_usb_cam-1]: process started
[hobot_usb_cam]: framerate: 30
[hobot_usb_cam]: pixel_format_name: mjpeg
[INFO] [hobot_locateanything]: prompt updated: /detect cardboard box,person
[INFO] [hobot_locateanything]: frame_id=532 prompt="/detect cardboard box,person" output="<ref>cardboard box</ref><box><461><615><516><656></box><ref>person</ref><box><381><638><420><780></box><|im_end|>" labels="cardboard box | person" boxes=2 points=0 fps=1 stop_reason=im_end prompt_tokens=618 generated_tokens=21 pbd_calls=5 pbd_accepted_tokens=16 mode=hybrid preprocess_ms=26.183 vision_ms=261.752 language_ms=552.914 postprocess_ms=0.017 total_ms=840.866
```

终端 2，检测结果输出：

```yaml
header:
  frame_id: '532'
fps: 1
perfs:
  - type: preprocess
    time_ms_duration: 26.182972
  - type: vision
    time_ms_duration: 261.751575
  - type: language
    time_ms_duration: 552.913972
  - type: postprocess
    time_ms_duration: 0.017050
targets:
  - type: cardboard box
    rois:
      - type: cardboard box
        rect: {x_offset: 590, y_offset: 507, height: 53, width: 70}
        confidence: -1.0
  - type: person
    rois:
      - type: person
        rect: {x_offset: 488, y_offset: 537, height: 181, width: 50}
        confidence: -1.0
```

终端 3，Prompt 发布输出：

```text
Waiting for at least 1 matching subscription(s)...
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/detect cardboard box,person')
```

摄像头持续发布期间可直接在终端 3 发布新的 Prompt，后续新帧使用新 Prompt，无需重启节点。

ROS 节点发布结构化结果；结果渲染和文件保存由下游 TROS 节点完成。

## 进阶功能

### Console 推理

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 run hobot_locateanything console --config config/config.yaml
```

终端输出：

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
Loading Vision HBM...
Loading Language HBM...
HBM loaded  [============================] 16.7 s
Ready  S600/Nash-P  |  hybrid  |  max tokens 4096
Tasks
  /detect cat,dog               目标检测
  /ground <query>[,<query>...]  指代表达，多查询
  /ground_single <query>[,...]  指代表达，单目标查询
  /gui <query>[,<query>...]     GUI 点定位
  /gui_box <query>[,<query>...] GUI 框定位
  /text                         文本 OCR
  /ground_text <query>[,...]    指定文本定位
  /layout title,table,figure    文档版面分析
  /point <query>[,<query>...]   通用点定位
Session
  /image <image_path>           加载图片
  /video <video_path>           加载视频并处理全部帧
  regen                         重跑上次请求
  reset                         清除当前媒体
  exit                          退出程序
```

多个查询使用逗号分隔。同一图像或视频帧只执行一次 Vision，各项 Language 推理完成后合并结果。

### GUI 定位

加载图片：

```text
/image image/02_gui_rstudio.jpg
```

图片加载结果：

```text
Image loaded  image/02_gui_rstudio.jpg
```

输入定位指令：

```text
/gui_box Go to file/function,Environment tab,Files tab
```

推理结果：

```text
[Assistant] >>> /gui_box Go to file/function,Environment tab,Files tab
Performance
  Vision   252.9 ms
  Prefill  463.7 ms  1848 tokens
  Decode   519.8 ms  36 tokens  69.3 tokens/s
  Host     29.4 ms
  Total    1342.8 ms
Result
  Labels Environment tab, Files tab, Go to file/function  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/gui_rstudio.jpg" alt="GUI 定位" width="720">

### 指代定位

加载图片：

```text
/image image/03_referring_graduation.jpg
```

图片加载结果：

```text
Image loaded  image/03_referring_graduation.jpg
```

输入定位指令：

```text
/ground person wearing a graduation cap,woman in a black dress,clock tower
```

推理结果：

```text
[Assistant] >>> /ground person wearing a graduation cap,woman in a black dress,clock tower
Performance
  Vision   250.4 ms
  Prefill  462.5 ms  1854 tokens
  Decode   461.2 ms  39 tokens  84.6 tokens/s
  Host     29.9 ms
  Total    1268.8 ms
Result
  Labels clock tower, person wearing a graduation cap, woman in a black dress  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/referring_graduation.jpg" alt="指代定位" width="520">

### OCR

加载图片：

```text
/image image/04_ocr_scrapbook.jpg
```

图片加载结果：

```text
Image loaded  image/04_ocr_scrapbook.jpg
```

输入 OCR 指令：

```text
/text
```

推理结果：

```text
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

### 指定文本定位

加载图片：

```text
/image image/04_ocr_scrapbook.jpg
```

图片加载结果：

```text
Image loaded  image/04_ocr_scrapbook.jpg
```

输入定位指令：

```text
/ground_text LIVE love LAUGH,laugh giggle be silly,Yes Virginia
```

推理结果：

```text
[Assistant] >>> /ground_text LIVE love LAUGH,laugh giggle be silly,Yes Virginia
Performance
  Vision   246.0 ms
  Prefill  471.6 ms  1838 tokens
  Decode   459.4 ms  43 tokens  93.6 tokens/s
  Host     30.4 ms
  Total    1311.1 ms
Result
  Labels LIVE love LAUGH., Yes Virginia., laugh giggle be silly.  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/ground_text_scrapbook.jpg" alt="指定文本定位" width="720">

### 版面定位

加载图片：

```text
/image image/05_layout_plot.jpg
```

图片加载结果：

```text
Image loaded  image/05_layout_plot.jpg
```

输入定位指令：

```text
/layout plot,text
```

推理结果：

```text
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

### 点定位

加载图片：

```text
/image image/06_pointing_succulent.jpg
```

图片加载结果：

```text
Image loaded  image/06_pointing_succulent.jpg
```

输入定位指令：

```text
/point succulent,the succulent in the center
```

推理结果：

```text
[Assistant] >>> /point succulent,the succulent in the center
Performance
  Vision   245.9 ms
  Prefill  310.5 ms  1220 tokens
  Decode   645.4 ms  50 tokens  77.5 tokens/s
  Host     47.4 ms
  Total    1272.7 ms
Result
  Labels succulent, the succulent in the center  |  Boxes 0  |  Points 9  |  Stop im_end
```

<img src="assets/results/point_succulent.jpg" alt="点定位" width="512">

## 图片与视频输出

图片结果保存在 `outputs/<图片名>/annotated.jpg` 和 `prediction.json`。

视频通过 `/video` 加载，任务命令与图片一致：

```text
/video image/person_video.avi
/detect person
```

视频结果保存在：

```text
outputs/person_video/
├── annotated.mp4
├── predictions.jsonl
└── summary.json
```
