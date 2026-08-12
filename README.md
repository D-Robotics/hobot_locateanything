English | [简体中文](./README_ZH.md)

# hobot_locateanything

![RDK S600](https://img.shields.io/badge/RDK-S600-2F6BFF)
![TROS Jazzy](https://img.shields.io/badge/TROS-Jazzy-00A6A6)
![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)
![LocateAnything-3B](https://img.shields.io/badge/model-LocateAnything--3B-4C8C4A)
![W8](https://img.shields.io/badge/quantization-W8-E67E22)

<p align="center">
  <img src="assets/LocateAnything.jpg" alt="LocateAnything on RDK S600" width="100%">
</p>

`hobot_locateanything` runs LocateAnything-3B on the D-Robotics RDK S600. The Console reads local images and videos and saves annotated results. The ROS 2 node receives TROS images and prompts, then publishes `ai_msgs/msg/PerceptionTargets`. Both entry points use the same C++ inference core.

## Features

### Tasks

| Command | Task |
| --- | --- |
| `/detect person,car` | Open-vocabulary object detection |
| `/ground <phrase>` | Referring grounding |
| `/ground_single <phrase>` | Single-instance grounding |
| `/gui <element>` | GUI point grounding |
| `/gui_box <element>` | GUI box grounding |
| `/text` | OCR with text boxes |
| `/ground_text <text>` | Text grounding |
| `/layout title,table,figure` | Document layout grounding |
| `/point <target>` | Point localization |

### Model and quantization

<p align="center">
  <img src="assets/LocateAnything_pipeline.png" alt="LocateAnything inference pipeline" width="100%">
</p>

The inference path is `Image + Prompt -> preprocessing -> MoonViT -> Qwen2.5 decoder -> structured result parsing`.

| Item | Configuration |
| --- | --- |
| Vision | MoonViT, 27 blocks, `672 x 672`, signed W8 weights |
| Language | Qwen2.5 decoder, 36 layers, hidden size 2048, signed W8 weights |
| Activations | Dynamic quantization |
| Visual tokens | 576 |
| LM Head | W8, vocabulary size 152681 |
| Prefill / KV cache | 1024 / 4096 tokens |
| Decoding | PBD q=6, AR q=1, Host sampling |
| Target | Nash-P, four BPU cores, L2 `6:6:6:6` |

Model calibration and HBM compilation are maintained in [Locateanything_PTQ](https://github.com/D-Robotics/Locateanything_PTQ). Runtime files are published at [D-Robotics/LocateAnything-3B-BPU](https://huggingface.co/D-Robotics/LocateAnything-3B-BPU).

## Environment

| Item | Version |
| --- | --- |
| Hardware | D-Robotics RDK S600, AArch64 |
| OS | Ubuntu 24.04 LTS |
| TROS | Jazzy |
| Language | C++17 |
| Build | CMake, colcon |
| Dependencies | `rclcpp`, `sensor_msgs`, `std_msgs`, `hbm_img_msgs`, `ai_msgs`, `hobot_codec`, OpenCV, yaml-cpp |

## Usage

### 1. Build the package

```bash
git clone https://github.com/D-Robotics/hobot_locateanything.git
cd hobot_locateanything

source /opt/tros/jazzy/setup.bash
colcon build --merge-install --packages-select hobot_locateanything
source install/setup.bash
```

### 2. Download the model

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

The runtime reads these files:

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

### 3. Run inference

#### 1. Interactive Console

##### Start Console

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 run hobot_locateanything console --config config/config.yaml
```

Console output:

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
Loading Vision HBM...
Loading Language HBM...
HBM loaded  [============================] 16.7 s
Ready  S600/Nash-P  |  hybrid  |  max tokens 4096
```

##### Object Detection

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

<img src="assets/results/detection_multiclass.jpg" alt="Object detection" width="720">

##### GUI Grounding

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

<img src="assets/results/gui_rstudio.jpg" alt="GUI grounding" width="720">

##### Referring Expression Grounding

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

<img src="assets/results/referring_graduation.jpg" alt="Referring expression grounding" width="520">

##### OCR

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

##### Text Grounding

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

<img src="assets/results/ground_text_scrapbook.jpg" alt="Text grounding" width="720">

##### Layout Grounding

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

<img src="assets/results/layout_plot.jpg" alt="Layout grounding" width="720">

##### Point Localization

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

<img src="assets/results/point_succulent.jpg" alt="Point localization" width="512">

Image results are saved to `outputs/<image-name>/annotated.jpg` and `prediction.json`. Load a video with `/video` and use the same task commands:

```text
[User] <<< /video image/person_video.avi
[User] <<< /detect person
```

Video results are saved to:

```text
outputs/person_video/
├── annotated.mp4
├── predictions.jsonl
└── summary.json
```

#### 2. ROS 2 Node

Local image replay and USB camera input are independent workflows. The launch file starts the official image node, Codec, and LocateAnything inference node together. The inference node ignores images until it receives a valid prompt.

##### Local image replay

Terminal 1: start local image replay and the inference node, then wait for `ready`.

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=fb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
  publish_image_source:=image/07_detection_multiclass.jpg
```

Inference node output:

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 16.5 s
[INFO] [hobot_locateanything]: ready: input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything
[WARN] [hobot_locateanything]: waiting for prompt on /locateanything/prompt; image frames are ignored until a valid prompt arrives
```

Terminal 2: continuously subscribe to structured results.

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic echo \
  /perception/locateanything \
  ai_msgs/msg/PerceptionTargets
```

Terminal 3: publish a prompt.

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic pub --once \
  /locateanything/prompt \
  std_msgs/msg/String \
  "{data: '/detect person'}"
```

Prompt publisher output:

```text
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/detect person')
```

The launch file replays `image/07_detection_multiclass.jpg` at 2 FPS. Change `publish_image_source` to use another image.

Image publisher output:

```text
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
```

Inference output:

```text
[INFO] [hobot_locateanything]: prompt updated: /detect person
[INFO] [hobot_locateanything]: frame_id=2 prompt="/detect person" output="<ref>person</ref><box><125><356><248><766></box><box><720><400><862><769></box><|im_end|>" labels="person | person" boxes=2 points=0 fps=2 stop_reason=im_end prompt_tokens=615 generated_tokens=16 pbd_calls=4 pbd_accepted_tokens=16 mode=hybrid preprocess_ms=17.932 vision_ms=253.260 language_ms=341.119 postprocess_ms=0.014 total_ms=612.326
```

Result topic output:

```yaml
header:
  frame_id: '2'
fps: 2
perfs:
  - type: preprocess
    time_ms_duration: 17.932082
  - type: vision
    time_ms_duration: 253.259757
  - type: language
    time_ms_duration: 341.118596
  - type: postprocess
    time_ms_duration: 0.013951
targets:
  - type: person
    rois:
      - type: person
        rect: {x_offset: 80, y_offset: 148, height: 262, width: 79}
        confidence: -1.0
  - type: person
    rois:
      - type: person
        rect: {x_offset: 461, y_offset: 176, height: 236, width: 91}
        confidence: -1.0
```

To run another task on the same image, keep terminals 1 and 2 running and publish a new prompt from terminal 3:

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic pub --once \
  /locateanything/prompt \
  std_msgs/msg/String \
  "{data: '/detect bus'}"
```

Subsequent replays of the same image use `/detect bus`. The image publisher and result subscription do not need to restart.

Inference node output:

```text
[INFO] [hobot_locateanything]: prompt updated: /detect bus
```

##### USB camera input

Terminal 1: start the USB camera and inference node, then wait for `ready`.

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

Inference node output:

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 16.5 s
[INFO] [hobot_locateanything]: ready: input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything
[WARN] [hobot_locateanything]: waiting for prompt on /locateanything/prompt; image frames are ignored until a valid prompt arrives
```

Terminal 2: continuously subscribe to structured results.

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic echo \
  /perception/locateanything \
  ai_msgs/msg/PerceptionTargets
```

Terminal 3: publish a prompt.

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 topic pub --once \
  /locateanything/prompt \
  std_msgs/msg/String \
  "{data: '/ground cardboard box'}"
```

Prompt publisher output:

```text
Waiting for at least 1 matching subscription(s)...
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/ground cardboard box')
```

USB camera node output:

```text
[INFO] [hobot_usb_cam-1]: process started
[hobot_usb_cam]: framerate: 30
[hobot_usb_cam]: pixel_format_name: mjpeg
[hobot_usb_cam]: Camera calibration file: [/opt/tros/jazzy/lib/hobot_usb_cam/config/usb_camera_calibration.yaml] does not exist!
```

Inference output:

```text
[INFO] [hobot_locateanything]: prompt updated: /ground cardboard box
[INFO] [hobot_locateanything]: frame_id=13 prompt="/ground cardboard box" output="<ref>cardboard box</ref><box><503><613><556><655></box><|im_end|>" labels="cardboard box" boxes=1 points=0 fps=2 stop_reason=im_end prompt_tokens=616 generated_tokens=12 pbd_calls=3 pbd_accepted_tokens=12 mode=hybrid preprocess_ms=27.256 vision_ms=253.491 language_ms=304.637 postprocess_ms=0.011 total_ms=585.397
```

Result topic output:

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

While the camera is publishing, send a new prompt directly from terminal 3. Subsequent frames use the new prompt without restarting the camera node or result subscription.

The ROS node publishes results only. Rendering, encoding, and file storage belong to downstream TROS nodes.

## Performance

| Task | Output tokens | Vision (ms) | Prefill (ms) | Decode (ms) | Total (ms) | Decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Object detection | 47 | 252.5 | 149.9 | 525.0 | 970.5 | 89.5 |
| GUI grounding | 14 | 253.2 | 149.7 | 266.0 | 720.7 | 52.6 |
| Referring grounding | 14 | 246.0 | 152.3 | 164.5 | 603.6 | 85.1 |
| OCR | 66 | 245.5 | 152.4 | 665.3 | 1148.3 | 99.2 |
| Text grounding | 15 | 253.0 | 150.2 | 166.6 | 653.5 | 90.0 |
| Layout grounding | 43 | 245.4 | 151.8 | 448.1 | 904.7 | 96.0 |
| Point localization | 37 | 246.0 | 152.2 | 480.5 | 923.5 | 77.0 |
