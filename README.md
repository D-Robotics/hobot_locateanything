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

Model calibration and HBM compilation are maintained in [Locateanything_PTQ](https://github.com/D-Robotics/Locateanything_PTQ). Runtime files are published at [xkj521999/LocateAnything-3B-S600](https://huggingface.co/xkj521999/LocateAnything-3B-S600).

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
python3 -m pip install -U huggingface_hub
export HF_ENDPOINT="https://hf-mirror.com"
hf download xkj521999/LocateAnything-3B-S600 \
  --local-dir install/lib/hobot_locateanything/models
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

#### Interactive console

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

ros2 run hobot_locateanything console
```

Load an image, then enter a task command:

```text
[User] <<< /image image/07_detection_multiclass.jpg
[User] <<< /detect person,bus,bicycle
```

For video input:

```text
[User] <<< /video image/person_video.avi
[User] <<< /detect person
```

The Console uses the input file name as the output directory. Repeated runs overwrite the previous result for that file.

```text
outputs/07_detection_multiclass/
├── annotated.jpg
└── prediction.json

outputs/person_video/
├── annotated.mp4
├── predictions.jsonl
└── summary.json
```

The video writer falls back to `annotated.avi` when the MP4 codec is unavailable.

Example output on the RDK S600:

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

#### ROS 2 node

Local image replay and USB camera input are independent workflows. The launch file starts the official image node, Codec, and LocateAnything inference node together. The inference node ignores images until it receives a valid prompt.

##### Local image replay

Terminal 1: start local image replay and the inference node, then wait for `ready`.

```bash
cd hobot_locateanything
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=fb
ros2 launch hobot_locateanything hobot_locateanything.launch.py
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

A valid prompt remains active until another valid prompt replaces it or the node restarts.

The launch file replays the installed `07_detection_multiclass.jpg` at 2 FPS by default. Pass `publish_image_source:=/absolute/path/image.jpg` to use another image.

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

While the camera is publishing, send a new prompt directly from terminal 3. Subsequent frames use the new prompt without restarting the camera node or result subscription.

The ROS node publishes results only. Rendering, encoding, and file storage belong to downstream TROS nodes.

## Results

### Examples

Object detection, prompt: `/detect person,bus,bicycle`

<img src="assets/results/detection_multiclass.jpg" alt="Object detection" width="720">

GUI grounding, queries: `Go to file/function`, `Environment tab`, `Files tab`

<img src="assets/results/gui_rstudio.jpg" alt="GUI grounding" width="720">

Referring grounding, queries: `person wearing a graduation cap`, `woman in a black dress`, `clock tower`

<img src="assets/results/referring_graduation.jpg" alt="Referring grounding" width="520">

OCR, prompt: `/text`

<img src="assets/results/ocr_scrapbook.jpg" alt="OCR" width="720">

Text grounding, queries: `LIVE love LAUGH`, `laugh giggle be silly`, `Yes Virginia`

<img src="assets/results/ground_text_scrapbook.jpg" alt="Text grounding" width="720">

Layout grounding, prompt: `/layout plot,text`

<img src="assets/results/layout_plot.jpg" alt="Layout grounding" width="720">

Point localization, prompt: `/point succulent`

<img src="assets/results/point_succulent.jpg" alt="Point localization" width="512">

### Performance

| Task | Output tokens | Vision (ms) | Prefill (ms) | Decode (ms) | Total (ms) | Decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Object detection | 47 | 252.5 | 149.9 | 525.0 | 970.5 | 89.5 |
| GUI grounding | 14 | 253.2 | 149.7 | 266.0 | 720.7 | 52.6 |
| Referring grounding | 14 | 246.0 | 152.3 | 164.5 | 603.6 | 85.1 |
| OCR | 66 | 245.5 | 152.4 | 665.3 | 1148.3 | 99.2 |
| Text grounding | 15 | 253.0 | 150.2 | 166.6 | 653.5 | 90.0 |
| Layout grounding | 43 | 245.4 | 151.8 | 448.1 | 904.7 | 96.0 |
| Point localization | 37 | 246.0 | 152.2 | 480.5 | 923.5 | 77.0 |

### Resource Usage

#### Console

| Task | Avg BPU (%) | CPU (%) | Console RSS (MiB) | System DDR Read (GiB/s) | System DDR Write (GiB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Object detection | 30.4 | 40.2 | 167.1 | 74.2 | 15.6 |
| GUI grounding | 39.9 | 28.7 | 179.3 | 70.9 | 18.6 |
| Referring grounding | 34.5 | 26.6 | 175.6 | 70.8 | 23.6 |
| OCR | 41.3 | 49.0 | 185.3 | 65.9 | 12.4 |
| Text grounding | 29.5 | 34.7 | 182.4 | 65.3 | 20.1 |
| Layout grounding | 39.6 | 39.4 | 182.6 | 69.1 | 16.0 |
| Point localization | 43.5 | 34.7 | 180.4 | 76.6 | 15.1 |

#### ROS Real-Time Inference (2 FPS)

| Task | Avg BPU (%) | Inference node CPU (%) | Inference node RSS (MiB) | System DDR Read (GiB/s) | System DDR Write (GiB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Object detection | 65.8 | 42.1 | 201.6 | 76.0 | 15.6 |
| GUI grounding | 67.7 | 28.2 | 209.5 | 78.2 | 25.2 |
| Referring grounding | 67.8 | 27.6 | 210.6 | 75.9 | 25.8 |
| OCR | 62.4 | 50.4 | 208.8 | 71.6 | 13.0 |
| Text grounding | 67.0 | 32.7 | 184.3 | 68.2 | 22.9 |
| Layout grounding | 65.9 | 39.2 | 193.4 | 74.6 | 18.4 |
| Point localization | 66.2 | 34.4 | 203.2 | 78.2 | 18.2 |
