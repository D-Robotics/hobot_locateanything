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

Model calibration and HBM compilation are maintained in [Locateanything_PTQ](https://github.com/LiuAnclouds/Locateanything_PTQ). Runtime files are published at [xkj521999/LocateAnything-3B-S600](https://huggingface.co/xkj521999/LocateAnything-3B-S600).

## Environment

| Item | Version |
| --- | --- |
| Hardware | D-Robotics RDK S600, AArch64 |
| OS | Ubuntu 24.04 LTS |
| TROS | Jazzy |
| Language | C++17 |
| Build | CMake, colcon |
| Dependencies | `rclcpp`, `sensor_msgs`, `std_msgs`, `hbm_img_msgs`, `ai_msgs`, OpenCV, yaml-cpp |

## Usage

### 1. Download the model

```bash
mkdir -p "$HOME/tros_ws/src"
cd "$HOME/tros_ws/src"
git clone https://github.com/LiuAnclouds/hobot_locateanything.git
cd hobot_locateanything

python3 -m pip install -U huggingface_hub
export HF_ENDPOINT="https://hf-mirror.com"
hf download xkj521999/LocateAnything-3B-S600 --local-dir models
```

The runtime reads these files:

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

### 2. Build the package

```bash
source /opt/tros/jazzy/setup.bash
cd "$HOME/tros_ws"
colcon build --merge-install --packages-select hobot_locateanything
source install/setup.bash
```

### 3. Run inference

#### Interactive console

```bash
cd "$HOME/tros_ws/src/hobot_locateanything"
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 run hobot_locateanything console --config "$PWD/config.yaml"
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

Run terminals 1 through 4 in order. The inference node ignores images until it receives a valid prompt.

Terminal 1: start the inference node and wait for `ready`.

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

Terminal 2: wait for one structured result before publishing an image.

```bash
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 topic echo --once \
  /perception/locateanything \
  ai_msgs/msg/PerceptionTargets
```

Terminal 3: publish a prompt.

```bash
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 topic pub --once \
  /locateanything/prompt \
  std_msgs/msg/String \
  "{data: '/detect person,bus,bicycle'}"
```

A valid prompt remains active until another valid prompt replaces it or the node restarts.

Terminal 4: replay a local image with the official TROS image publisher.

```bash
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 launch hobot_image_publisher hobot_image_publisher.launch.py \
  publish_image_source:="$HOME/tros_ws/src/hobot_locateanything/image/07_detection_multiclass.jpg" \
  publish_image_format:=jpg \
  publish_message_topic_name:=/hbmem_img \
  publish_fps:=10 \
  publish_is_loop:=True \
  publish_is_shared_mem:=True \
  publish_encoding:=nv12
```

The image node publishes continuously. Press `Ctrl+C` in this terminal after receiving the result.

For USB camera input, remove `--once` from the terminal 2 command above and replace terminal 4 with:

```bash
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 launch hobot_usb_cam hobot_usb_cam.launch.py \
  usb_video_device:=/dev/video0 \
  usb_image_width:=1280 \
  usb_image_height:=720 \
  usb_framerate:=30 \
  usb_pixel_format:=mjpeg \
  usb_io_method:=mmap \
  usb_zero_copy:=True
```

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
