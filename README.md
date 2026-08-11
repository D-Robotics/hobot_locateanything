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

`hobot_locateanything` runs LocateAnything-3B on the D-Robotics RDK S600. It provides an interactive Console for local media and a ROS 2 node that subscribes to TROS images and prompts, then publishes `ai_msgs/msg/PerceptionTargets`. Both programs use the same C++ inference core.

## Introduction

### Supported tasks

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

The shared inference path is `Image + Prompt -> preprocessing -> MoonViT -> Qwen2.5 decoder -> structured result parsing`.

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

Model calibration and HBM compilation are maintained in [Locateanything_PTQ](https://github.com/LiuAnclouds/Locateanything_PTQ).

## Development environment

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
hf download LiuAnclouds/LocateAnything-3B-S600 --local-dir models
```

The runtime requires these files:

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
colcon build --packages-select hobot_locateanything
source install/setup.bash
```

### 3. Run TROS inference

#### Interactive console

```bash
cd "$HOME/tros_ws/src/hobot_locateanything"
source /opt/tros/jazzy/setup.bash
source "$HOME/tros_ws/install/setup.bash"

ros2 run hobot_locateanything console --config "$PWD/config.yaml"
```

Console output on the RDK S600:

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

#### ROS node

Shared-memory input from `/hbmem_img`:

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

Standard `sensor_msgs/msg/Image` input from `/image`:

```bash
ros2 run hobot_locateanything hobot_locateanything \
  --ros-args \
  --params-file "$PWD/config.yaml" \
  -p input_topic:=/image \
  -p is_shared_mem_sub:=false
```

Update the prompt while the node is running:

```bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  '{data: "/detect kite"}'
```

ROS node output on the RDK S600:

```text
[INFO] [hobot_locateanything]: ready: input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything
[INFO] [hobot_locateanything]: prompt updated: /detect kite
[INFO] [hobot_locateanything]: frame_id=24643 prompt="/detect kite" output="<ref>kite</ref><box><403><458><832><999></box><|im_end|>" labels="kite" boxes=1 points=0 fps=2 stop_reason=im_end prompt_tokens=615 generated_tokens=11 pbd_calls=3 pbd_accepted_tokens=11 mode=hybrid preprocess_ms=15.850 vision_ms=247.862 language_ms=303.215 postprocess_ms=0.013 total_ms=566.942
```

MIPI cameras, USB cameras, and local media publishers are provided by TROS. The inference node only subscribes to `/hbmem_img` or `/image`.

## Results

### Inference examples

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

Measurements were collected on an RDK S600 with the stable W8 HBM. Times are single-image latency, not camera input FPS.

| Task | Output tokens | Vision (ms) | Prefill (ms) | Decode (ms) | Total (ms) | Decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Object detection | 47 | 251.8 | 150.0 | 525.2 | 969.5 | 89.5 |
| GUI grounding | 14 | 253.2 | 149.7 | 266.0 | 720.7 | 52.6 |
| Referring grounding | 14 | 246.0 | 152.3 | 164.5 | 603.6 | 85.1 |
| OCR | 66 | 245.5 | 152.4 | 665.3 | 1148.3 | 99.2 |
| Text grounding | 15 | 253.0 | 150.2 | 166.6 | 653.5 | 90.0 |
| Layout grounding | 43 | 245.4 | 151.8 | 448.1 | 904.7 | 96.0 |
| Point localization | 37 | 246.0 | 152.2 | 480.5 | 923.5 | 77.0 |
