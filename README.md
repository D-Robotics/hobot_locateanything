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

## Model Overview

[LocateAnything](https://github.com/NVlabs/Eagle/tree/main/Embodied) is an open-semantic visual grounding model. It performs object detection, referring expression grounding, GUI and text grounding, document layout grounding, and point localization from text instructions. PBD (Parallel Box Decoding) generates bounding-box coordinates in parallel.

### Task Categories

| Type                             | Description                                                                                              | Output                                  |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Open-vocabulary object detection | Detects objects by user-provided category names without a fixed category list                            | Object categories and bounding boxes    |
| Referring expression grounding   | Locates objects from natural-language descriptions of appearance, attributes, position, or relationships | Object bounding boxes                   |
| GUI grounding                    | Locates buttons, icons, input fields, and other controls from text descriptions                          | Control points or bounding boxes        |
| OCR                              | Recognizes text content and its position in an image                                                     | Recognized text and text bounding boxes |
| Text grounding                   | Locates user-specified text in an image                                                                  | Specified text and bounding boxes       |
| Document layout grounding        | Locates titles, body text, tables, figures, and other document regions                                   | Layout categories and bounding boxes    |
| Point localization               | Locates objects in general visual scenes from natural-language descriptions                              | Object point coordinates                |

LocateAnything is designed primarily for visual detection and grounding tasks, whose Prompt formats are relatively fixed. We provide built-in task templates based on the Prompt formats used in the training data. Users only need to enter the query target through the corresponding command. `<query>` denotes a query target; separate multiple queries with commas. `<type>` denotes a document layout element type.

| Command                               | Example                                                                      | Description                                                          |
| ------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `/detect <query>[,<query>...]`        | `/detect person,bus,bicycle`                                                 | Detects all instances of the person, bus, and bicycle categories     |
| `/ground <query>[,<query>...]`        | `/ground person wearing a graduation cap,woman in a black dress,clock tower` | Locates all objects matching the three natural-language descriptions |
| `/ground_single <query>[,<query>...]` | `/ground_single person wearing a graduation cap`                             | Locates one object matching the natural-language description         |
| `/gui <query>[,<query>...]`           | `/gui Go to file/function`                                                   | Locates the specified GUI control and returns an interaction point   |
| `/gui_box <query>[,<query>...]`       | `/gui_box Go to file/function,Environment tab,Files tab`                     | Locates the three GUI controls and returns their bounding boxes      |
| `/text`                               | `/text`                                                                      | Recognizes all text in the image and its position                    |
| `/ground_text <query>[,<query>...]`   | `/ground_text LIVE love LAUGH,laugh giggle be silly,Yes Virginia`            | Locates the three specified text strings                             |
| `/layout <type>[,<type>...]`          | `/layout plot,text`                                                          | Locates plot and text regions in a document                          |
| `/point <query>[,<query>...]`         | `/point succulent,the succulent in the center`                               | Returns point coordinates for the two queries                        |

Model: [D-Robotics/LocateAnything-3B-BPU](https://huggingface.co/D-Robotics/LocateAnything-3B-BPU)

Calibration and HBM compilation: [D-Robotics/Locateanything_PTQ](https://github.com/D-Robotics/Locateanything_PTQ)

## Inference Performance

### Max (672)

| Platform | Task                | Output tokens | Vision (ms) | Prefill (ms) | Decode (ms) | Total (ms) | Decode (tokens/s) |
| -------- | ------------------- | -------------:| -----------:| ------------:| -----------:| ----------:| -----------------:|
| RDK S600 | Object detection    | 47            | 246.7       | 147.8        | 461.6       | 893.1      | 101.8             |
| RDK S600 | GUI grounding       | 36            | 246.9       | 440.0        | 449.4       | 1189.3     | 80.1              |
| RDK S600 | Referring grounding | 39            | 245.8       | 439.6        | 380.9       | 1091.2     | 102.4             |
| RDK S600 | OCR                 | 66            | 245.1       | 146.1        | 575.6       | 1030.8     | 114.7             |
| RDK S600 | Text grounding      | 43            | 245.9       | 438.7        | 386.2       | 1139.8     | 111.3             |
| RDK S600 | Layout grounding    | 43            | 245.7       | 146.2        | 386.2       | 816.1      | 111.3             |
| RDK S600 | Point localization  | 50            | 245.8       | 292.5        | 581.8       | 1145.6     | 85.9              |

### Balance (448)

| Platform | Task                | Output tokens | Vision (ms) | Prefill (ms) | Decode (ms) | Total (ms) | Decode (tokens/s) |
| -------- | ------------------- | -------------:| -----------:| ------------:| -----------:| ----------:| -----------------:|
| RDK S600 | Object detection    | 41            | 54.3        | 60.5         | 423.3       | 557.6      | 120.3             |
| RDK S600 | GUI grounding       | 36            | 51.4        | 176.6        | 301.2       | 554.7      | 119.5             |
| RDK S600 | Referring grounding | 39            | 51.3        | 176.5        | 248.3       | 490.9      | 157.1             |
| RDK S600 | OCR                 | 82            | 51.7        | 59.0         | 665.3       | 828.0      | 123.3             |
| RDK S600 | Text grounding      | 43            | 51.1        | 176.4        | 253.1       | 534.8      | 169.9             |
| RDK S600 | Layout grounding    | 43            | 51.4        | 58.9         | 335.6       | 473.9      | 128.1             |
| RDK S600 | Point localization  | 42            | 51.4        | 117.5        | 399.9       | 583.1      | 105.0             |

## Model and Quantization

<p align="center">
  <img src="assets/LocateAnything_pipeline.png" alt="LocateAnything inference pipeline" width="100%">
</p>

The inference path is `Image + Prompt -> preprocessing -> MoonViT -> Qwen2.5 decoder -> structured result parsing`.

### Max (672)

| Item               | Configuration                                                   |
| ------------------ | --------------------------------------------------------------- |
| Vision             | MoonViT, 27 blocks, `672 x 672`, signed W8 weights              |
| Language           | Qwen2.5 decoder, 36 layers, hidden size 2048, signed W8 weights |
| Activations        | Dynamic quantization                                            |
| Visual tokens      | 576                                                             |
| LM Head            | W8, vocabulary size 152681                                      |
| Prefill / KV Cache | 1024 / 4096 tokens                                              |
| Decoding           | PBD q=6, AR q=1, Host sampling                                  |
| Target             | Nash-P, four BPU cores, L2 `6:6:6:6`                            |

### Balance (448)

| Item               | Configuration                                                   |
| ------------------ | --------------------------------------------------------------- |
| Vision             | MoonViT, 27 blocks, `448 x 448`, signed W8 weights              |
| Language           | Qwen2.5 decoder, 36 layers, hidden size 2048, signed W8 weights |
| Activations        | Dynamic quantization                                            |
| Visual tokens      | 256                                                             |
| LM Head            | W8, vocabulary size 152681                                      |
| Prefill / KV Cache | 384 / 1024 tokens                                               |
| Decoding           | PBD q=6, AR q=1, Host sampling                                  |
| Target             | Nash-P, four BPU cores, L2 `6:6:6:6`                            |

## Development Environment

| Item         | Version                                                                                         |
| ------------ | ----------------------------------------------------------------------------------------------- |
| Hardware     | D-Robotics RDK S600, AArch64                                                                    |
| OS           | Ubuntu 24.04 LTS                                                                                |
| TROS         | Jazzy                                                                                           |
| Language     | C++17                                                                                           |
| Build tools  | CMake, colcon                                                                                   |
| Dependencies | `rclcpp`, `sensor_msgs`, `std_msgs`, `hbm_img_msgs`, `ai_msgs`, `hobot_codec`, OpenCV, yaml-cpp |

## Preparation

The RDK S600 requires Ubuntu 24.04 and TogetheROS.Bot Jazzy.

### Build the Package

```bash
git clone https://github.com/D-Robotics/hobot_locateanything.git
cd hobot_locateanything

source /opt/tros/jazzy/setup.bash
colcon build --merge-install --packages-select hobot_locateanything
source install/setup.bash
```

### Download the Model

We provide two compiled inference models: **Max** (high accuracy) and **Balance** (high performance). Max focuses on recognition accuracy optimization, while Balance focuses on inference performance optimization. Download the corresponding models from the links below.

#### Max (672)

```bash
mkdir -p install/lib/hobot_locateanything/models
wget -c -P install/lib/hobot_locateanything/models \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_vision.hbm
wget -c -P install/lib/hobot_locateanything/models \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_language.hbm
wget -c -P install/lib/hobot_locateanything/models \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU/resolve/main/LocateAnything-3B_embed_tokens.bin
```

#### Balance (448)

```bash
mkdir -p install/lib/hobot_locateanything/models
wget -c -O install/lib/hobot_locateanything/models/LocateAnything-3B_vision_balance.hbm \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU-Balance/resolve/main/LocateAnything-3B_vision.hbm
wget -c -O install/lib/hobot_locateanything/models/LocateAnything-3B_language_balance.hbm \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU-Balance/resolve/main/LocateAnything-3B_language.hbm
wget -c -O install/lib/hobot_locateanything/models/LocateAnything-3B_embed_tokens.bin \
  https://hf-mirror.com/D-Robotics/LocateAnything-3B-BPU-Balance/resolve/main/LocateAnything-3B_embed_tokens.bin
```

## Basic Feature: Object Detection

### Console Inference

#### Max (672)

```bash
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
HBM loaded  [============================] 16.6 s
Ready  S600/Nash-P  |  672x672  |  hybrid  |  max tokens 4096
Tasks
  /detect cat,dog               Object detection
  /ground <query>[,<query>...]  Referring expression grounding (multi-query)
  /ground_single <query>[,...]  Referring expression grounding (single target)
  /gui <query>[,<query>...]     GUI point grounding
  /gui_box <query>[,<query>...] GUI box grounding
  /text                         Text OCR
  /ground_text <query>[,...]    Text grounding
  /layout title,table,figure    Document layout analysis
  /point <query>[,<query>...]   Point grounding
Session
  /image <image_path>           Load an image
  /video <video_path>           Process all video frames
  regen                         Re-run the previous request
  reset                         Clear the current media
  exit                          Exit the application
```

Load an image:

```text
/image image/07_detection_multiclass.jpg
```

Image loading output:

```text
Image loaded  image/07_detection_multiclass.jpg
```

Enter a detection command:

```text
/detect person,bus,bicycle
```

Inference output:

```text
[Assistant] >>> /detect person,bus,bicycle
Performance
  Vision   246.7 ms
  Prefill  147.8 ms  620 tokens
  Decode   461.6 ms  47 tokens  101.8 tokens/s
  Host     30.4 ms
  Total    893.1 ms
Result
  Labels bicycle, bus, person  |  Boxes 6  |  Points 0  |  Stop im_end
```

Results are saved to `outputs/07_detection_multiclass/annotated.jpg` and `prediction.json`.

<img src="assets/results/detection_multiclass_max.jpg" alt="Open-vocabulary object detection with Max (672)" width="720">

#### Balance (448)

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 run hobot_locateanything console --config config/config_balance.yaml
```

Console output:

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
Loading Vision HBM...
Loading Language HBM...
HBM loaded  [============================] 12.1 s
Ready  S600/Nash-P  |  448x448  |  hybrid  |  max tokens 640
Tasks
  /detect cat,dog               Object detection
  /ground <query>[,<query>...]  Referring expression grounding (multi-query)
  /ground_single <query>[,...]  Referring expression grounding (single target)
  /gui <query>[,<query>...]     GUI point grounding
  /gui_box <query>[,<query>...] GUI box grounding
  /text                         Text OCR
  /ground_text <query>[,...]    Text grounding
  /layout title,table,figure    Document layout analysis
  /point <query>[,<query>...]   Point grounding
Session
  /image <image_path>           Load an image
  /video <video_path>           Process all video frames
  regen                         Re-run the previous request
  reset                         Clear the current media
  exit                          Exit the application
```

Load an image:

```text
/image image/07_detection_multiclass.jpg
```

Image loading output:

```text
Image loaded  image/07_detection_multiclass.jpg
```

Enter a detection command:

```text
/detect person,bus,bicycle
```

Inference output:

```text
[Assistant] >>> /detect person,bus,bicycle
Performance
  Vision   54.3 ms
  Prefill  60.5 ms  300 tokens
  Decode   423.3 ms  41 tokens  96.9 tokens/s
  Host     26.5 ms
  Total    557.6 ms
Result
  Labels bicycle, bus, person  |  Boxes 5  |  Points 0  |  Stop im_end
```

Results are saved to `outputs/07_detection_multiclass/annotated.jpg` and `prediction.json`.

<img src="assets/results/detection_multiclass_balance.jpg" alt="Open-vocabulary object detection with Balance (448)" width="720">

### ROS 2 Inference

Results are published on `/perception/locateanything`. Prompts are updated through `/locateanything/prompt`.

#### Local Image Replay

The default launch replays `image/07_detection_multiclass.jpg` at 2 FPS. Change `publish_image_source` to use another image.

##### Max (672)

###### Commands

Terminal 1, start image replay and the inference node:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=fb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
  config_file:=config/config.yaml \
  publish_image_source:=image/07_detection_multiclass.jpg
```

Terminal 2, subscribe to detection results:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic echo /perception/locateanything ai_msgs/msg/PerceptionTargets
```

Terminal 3, publish a detection prompt:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect person,bus,bicycle'}"
```

###### Outputs

Terminal 1, image publisher and inference node output:

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 16.6 s
[INFO] [hobot_locateanything]: ready: image=672x672 input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything pipelined=true
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
[INFO] [hobot_locateanything]: Inference
  Input       frame_id=152 prompt="/detect person,bus,bicycle"
  Prediction  labels="person | person | bus | bicycle | bicycle" boxes=5 points=0
  Language    mode=hybrid prompt_tokens=620 generated_tokens=41 stop_reason=im_end
  PBD         calls=9 accepted_tokens=41
  Throughput  fps=1 total_ms=1022.971
  Timing      preprocess_ms=45.153 vision_ms=246.513 language_ms=731.251 postprocess_ms=0.023
```

Terminal 2, detection result output:

```yaml
header:
  frame_id: '152'
fps: 1
perfs:
  - type: preprocess
    time_ms_duration: 45.153442
  - type: vision
    time_ms_duration: 246.513345
  - type: language
    time_ms_duration: 731.251405
  - type: postprocess
    time_ms_duration: 0.023075
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

The image publisher supplies input at 2 FPS. The result topic's `fps: 1` is the measured inference result rate for this run.

Terminal 3, prompt publisher output:

```text
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/detect person,bus,bicycle')
```

Terminal 3, update the detection prompt:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect bus'}"
```

Inference output after the prompt update:

```text
[INFO] [hobot_locateanything]: prompt updated: /detect bus
[INFO] [hobot_locateanything]: Inference
  Input       frame_id=229 prompt="/detect bus"
  Prediction  labels="bus" boxes=1 points=0
  Language    mode=hybrid prompt_tokens=615 generated_tokens=10 stop_reason=im_end
  PBD         calls=3 accepted_tokens=10
  Throughput  fps=2 total_ms=580.160
  Timing      preprocess_ms=44.434 vision_ms=246.067 language_ms=258.649 postprocess_ms=0.035
```

Updated detection result:

```yaml
header:
  frame_id: '229'
fps: 2
targets:
  - type: bus
    rois:
      - type: bus
        rect: {x_offset: 238, y_offset: 85, height: 756, width: 904}
        confidence: -1.0
```

##### Balance (448)

###### Commands

Terminal 1, start image replay and the inference node:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=fb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
  config_file:=config/config_balance.yaml \
  publish_image_source:=image/07_detection_multiclass.jpg
```

Terminal 2, subscribe to detection results:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic echo /perception/locateanything ai_msgs/msg/PerceptionTargets
```

Terminal 3, publish a detection prompt:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect person,bus,bicycle'}"
```

###### Outputs

Terminal 1, image publisher and inference node output:

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 12.7 s
[INFO] [hobot_locateanything]: ready: image=448x448 input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything pipelined=true
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
[INFO] [hobot_locateanything]: Inference
  Input       frame_id=92 prompt="/detect person,bus,bicycle"
  Prediction  labels="person | person | bus | bicycle | bicycle" boxes=5 points=0
  Language    mode=hybrid prompt_tokens=300 generated_tokens=41 stop_reason=im_end
  PBD         calls=9 accepted_tokens=41
  Throughput  fps=2 total_ms=420.363
  Timing      preprocess_ms=38.225 vision_ms=54.308 language_ms=327.780 postprocess_ms=0.021
```

Terminal 2, detection result output:

```yaml
header:
  frame_id: '92'
fps: 2
perfs:
  - type: preprocess
    time_ms_duration: 38.225197
  - type: vision
    time_ms_duration: 54.307770
  - type: language
    time_ms_duration: 327.779695
  - type: postprocess
    time_ms_duration: 0.021100
targets:
  - type: person
    rois:
      - type: person
        rect: {x_offset: 420, y_offset: 331, height: 576, width: 191}
        confidence: -1.0
  - type: person
    rois:
      - type: person
        rect: {x_offset: 1283, y_offset: 394, height: 542, width: 176}
        confidence: -1.0
  - type: bus
    rois:
      - type: bus
        rect: {x_offset: 240, y_offset: 77, height: 763, width: 1004}
        confidence: -1.0
  - type: bicycle
    rois:
      - type: bicycle
        rect: {x_offset: 991, y_offset: 490, height: 307, width: 251}
        confidence: -1.0
  - type: bicycle
    rois:
      - type: bicycle
        rect: {x_offset: 1409, y_offset: 690, height: 390, width: 271}
        confidence: -1.0
```

The image publisher supplies input at 2 FPS. The result topic's `fps: 2` is the measured inference result rate for this run.

After a new valid prompt is published, subsequent images use the new prompt without restarting the nodes. A frame already in inference may still produce one result for the previous prompt.

#### USB Camera

##### Max (672)

###### Commands

Terminal 1, start the USB camera and inference node:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=usb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
  config_file:=config/config.yaml \
  device:=/dev/video0 \
  locateanything_image_width:=1280 \
  locateanything_image_height:=720
```

Terminal 2, subscribe to detection results:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic echo /perception/locateanything ai_msgs/msg/PerceptionTargets
```

Terminal 3, publish a detection prompt:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect bottle'}"
```

###### Outputs

Terminal 1, USB camera and inference node output:

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 16.2 s
[INFO] [hobot_locateanything]: ready: image=672x672 input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything pipelined=true
[WARN] [hobot_locateanything]: waiting for prompt on /locateanything/prompt; image frames are ignored until a valid prompt arrives
[INFO] [hobot_usb_cam-1]: process started
[hobot_usb_cam]: framerate: 30
[hobot_usb_cam]: pixel_format_name: mjpeg
[INFO] [hobot_locateanything]: prompt updated: /detect bottle
[INFO] [hobot_locateanything]: Inference
  Input       frame_id=384 prompt="/detect bottle"
  Prediction  labels="" boxes=0 points=0
  Language    mode=hybrid prompt_tokens=615 generated_tokens=8 stop_reason=im_end
  PBD         calls=3 accepted_tokens=8
  Throughput  fps=2 total_ms=967.905
  Timing      preprocess_ms=26.812 vision_ms=358.622 language_ms=483.846 postprocess_ms=0.011
```

Terminal 2, detection result output:

```yaml
header:
  frame_id: '384'
fps: 2
perfs:
  - type: preprocess
    time_ms_duration: 26.811845
  - type: vision
    time_ms_duration: 358.622130
  - type: language
    time_ms_duration: 483.845598
  - type: postprocess
    time_ms_duration: 0.010776
targets: []
```

##### Balance (448)

###### Commands

Terminal 1, start the USB camera and inference node:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=usb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
  config_file:=config/config_balance.yaml \
  device:=/dev/video0 \
  locateanything_image_width:=1280 \
  locateanything_image_height:=720
```

Terminal 2, subscribe to detection results:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic echo /perception/locateanything ai_msgs/msg/PerceptionTargets
```

Terminal 3, publish a detection prompt:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect bottle'}"
```

###### Outputs

Terminal 1, USB camera and inference node output:

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
[INFO] [hobot_locateanything]: loading Vision HBM
[INFO] [hobot_locateanything]: loading Language HBM
[INFO] [hobot_locateanything]: inference core ready in 12.4 s
[INFO] [hobot_locateanything]: ready: image=448x448 input=/hbmem_img transport=hbmem prompt_topic=/locateanything/prompt result=/perception/locateanything pipelined=true
[WARN] [hobot_locateanything]: waiting for prompt on /locateanything/prompt; image frames are ignored until a valid prompt arrives
[INFO] [hobot_usb_cam-1]: process started
[hobot_usb_cam]: framerate: 30
[hobot_usb_cam]: pixel_format_name: mjpeg
[INFO] [hobot_locateanything]: prompt updated: /detect bottle
[INFO] [hobot_locateanything]: Inference
  Input       frame_id=332 prompt="/detect bottle"
  Prediction  labels="" boxes=0 points=0
  Language    mode=hybrid prompt_tokens=295 generated_tokens=8 stop_reason=im_end
  PBD         calls=3 accepted_tokens=8
  Throughput  fps=6 total_ms=323.855
  Timing      preprocess_ms=19.682 vision_ms=85.117 language_ms=161.977 postprocess_ms=0.012
```

Terminal 2, detection result output:

```yaml
header:
  frame_id: '332'
fps: 6
perfs:
  - type: preprocess
    time_ms_duration: 19.682163
  - type: vision
    time_ms_duration: 85.116727
  - type: language
    time_ms_duration: 161.977154
  - type: postprocess
    time_ms_duration: 0.011500
targets: []
```

Terminal 3, prompt publisher output for both modes:

```text
Waiting for at least 1 matching subscription(s)...
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/detect bottle')
```

While the camera is publishing, send a new prompt from terminal 3. Subsequent frames use the new prompt without restarting the nodes.

The ROS node publishes structured results. Downstream TROS nodes handle rendering and file storage.

## Advanced Features

### Console Inference

#### Max (672)

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 run hobot_locateanything console --config config/config.yaml
```

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
Loading Vision HBM...
Loading Language HBM...
HBM loaded  [============================] 16.7 s
Ready  S600/Nash-P  |  672x672  |  hybrid  |  max tokens 4096
Tasks
  /detect cat,dog               Object detection
  /ground <query>[,<query>...]  Referring expression grounding (multi-query)
  /ground_single <query>[,...]  Referring expression grounding (single target)
  /gui <query>[,<query>...]     GUI point grounding
  /gui_box <query>[,<query>...] GUI box grounding
  /text                         Text OCR
  /ground_text <query>[,...]    Text grounding
  /layout title,table,figure    Document layout analysis
  /point <query>[,<query>...]   Point grounding
Session
  /image <image_path>           Load an image
  /video <video_path>           Process all video frames
  regen                         Re-run the previous request
  reset                         Clear the current media
  exit                          Exit the application
```

#### Balance (448)

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash
ros2 run hobot_locateanything console --config config/config_balance.yaml
```

```text
[UCP]: UCP version = 3.12.3
[DNN]: 3.12.3_(4.5.4 HBRT)
Loading Vision HBM...
Loading Language HBM...
HBM loaded  [============================] 12.3 s
Ready  S600/Nash-P  |  448x448  |  hybrid  |  max tokens 640
Tasks
  /detect cat,dog               Object detection
  /ground <query>[,<query>...]  Referring expression grounding (multi-query)
  /ground_single <query>[,...]  Referring expression grounding (single target)
  /gui <query>[,<query>...]     GUI point grounding
  /gui_box <query>[,<query>...] GUI box grounding
  /text                         Text OCR
  /ground_text <query>[,...]    Text grounding
  /layout title,table,figure    Document layout analysis
  /point <query>[,<query>...]   Point grounding
Session
  /image <image_path>           Load an image
  /video <video_path>           Process all video frames
  regen                         Re-run the previous request
  reset                         Clear the current media
  exit                          Exit the application
```

Separate multiple queries with commas. Vision runs once per image or video frame, followed by each Language query and merged results.

### GUI Grounding

#### Max (672)

Load an image:

```text
/image image/02_gui_rstudio.jpg
```

Image loading output:

```text
Image loaded  image/02_gui_rstudio.jpg
```

Enter a grounding command:

```text
/gui_box Go to file/function,Environment tab,Files tab
```

Inference output:

```text
[Assistant] >>> /gui_box Go to file/function,Environment tab,Files tab
Performance
  Vision   246.9 ms
  Prefill  440.0 ms  1848 tokens
  Decode   449.4 ms  36 tokens  80.1 tokens/s
  Host     22.5 ms
  Total    1189.3 ms
Result
  Labels Environment tab, Files tab, Go to file/function  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/gui_rstudio_max.jpg" alt="GUI grounding with Max (672)" width="720">

#### Balance (448)

Load an image:

```text
/image image/02_gui_rstudio.jpg
```

Image loading output:

```text
Image loaded  image/02_gui_rstudio.jpg
```

Enter a grounding command:

```text
/gui_box Go to file/function,Environment tab,Files tab
```

Inference output:

```text
[Assistant] >>> /gui_box Go to file/function,Environment tab,Files tab
Performance
  Vision   51.4 ms
  Prefill  176.6 ms  888 tokens
  Decode   301.2 ms  36 tokens  119.5 tokens/s
  Host     20.8 ms
  Total    554.7 ms
Result
  Labels Environment tab, Files tab, Go to file/function  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/gui_rstudio_balance.jpg" alt="GUI grounding with Balance (448)" width="720">

### Referring Expression Grounding

#### Max (672)

Load an image:

```text
/image image/03_referring_graduation.jpg
```

Image loading output:

```text
Image loaded  image/03_referring_graduation.jpg
```

Enter a grounding command:

```text
/ground person wearing a graduation cap,woman in a black dress,clock tower
```

Inference output:

```text
[Assistant] >>> /ground person wearing a graduation cap,woman in a black dress,clock tower
Performance
  Vision   245.8 ms
  Prefill  439.6 ms  1854 tokens
  Decode   380.9 ms  39 tokens  102.4 tokens/s
  Host     21.4 ms
  Total    1091.2 ms
Result
  Labels clock tower, person wearing a graduation cap, woman in a black dress  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/referring_graduation_max.jpg" alt="Referring expression grounding with Max (672)" width="520">

#### Balance (448)

Load an image:

```text
/image image/03_referring_graduation.jpg
```

Image loading output:

```text
Image loaded  image/03_referring_graduation.jpg
```

Enter a grounding command:

```text
/ground person wearing a graduation cap,woman in a black dress,clock tower
```

Inference output:

```text
[Assistant] >>> /ground person wearing a graduation cap,woman in a black dress,clock tower
Performance
  Vision   51.3 ms
  Prefill  176.5 ms  894 tokens
  Decode   248.3 ms  39 tokens  157.1 tokens/s
  Host     21.6 ms
  Total    490.9 ms
Result
  Labels clock tower, person wearing a graduation cap, woman in a black dress  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/referring_graduation_balance.jpg" alt="Referring expression grounding with Balance (448)" width="520">

### OCR

#### Max (672)

Load an image:

```text
/image image/04_ocr_scrapbook.jpg
```

Image loading output:

```text
Image loaded  image/04_ocr_scrapbook.jpg
```

Enter the OCR command:

```text
/text
```

Inference output:

```text
[Assistant] >>> /text
Performance
  Vision   245.1 ms
  Prefill  146.1 ms  610 tokens
  Decode   575.6 ms  66 tokens  114.7 tokens/s
  Host     49.0 ms
  Total    1030.8 ms
Result
  Labels LIVE love LAUGH, Yes, Virginiaina, [to-day]], laugh giggle be silly  |  Boxes 5  |  Points 0  |  Stop im_end
```

<img src="assets/results/ocr_scrapbook_max.jpg" alt="OCR with Max (672)" width="720">

#### Balance (448)

Load an image:

```text
/image image/04_ocr_scrapbook.jpg
```

Image loading output:

```text
Image loaded  image/04_ocr_scrapbook.jpg
```

Enter the OCR command:

```text
/text
```

Inference output:

```text
[Assistant] >>> /text
Performance
  Vision   51.7 ms
  Prefill  59.0 ms  290 tokens
  Decode   665.3 ms  82 tokens  123.3 tokens/s
  Host     71.6 ms
  Total    828.0 ms
Result
  Labels LAUGH, LIVE, V's Virginia., [to-day]], laugh ggle be silly, love  |  Boxes 7  |  Points 0  |  Stop im_end
```

<img src="assets/results/ocr_scrapbook_balance.jpg" alt="OCR with Balance (448)" width="720">

### Text Grounding

#### Max (672)

Load an image:

```text
/image image/04_ocr_scrapbook.jpg
```

Image loading output:

```text
Image loaded  image/04_ocr_scrapbook.jpg
```

Enter a grounding command:

```text
/ground_text LIVE love LAUGH,laugh giggle be silly,Yes Virginia
```

Inference output:

```text
[Assistant] >>> /ground_text LIVE love LAUGH,laugh giggle be silly,Yes Virginia
Performance
  Vision   245.9 ms
  Prefill  438.7 ms  1838 tokens
  Decode   386.2 ms  43 tokens  111.3 tokens/s
  Host     23.0 ms
  Total    1139.8 ms
Result
  Labels LIVE love LAUGH., Yes Virginia., laugh giggle be silly.  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/ground_text_scrapbook_max.jpg" alt="Text grounding with Max (672)" width="720">

#### Balance (448)

Load an image:

```text
/image image/04_ocr_scrapbook.jpg
```

Image loading output:

```text
Image loaded  image/04_ocr_scrapbook.jpg
```

Enter a grounding command:

```text
/ground_text LIVE love LAUGH,laugh giggle be silly,Yes Virginia
```

Inference output:

```text
[Assistant] >>> /ground_text LIVE love LAUGH,laugh giggle be silly,Yes Virginia
Performance
  Vision   51.1 ms
  Prefill  176.4 ms  878 tokens
  Decode   253.1 ms  43 tokens  169.9 tokens/s
  Host     23.3 ms
  Total    534.8 ms
Result
  Labels LIVE love LAUGH., Yes Virginia., laugh giggle be silly.  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/ground_text_scrapbook_balance.jpg" alt="Text grounding with Balance (448)" width="720">

### Layout Grounding

#### Max (672)

Load an image:

```text
/image image/05_layout_plot.jpg
```

Image loading output:

```text
Image loaded  image/05_layout_plot.jpg
```

Enter a layout command:

```text
/layout plot,text
```

Inference output:

```text
[Assistant] >>> /layout plot,text
Performance
  Vision   245.7 ms
  Prefill  146.2 ms  620 tokens
  Decode   386.2 ms  43 tokens  111.3 tokens/s
  Host     28.9 ms
  Total    816.1 ms
Result
  Labels plot, text  |  Boxes 6  |  Points 0  |  Stop im_end
```

<img src="assets/results/layout_plot_max.jpg" alt="Layout grounding with Max (672)" width="720">

#### Balance (448)

Load an image:

```text
/image image/05_layout_plot.jpg
```

Image loading output:

```text
Image loaded  image/05_layout_plot.jpg
```

Enter a layout command:

```text
/layout plot,text
```

Inference output:

```text
[Assistant] >>> /layout plot,text
Performance
  Vision   51.4 ms
  Prefill  58.9 ms  300 tokens
  Decode   335.6 ms  43 tokens  128.1 tokens/s
  Host     27.8 ms
  Total    473.9 ms
Result
  Labels plot, text  |  Boxes 6  |  Points 0  |  Stop im_end
```

<img src="assets/results/layout_plot_balance.jpg" alt="Layout grounding with Balance (448)" width="720">

### Point Localization

#### Max (672)

Load an image:

```text
/image image/06_pointing_succulent.jpg
```

Image loading output:

```text
Image loaded  image/06_pointing_succulent.jpg
```

Enter a point localization command:

```text
/point succulent,the succulent in the center
```

Inference output:

```text
[Assistant] >>> /point succulent,the succulent in the center
Performance
  Vision   245.8 ms
  Prefill  292.5 ms  1220 tokens
  Decode   581.8 ms  50 tokens  85.9 tokens/s
  Host     39.0 ms
  Total    1145.6 ms
Result
  Labels succulent, the succulent in the center  |  Boxes 0  |  Points 9  |  Stop im_end
```

<img src="assets/results/point_succulent_max.jpg" alt="Point localization with Max (672)" width="512">

#### Balance (448)

Load an image:

```text
/image image/06_pointing_succulent.jpg
```

Image loading output:

```text
Image loaded  image/06_pointing_succulent.jpg
```

Enter a point localization command:

```text
/point succulent,the succulent in the center
```

Inference output:

```text
[Assistant] >>> /point succulent,the succulent in the center
Performance
  Vision   51.4 ms
  Prefill  117.5 ms  580 tokens
  Decode   399.9 ms  42 tokens  105.0 tokens/s
  Host     34.4 ms
  Total    583.1 ms
Result
  Labels succulent, the succulent in the center  |  Boxes 0  |  Points 7  |  Stop im_end
```

<img src="assets/results/point_succulent_balance.jpg" alt="Point localization with Balance (448)" width="512">

## Image and Video Outputs

Image results are saved to `outputs/<image-name>/annotated.jpg` and `prediction.json`.

Load a video with `/video` and use the same task commands:

```text
/video image/person_video.avi
/detect person
```

Video results are saved to:

```text
outputs/person_video/
├── annotated.mp4
├── predictions.jsonl
└── summary.json
```
