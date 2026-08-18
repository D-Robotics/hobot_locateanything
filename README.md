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

| Type | Description | Output |
| --- | --- | --- |
| Open-vocabulary object detection | Detects objects by user-provided category names without a fixed category list | Object categories and bounding boxes |
| Referring expression grounding | Locates objects from natural-language descriptions of appearance, attributes, position, or relationships | Object bounding boxes |
| GUI grounding | Locates buttons, icons, input fields, and other controls from text descriptions | Control points or bounding boxes |
| OCR | Recognizes text content and its position in an image | Recognized text and text bounding boxes |
| Text grounding | Locates user-specified text in an image | Specified text and bounding boxes |
| Document layout grounding | Locates titles, body text, tables, figures, and other document regions | Layout categories and bounding boxes |
| Point localization | Locates objects in general visual scenes from natural-language descriptions | Object point coordinates |

LocateAnything is designed primarily for visual detection and grounding tasks, whose Prompt formats are relatively fixed. We provide built-in task templates based on the Prompt formats used in the training data. Users only need to enter the query target through the corresponding command. `<query>` denotes a query target; separate multiple queries with commas. `<type>` denotes a document layout element type.

| Command | Example | Description |
| --- | --- | --- |
| `/detect <query>[,<query>...]` | `/detect person,bus,bicycle` | Detects all instances of the person, bus, and bicycle categories |
| `/ground <query>[,<query>...]` | `/ground person wearing a graduation cap,woman in a black dress,clock tower` | Locates all objects matching the three natural-language descriptions |
| `/ground_single <query>[,<query>...]` | `/ground_single person wearing a graduation cap` | Locates one object matching the natural-language description |
| `/gui <query>[,<query>...]` | `/gui Go to file/function` | Locates the specified GUI control and returns an interaction point |
| `/gui_box <query>[,<query>...]` | `/gui_box Go to file/function,Environment tab,Files tab` | Locates the three GUI controls and returns their bounding boxes |
| `/text` | `/text` | Recognizes all text in the image and its position |
| `/ground_text <query>[,<query>...]` | `/ground_text LIVE love LAUGH,laugh giggle be silly,Yes Virginia` | Locates the three specified text strings |
| `/layout <type>[,<type>...]` | `/layout plot,text` | Locates plot and text regions in a document |
| `/point <query>[,<query>...]` | `/point succulent,the succulent in the center` | Returns point coordinates for the two queries |

Model: [D-Robotics/LocateAnything-3B-BPU](https://huggingface.co/D-Robotics/LocateAnything-3B-BPU)

Calibration and HBM compilation: [D-Robotics/Locateanything_PTQ](https://github.com/D-Robotics/Locateanything_PTQ)

## Inference Performance

### 30 FPS ROS Local Replay

The fast_336 runtime uses a two-stage pipeline: the next frame's preprocessing and Vision stage overlap the current batch's Language stage. The static Batch 2 Language HBM processes up to two prepared frames together. Model weights, graph metadata, tokenizer state, graph I/O, and Language KV workspaces remain shared.

| Prompt | Samples | Result | Output FPS | Preprocess mean (ms) | Vision mean (ms) | Language mean (ms) | End-to-end record mean (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `/detect bus` | 300 | 300/300, 1 box, `im_end` | 11.616 | 8.630 | 61.791 | 171.923 | 296.697 |
| `/detect person,bus,bicycle` | 240 | 240/240, 5 boxes, `im_end` | 3.637 | 8.633 | 61.606 | 549.606 | 1052.182 |

Batch 2 Language time is shared by two independent frames, so Output FPS is throughput and is not calculated as `1000 / End-to-end record mean`.

| Prompt | Process CPU mean / peak | RSS mean / peak | Four-core BPU mean / peak | DDR Read mean | DDR Write mean | DDR Read+Write mean / peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `/detect bus` | 68.74% / 72.00% | 198.75 / 198.88 MiB | 85.73% / 97.00% | 71.83 GiB/s | 9.25 GiB/s | 81.07 / 87.44 GiB/s |
| `/detect person,bus,bicycle` | 84.33% / 91.00% | 202.69 / 202.69 MiB | 76.77% / 95.00% | 85.89 GiB/s | 3.68 GiB/s | 89.57 / 102.81 GiB/s |

The two resource windows exclude model loading. DDR values use the Read and Write `Bandwidth` fields from the same `hrut_ddr` sample. The observed `ion_uncache` heap was 302.06 MiB in both tests; it is a system-wide ION heap value, not process-private memory.

## Model and Quantization

<p align="center">
  <img src="assets/LocateAnything_pipeline.png" alt="LocateAnything inference pipeline" width="100%">
</p>

The inference path is `Image + Prompt -> preprocessing -> MoonViT -> Qwen2.5 decoder -> structured result parsing`.

| Item | Configuration |
| --- | --- |
| Vision | MoonViT, 27 blocks, `336 x 336`, signed W8 weights |
| Language | Qwen2.5 decoder, 36 layers, hidden size 2048, signed W8 weights |
| Activations | Dynamic quantization |
| Visual tokens | 144 |
| LM Head | W8, vocabulary size 152681 |
| Prefill / KV Cache | 256 / 1024 tokens |
| Decoding | PBD q=6, AR q=1, Host sampling |
| Target | Nash-P, four BPU cores, L2 `6:6:6:6` |

## Development Environment

| Item | Version |
| --- | --- |
| Hardware | D-Robotics RDK S600, AArch64 |
| OS | Ubuntu 24.04 LTS |
| TROS | Jazzy |
| Language | C++17 |
| Build tools | CMake, colcon |
| Dependencies | `rclcpp`, `sensor_msgs`, `std_msgs`, `hbm_img_msgs`, `ai_msgs`, `hobot_codec`, OpenCV, yaml-cpp |

## Preparation

The RDK S600 requires Ubuntu 24.04 and TogetheROS.Bot Jazzy.

### Build the Package

```bash
git clone -b fast_336 --single-branch \
  https://github.com/D-Robotics/hobot_locateanything.git
cd hobot_locateanything

source /opt/tros/jazzy/setup.bash
colcon build --merge-install --packages-select hobot_locateanything
source install/setup.bash
```

### Prepare the Model

The fast_336 HBM is not currently published as a download. Build it from the [PTQ `fast_336` branch](https://github.com/D-Robotics/Locateanything_PTQ/tree/fast_336) with `compiler/config/fast_336_batch2.yaml`, then copy the three generated files into the package's single model directory:

```bash
mkdir -p install/lib/hobot_locateanything/models
cp ../Locateanything_PTQ/compiler/outputs/fast_336_prefill256_cache1024_w8_batch2/build/vision/LocateAnything-3B_vision.hbm \
  install/lib/hobot_locateanything/models/LocateAnything-3B_vision_336x336.hbm
cp ../Locateanything_PTQ/compiler/outputs/fast_336_prefill256_cache1024_w8_batch2/build/language/LocateAnything-3B_language_batch2.hbm \
  install/lib/hobot_locateanything/models/LocateAnything-3B_language_336x336.hbm
cp ../Locateanything_PTQ/compiler/outputs/fast_336_prefill256_cache1024_w8_batch2/build/language/LocateAnything-3B_embed_tokens.bin \
  install/lib/hobot_locateanything/models/LocateAnything-3B_embed_tokens.bin
```

Runtime files:

```text
install/lib/hobot_locateanything/models/
├── LocateAnything-3B_vision_336x336.hbm
├── LocateAnything-3B_language_336x336.hbm
├── LocateAnything-3B_embed_tokens.bin
└── tokenizer/
    ├── vocab.json
    ├── merges.txt
    └── added_tokens.json
```

The branch has one runtime entry configuration:

```bash
ros2 run hobot_locateanything console --config config/config.yaml
```

## Basic Feature: Object Detection

### Console Inference

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
HBM loaded  [============================] 16.7 s
Ready  S600/Nash-P  |  hybrid  |  max tokens 768
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
  Vision   254.7 ms
  Prefill  151.6 ms  620 tokens
  Decode   526.3 ms  47 tokens  89.3 tokens/s
  Host     41.4 ms
  Total    978.5 ms
Result
  Labels bicycle, bus, person  |  Boxes 6  |  Points 0  |  Stop im_end
```

Results are saved to `outputs/07_detection_multiclass/annotated.jpg` and `prediction.json`.

<img src="assets/results/detection_multiclass.jpg" alt="Open-vocabulary object detection" width="720">

### ROS 2 Inference

Results are published on `/perception/locateanything`. Prompts are updated through `/locateanything/prompt`.

#### Local Image Replay

The default launch replays `image/07_detection_multiclass.jpg` at 2 FPS. Change `publish_image_source` to use another image.

##### Commands

Terminal 1, start image replay and the inference node:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=fb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
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

##### Outputs

Terminal 1, image publisher and inference node output:

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

Terminal 2, detection result output:

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
[INFO] [hobot_locateanything]: frame_id=44 prompt="/detect bus" output="<ref>bus</ref><box><124><263><595><657></box><|im_end|>" labels="bus" boxes=1 points=0 fps=1 stop_reason=im_end prompt_tokens=615 generated_tokens=10 pbd_calls=3 pbd_accepted_tokens=10 mode=hybrid preprocess_ms=43.837 vision_ms=245.829 language_ms=304.013 postprocess_ms=0.013 total_ms=593.692
```

Updated detection result:

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

After a new valid prompt is published, subsequent images use the new prompt without restarting the nodes. A frame already in inference may still produce one result for the previous prompt.

#### USB Camera

##### Commands

Terminal 1, start the USB camera and inference node:

```bash
source /opt/tros/jazzy/setup.bash
source install/setup.bash

export CAM_TYPE=usb
ros2 launch hobot_locateanything hobot_locateanything.launch.py \
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
  "{data: '/detect cardboard box,person'}"
```

##### Outputs

Terminal 1, USB camera and inference node output:

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

Terminal 2, detection result output:

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

Terminal 3, prompt publisher output:

```text
Waiting for at least 1 matching subscription(s)...
publisher: beginning loop
publishing #1: std_msgs.msg.String(data='/detect cardboard box,person')
```

While the camera is publishing, send a new prompt from terminal 3. Subsequent frames use the new prompt without restarting the nodes.

The ROS node publishes structured results. Downstream TROS nodes handle rendering and file storage.

## Advanced Features

### Console Inference

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
HBM loaded  [============================] 16.7 s
Ready  S600/Nash-P  |  hybrid  |  max tokens 768
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

Separate multiple queries with commas. Vision runs once per image or video frame, followed by each Language query and merged results.

### GUI Grounding

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
  Vision   252.9 ms
  Prefill  463.7 ms  1848 tokens
  Decode   519.8 ms  36 tokens  69.3 tokens/s
  Host     29.4 ms
  Total    1342.8 ms
Result
  Labels Environment tab, Files tab, Go to file/function  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/gui_rstudio.jpg" alt="GUI grounding" width="720">

### Referring Expression Grounding

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
  Vision   250.4 ms
  Prefill  462.5 ms  1854 tokens
  Decode   461.2 ms  39 tokens  84.6 tokens/s
  Host     29.9 ms
  Total    1268.8 ms
Result
  Labels clock tower, person wearing a graduation cap, woman in a black dress  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/referring_graduation.jpg" alt="Referring expression grounding" width="520">

### OCR

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

### Text Grounding

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
  Vision   246.0 ms
  Prefill  471.6 ms  1838 tokens
  Decode   459.4 ms  43 tokens  93.6 tokens/s
  Host     30.4 ms
  Total    1311.1 ms
Result
  Labels LIVE love LAUGH., Yes Virginia., laugh giggle be silly.  |  Boxes 3  |  Points 0  |  Stop im_end
```

<img src="assets/results/ground_text_scrapbook.jpg" alt="Text grounding" width="720">

### Layout Grounding

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
  Vision   245.6 ms
  Prefill  155.0 ms  620 tokens
  Decode   448.1 ms  43 tokens  96.0 tokens/s
  Host     37.2 ms
  Total    908.8 ms
Result
  Labels plot, text  |  Boxes 6  |  Points 0  |  Stop im_end
```

<img src="assets/results/layout_plot.jpg" alt="Layout grounding" width="720">

### Point Localization

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
  Vision   245.9 ms
  Prefill  310.5 ms  1220 tokens
  Decode   645.4 ms  50 tokens  77.5 tokens/s
  Host     47.4 ms
  Total    1272.7 ms
Result
  Labels succulent, the succulent in the center  |  Boxes 0  |  Points 9  |  Stop im_end
```

<img src="assets/results/point_succulent.jpg" alt="Point localization" width="512">

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
