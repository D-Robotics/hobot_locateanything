<div align="center">

<img src="assets/LocateAnything.jpg" alt="LocateAnything 在 D-Robotics S600 上的检测效果" width="820">

# LocateAnything-3B on D-Robotics S600

在 D-Robotics S600 上运行 LocateAnything-3B，支持开放词汇检测、指代表达定位、点定位、OCR 和版面定位。

</div>

## 项目布局

```text
hobot_locateanything/
├── compiler/                    # 校准、量化和 BC/HBO/HBM 编译
│   ├── config/                  # 编译配置
│   ├── datasets/                # 校准数据
│   ├── models/                  # Float 模型
│   └── outputs/                 # 编译产物和日志
├── inference/                   # S600 TROS C++ 推理包
│   ├── src/                     # 推理实现、运行器和 TROS 节点
│   ├── include/                 # C++ 推理模块接口
│   ├── config.yaml              # Console 和 ROS 共用参数
│   ├── image/                   # 测试图片
│   ├── models/                  # HBM、Embedding 和 Tokenizer
│   └── outputs/                 # 推理结果
└── README.md
```

编译和推理相互独立。推理端全部使用 C++，不依赖 Python 运行时。

## 默认配置

| 项目 | 配置 |
|---|---|
| 输入尺寸 | 672 x 672，保持宽高比并填充 |
| Vision | W8 |
| Language | W8 |
| LM Head | W8 |
| Prefill | 768 tokens |
| KV Cache | 4096 tokens |
| PBD | q=6 |
| Language 图 | fused decode，Prefill + PBD q6-q12 + AR q1-q5 |
| 最大生成长度 | 4096 tokens |

默认配置提供完整的 fused decode 流程。常用编译参数集中在
`compiler/config/quantization.yaml`，推理参数集中在
`inference/config.yaml`。运行时按 HBM 实际图接口执行，允许用户在同步修改编译图定义和 C++ 解码逻辑后扩展图集合。

## 直接部署 HBM

### 1. 获取代码和模型

```bash
git clone git@github.com:LiuAnclouds/hobot_locateanything.git
cd hobot_locateanything

mkdir -p inference/models
hf download <模型仓库> \
  --local-dir inference/models
```

模型目录应包含：

```text
inference/models/
├── LocateAnything-3B_vision.hbm
├── LocateAnything-3B_language.hbm
├── LocateAnything-3B_embed_tokens.bin
└── tokenizer/
```

### 2. 构建 TROS C++ 包

```bash
source /opt/tros/jazzy/setup.bash
colcon build --merge-install \
  --base-paths inference \
  --packages-select hobot_locateanything
source install/setup.bash
```

与 D-Robotics 官方 TROS 示例一致，`build/`、`install/` 和 `log/` 由 colcon
生成在仓库根目录，不进入 `inference/` 源码包。`install/` 是
`ros2 run` 使用的包索引和程序目录；`build/`、`log/` 可以在停止运行后重新生成。

`inference/config.yaml` 是唯一推理配置，模型只放在
`inference/models/`。配置和模型不复制到 `install/`。

### 3. 运行推理

用户入口固定为两个：`console` 用于本地图片和视频，
`hobot_locateanything` 用于订阅 TROS 图像话题。Vision 和 Language HBM
worker 是内部实现，不作为 `ros2 run` 入口。

本地图片和视频使用交互式 C++ Console：

```bash
cd inference
ros2 run hobot_locateanything console --config config.yaml
```

Console 默认读取当前目录的 `config.yaml`，也可以显式指定
另一份完整配置：

```bash
ros2 run hobot_locateanything console \
  --config /path/to/config.yaml
```

`max_new_tokens`、模型目录、输出目录、生成模式和 BPU 参数均在 YAML 中设置，
不再提供重复的命令行覆盖参数。

进入 Console 后先加载媒体，再输入任务：

```text
/image /path/to/image.jpg
/detect person,motorcycle

/video /path/to/video.mp4
/detect person
```

实时 TROS 推理直接运行节点，不依赖 launch XML。默认配置为共享内存话题
`/hbmem_img` 和任务 `/detect person`：

```bash
cd inference
ros2 run hobot_locateanything hobot_locateanything --ros-args \
  --params-file config.yaml
```

启动时覆盖输入话题和任务：

```bash
ros2 run hobot_locateanything hobot_locateanything --ros-args \
  --params-file config.yaml \
  -p input_topic:=/hbmem_img \
  -p default_prompt:="/detect person,motorcycle"
```

USB 和 MIPI 摄像头由 TROS 系统独立启动并发布 `/hbmem_img`。LA 不管理
摄像头进程，也不限制同一视频流被其他节点订阅。例如 USB 摄像头可以独立运行：

```bash
ros2 launch hobot_usb_cam hobot_usb_cam.launch.py \
  usb_video_device:=/dev/video0 usb_zero_copy:=True
```

MIPI 摄像头同样独立运行：

```bash
ros2 launch mipi_cam mipi_cam.launch.py \
  mipi_io_method:=shared_mem mipi_frame_ts_type:=realtime
```

默认任务为 `/detect person`。运行期间需要切换任务时，再从另一个终端发布：

```bash
ros2 topic pub --once /locateanything/prompt std_msgs/msg/String \
  "{data: '/detect person,motorcycle'}"
```

### 4. 输入与处理策略

LA 默认订阅共享内存 NV12 话题 `/hbmem_img`。话题名由参数 `input_topic` 指定，
共享内存或普通 ROS Image 传输由配置项 `is_shared_mem_sub` 决定。USB 和 MIPI
保持实时输入，模型忙碌时只保留最新一帧，不累积过期画面。

每帧创建独立的 Language 状态，不跨帧复用 KV Cache。

### 5. 获取结果

| 话题 | 消息类型 | 内容 |
|---|---|---|
| `/locateanything/result` | `std_msgs/msg/String` | 检测框、点坐标和各阶段耗时 JSON |
| `/locateanything/annotated` | `sensor_msgs/msg/Image` | 已绘制预测结果的图像 |

结果同时保存到 `inference/config.yaml` 中 `output_directory` 指定的目录：

```text
inference/outputs/
├── predictions.jsonl
└── frames/
    ├── frame_000001.jpg
    └── ...
```

## 从零校准和编译

从零编译需要 CUDA 主机、D-Robotics S600 OE_LLM SDK、LocateAnything 原始模型和校准数据。OE_LLM SDK 和 Python 环境独立安装在项目目录之外；仓库不复制 SDK 源码、wheel 或运行库。

### 1. 安装编译环境

```bash
mkdir -p ~/oellm/s600_sdk
cd ~/oellm
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz -C s600_sdk

source ~/miniforge3/etc/profile.d/conda.sh
conda create -n oellm_clean python=3.10 -y
conda activate oellm_clean
cd ~/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
python -m pip install -r oellm_build/requirements.txt
python -m pip install oellm_build/hbdk4_compiler-*.whl
```

这里的 `~/oellm/` 与 `hobot_locateanything/` 是两个独立目录。升级 OE_LLM
时，直接在项目外 wget、解压并安装新版 SDK，然后激活对应环境；不要把 SDK
目录复制到 `compiler/`，也不需要在量化 YAML 中配置 OE_LLM 路径。编译始终
使用当前激活环境的 Python，`compiler/build_adapter.py` 只包含
LocateAnything 的模型适配逻辑。

### 2. 准备模型和校准数据

```bash
cd /path/to/hobot_locateanything
mkdir -p compiler/models/LocateAnything-3B compiler/datasets/calibration/download
hf download nvidia/LocateAnything-3B \
  --local-dir compiler/models/LocateAnything-3B
hf download xkj521999/OE_LA_Calibration_data source.zip \
  --repo-type dataset --local-dir compiler/datasets/calibration/download
python -m zipfile -e compiler/datasets/calibration/download/source.zip \
  compiler/datasets/calibration/locateanything/source
```

也可以使用自有校准数据，并在 `compiler/config/quantization.yaml` 中修改数据路径。
默认只收集构建所需的量化 Scale 和校验信息；需要完整激活分布报告时，将
`calibration.detailed_statistics` 设为 `true`。

### 3. 校准并编译 HBM

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean
python -m pip install -r compiler/requirements-host.txt

CONFIG=compiler/config/quantization.yaml
python compiler/quantize.py --config "$CONFIG" prepare
python compiler/quantize.py --config "$CONFIG" calibrate --component all
python compiler/quantize.py --config "$CONFIG" build --component all --target hbm
```

每次正常编译前，将 `paths.output_dir` 的最后一级改为新的目录名；已有
`build/` 的目录只允许使用 `--resume` 续编。编译产物保存在该输出目录内。
将最终 Vision HBM、Language HBM 和 Embedding 放入 `inference/models/`，再按直接部署流程构建 TROS 包。

## 任务命令

```text
/detect cat,dog
/ground <phrase>
/ground_single <phrase>
/gui <element>
/gui_box <element>
/text
/ground_text <text>
/layout title,table,figure
/point <target>
```
