# 基于 OELLM 的 LLM/VLM 量化流程

本文说明如何将 LLM/VLM 的 checkpoint 转换为可在目标设备运行的 OELLM HBM 模型。流程从原始权重、模型配置和 tokenizer/processor 开始，依次完成浮点模型适配、量化参数生成、BC 计算图导出、目标代码编译和板端运行。

本文只讨论实施顺序和各阶段应完成的工作。W4/W8、Static/Dynamic A8、正交旋转、Value 中心化及混合精度等具体量化方法，见[《LLM-VLM 量化技巧整理》](./LLM_VLM_QUANTIZATION_TRICKS.zh-CN.md)。

```text
checkpoint + config + tokenizer/processor
  -> 1. 原始浮点模型
       -> 检查模型文件与权重结构
       -> 复现原生浮点推理
       -> 确定静态部署规格
       -> 完成 OELLM 浮点适配
       -> 保存参考输入与浮点结果

  -> 2. 量化仿真模型
       -> 制定量化配置
       -> 准备校准输入
       -> 收集统计并冻结量化参数
       -> 运行 PyTorch 量化仿真
       -> 确认量化精度

  -> 3. BC 计算图
       -> 定义 TensorType 和图接口
       -> 导出前端 BC
       -> 转换目标 BC
       -> 检查图结构与数值结果

  -> 4. HBM 部署模型
       -> 配置目标编译参数
       -> 编译 HBO
       -> 链接 HBM
       -> 接入 HBRT 运行时
       -> 完成板端精度与性能验证
```

四个阶段的输入和输出如下。

| 阶段 | 输入 | 本阶段输出 |
|---|---|---|
| 原始浮点模型 | checkpoint、配置、tokenizer/processor、原生模型代码 | OELLM 浮点模型、静态部署规格、参考输入与参考结果 |
| 量化仿真模型 | OELLM 浮点模型、量化配置、校准数据 | 冻结的量化状态、量化参考模型与比较结果 |
| BC 计算图 | OELLM 模型、冻结的量化状态、静态 TensorType、目标架构 | Exported BC、Converted BC、图接口与算子统计 |
| HBM 部署模型 | Converted BC、目标编译配置、Host 运行时 | HBO、HBM、配套资源、板端精度与性能结果 |

OELLM 以两种执行状态连接上述阶段。`compile_mode(False)` 调用各模块的 PyTorch `forward()`，用于浮点计算和校准；量化状态固定后，`compile_mode(True)` 将同一套模块切换到 `build()`，由 LEAP/HBIR 构造静态计算图。`forward()` 是否同时模拟量化数学，取决于具体量化模块的实现。后续产物按固定顺序生成：

```text
forward() 数值与校准路径
  -> 校准并冻结量化参数
  -> build() 图构造路径
  -> export_module     -> Exported BC
  -> convert_mlir      -> Converted BC
  -> compile_hbo       -> HBO
  -> link_models       -> HBM
```

模型适配完成后，OELLM 统一构建入口可以连续执行校准、BC 导出、BC 转换、HBO 编译和 HBM 链接。原生浮点复现、OELLM 浮点等价性验证和量化参考验证需要独立完成。各阶段均保留产物，以便只跨一个阶段比较数值结果。

## 1. 原始浮点模型

本阶段将 checkpoint 恢复为可执行的原生浮点模型，再实现与其数值等价的 OELLM 浮点模型。需要完成模型文件检查、原生推理复现、静态部署规格确定、权重装载和浮点结果比较。该阶段不引入量化。

完成本阶段后，应得到一套能够重复运行的浮点参考结果，以及一份输入输出固定、可进入校准和 BC 导出的 OELLM 模型实现。

### 1.1 检查 checkpoint 与配套文件

首先确认模型目录能够完整恢复原模型。一个可用的 checkpoint 包通常包括：

```text
模型权重或分片权重
权重索引文件
模型结构配置
tokenizer 及其配置
VLM processor / 图像预处理配置
特殊 token 与生成配置
```

检查权重分片是否齐全，配置中的层数、hidden size、Attention head、KV head、词表大小和 dtype 是否与权重形状一致。若模型使用 tied embedding、共享投影或其他共享参数，也应在此阶段确认共享关系。

随后记录 checkpoint 来源、版本和文件校验值。这些信息用于区分模型版本变化与后续量化变化，不应只通过目录名判断模型身份。

### 1.2 复现原生框架的浮点推理

使用模型原仓库或 Hugging Face 实现加载 checkpoint，先完成一次不经过 OELLM 的浮点推理。加载日志中不应存在未经解释的 missing keys、unexpected keys 或 shape mismatch。

参考推理需要固定以下条件：

```text
模型 eval 状态与计算 dtype
图像和文本预处理
prompt / chat template
tokenizer 与特殊 token
生成方式和采样参数
随机种子或确定性解码设置
```

保存预处理后的模型输入，而不只保存原始图片和文本。LLM 至少保存 token IDs、position IDs、attention mask 和必要的 KV 输入；VLM 还应保存 Vision 输入及视觉特征注入前后的 Language 输入。

参考输出包括最终结果和关键张量。Vision 可保存最终视觉特征；Language 可保存 Prefill logits、Decode logits 和 K/V update。它们将作为后续浮点适配和量化仿真的直接比较对象。

### 1.3 确定模型拆分与静态部署规格

原生模型通过后，再确定部署时需要编译的计算图。常见 LLM 拆分为 Prefill 和 Decode，Embedding 可按运行时能力放在 Host 或单独导出。VLM 还需增加 Vision 图，并明确视觉特征如何进入 Language。

每张图都要固定名称、输入输出顺序、shape、dtype 和状态更新方式。LLM/VLM 的静态规格通常包括：

```text
batch size
Prefill chunk size
Decode query length
KV cache length 与 KV layout
hidden size、层数、Attention head 和 KV head
词表大小与 logits 输出范围
图像宽高、patch 规格和视觉 token 数量
Vision 输出顺序及视觉特征注入位置
```

静态规格必须由目标任务的真实输入长度和设备资源共同确定。`chunk_size` 决定 Prefill 图一次接收多少位置，`cache_len` 决定可保留多少历史 K/V；两者不是模型最大生成长度的同义参数。VLM 还要为图像 token、模板和文本 prompt 同时预留位置。

此时再划分 Host 与 BPU 的职责。Host 通常保留 tokenizer、图像解码、输入组装、采样和图调度；BPU 执行 Vision、Prefill 和 Decode 等静态计算图。职责划分会直接决定图接口，因此必须在实现 OELLM 模型前固定。

### 1.4 实现 OELLM 浮点模型

OELLM 适配包括模型 API 和静态模型实现。模型 API 读取配置与 checkpoint，构造真实输入，并组织校准和编译。静态模型为同一计算定义两条路径：`forward()` 使用 PyTorch 张量复现数值，`build()` 使用 LEAP/HBIR 值构造部署图。两条路径共用权重、量化配置和静态接口。

接入一个新模型需要完成三层注册关系：

```text
model_factory
  -> 注册 model_name、目标 march 和 Model API 构造函数
Model API
  -> 处理 checkpoint、校准数据、静态参数和 compile() 编排
Model implementation
  -> 实现 forward()/build()、各图 TensorType 和构建调用
```

实现顺序如下：

1. 根据原模型配置构造 OELLM 模型结构；
2. 将 checkpoint 参数名映射到 OELLM 模块；
3. 处理必要的权重转置、共享参数和常量；
4. 检查每个 checkpoint 参数是否被正确加载；
5. 为 Vision、Prefill、Decode 等图实现固定 shape 的输入构造；
6. 在 `compile_mode(False)` 下运行 `forward()`。

静态 shape、动态控制流和后端不支持算子的改写必须保持原计算语义。具体算子适配与等价权重折叠属于模型实现技巧，应单独记录在模型适配文档中，不应改变本流程的阶段顺序。

### 1.5 验证 OELLM 浮点结果

使用 1.2 保存的同一份预处理张量，分别运行原生浮点模型和 OELLM 浮点模型。比较顺序应从图输入附近开始，再逐步检查 Transformer block、最终输出和端到端结果。

Vision 重点比较视觉特征的 shape、token 顺序和数值；Language 在相同 token 历史、position、mask 和 KV 下比较 Prefill/Decode logits 与 K/V update。端到端文本或结构化结果用于确认整体推理语义，但不能替代中间张量比较。

本阶段应保留：

```text
checkpoint 与配置文件清单
静态部署规格
OELLM 浮点模型代码
固定参考输入
原生浮点输出与 OELLM 浮点输出
浮点结果比较记录
```

只有浮点适配结果满足要求，才进入量化。否则后续误差无法区分是模型改写造成，还是量化造成。

## 2. 量化仿真模型

本阶段根据部署目标配置量化模块，通过真实校准输入固定权重 Scale、激活统计和其他量化状态。随后建立与 `build()` 量化数学一致的 PyTorch 参考计算，用于在导图前评估量化误差。该阶段不生成 BC，也不依赖目标设备。

完成本阶段后，应得到模块级量化配置、冻结的量化状态和量化参考结果。若某些量化算术只在 `build()` 中实现，则需明确记录，并在 Exported BC 上完成这部分数值验证。

### 2.1 制定量化配置

量化配置应落实到具体模块，而不是只写一个全局位宽。至少需要确定：

```text
哪些权重使用 W4、W8 或浮点
哪些激活使用静态量化、动态量化或浮点
QK、WV、KV cache 和 lm_head 是否量化
Scale 粒度、signed/unsigned 和 zero point
累加与输出 dtype
不参与量化的算子和边界
```

配置结果应形成可读取的模块清单，使校准、PyTorch 仿真和 BC 导出使用同一份定义。量化表示和模块选择方法见量化技巧文档，流程文档只要求配置在三个阶段保持一致。

### 2.2 准备校准输入

校准输入必须由 1.3 确定的静态部署规格生成，并复用真实推理的数据处理过程。

LLM 校准需要覆盖实际 prompt 长度、Prefill 输入和带历史 KV 的 Decode 输入。VLM 校准还需执行图像预处理、Vision forward、视觉特征注入和多模态 prompt 构造。随机 embedding、固定零 KV 或脱离 tokenizer 的随机 token 不能代表真实推理分布。

校准数据用于生成量化参数，独立验证数据用于判断量化后的任务精度。两者应分别保存样本清单和处理配置，避免用同一批结果同时决定参数并评价最终效果。

### 2.3 收集统计并冻结量化参数

按量化配置构造或切换 OELLM 量化模块，在 `compile_mode(False)` 下执行校准前向。静态激活量化在此过程中收集数值范围；权重量化根据既定粒度计算参数；动态激活量化则保存运行时计算 Scale 所需的配置。

校准应覆盖每一种实际图形态。例如 Prefill 与 Decode 的 query length、mask 和 KV 历史不同，应分别执行。VLM 的 Vision 和 Language 也应分别确认量化节点已经经过真实输入。

校准结束后冻结量化参数，并检查：

```text
计划中的量化模块是否全部执行
Scale 和 zero point 的 shape 是否正确
Scale 是否存在异常零值、NaN 或 Inf
量化轴和 group size 是否与模块一致
参数是否绑定到正确的 checkpoint 和校准数据版本
```

冻结后的量化状态应随模型状态、构建配置或独立清单固定，并能追溯到 checkpoint 与校准数据。后续 BC 导出不得重新统计或静默生成另一套参数。

### 2.4 运行 PyTorch 量化仿真

量化仿真是与部署图量化数学一致的 PyTorch 参考计算。它使用相同的权重、Scale、码域、舍入和截断规则，使量化误差能够在进入编译器前单独测量。

OELLM 不保证所有量化模块的 `forward()` 都执行完整量化数学。带 Fake Quant 的模块可以直接形成参考结果；部分动态量化模块的 `forward()` 仍执行浮点运算，量化与整数矩阵乘只在 `build()` 中生成。遇到后一种情况，应补充对应的 PyTorch 参考实现，或将 Exported BC 作为该模块的第一个完整量化数值检查点。普通浮点 `forward()` 不能直接称为量化仿真。

仿真模型应复用第 1 章的图拆分和输入构造。Vision、Prefill、Decode 的输入输出不能为仿真单独改变；否则比较结果同时包含接口变化和量化变化。

### 2.5 比较浮点模型与量化仿真模型

量化参考计算完整时，使用固定验证输入比较 OELLM 浮点模型和量化仿真模型。先比较单图输出和关键层张量，再比较端到端任务结果。自回归模型应先固定历史 token 和 KV 比较单步 logits，再运行完整生成观察累积变化。

量化参考结果满足精度要求后，再固定量化配置并进入 BC 导出。结果不满足要求时，返回 2.1 调整模块配置，或返回 2.2 补充校准数据。

完整量化参考计算尚未实现时，本阶段至少完成校准状态检查和已覆盖模块的数值比较，并将 Exported BC 设为全图量化结果的下一检查点。

本阶段应保留：

```text
模块级量化配置
校准样本清单与处理参数
冻结的 Scale、zero point 和量化权重
PyTorch 量化参考实现及其覆盖范围
浮点模型与量化参考计算的比较结果
```

## 3. BC 计算图

本阶段将 OELLM 模型、冻结的量化状态和静态 TensorType 导出为 OELLM/HBDK 计算图。首先生成保留前端计算语义的 Exported BC，再转换为面向目标架构的 Converted BC。

完成本阶段后，每张图都应具有确定的名称和接口，量化参数已经写入图中，目标转换后的算子归属和数值结果也已完成检查。

### 3.1 定义 TensorType 与图接口

为每张 Vision、Prefill、Decode 或其他计算图建立静态 TensorType。TensorType 需要明确输入输出的顺序、shape 和 dtype，并与 1.3 的部署规格完全一致。

图接口清单至少包括：

```text
graph name
输入输出名称和顺序
shape、dtype 与 layout
常量权重和量化参数
KV cache 的层数、顺序和更新范围
Host 与图之间的边界 dtype
```

同时准备一组符合该接口的固定样例输入，用于导出后检查图结构和数值结果。

### 3.2 导出 Exported BC

量化参数冻结后调用 `compile_mode(True)`。此时模块调用从 `forward()` 转入 `build()`，并发射 LEAP/HBIR 算子。典型调用由模型 API 组织，核心过程为：

```text
OELLM 模型 + 冻结量化状态 + TensorType + graph name
  -> export_module / leap_export
  -> Exported BC
```

Exported BC 保存前端计算图、常量权重和量化表达，还不包含目标设备的指令与分核结果。Vision、Prefill 和 Decode 应分别导出，文件名和 graph name 应保持一一对应。

导出后重新加载 BC，检查图数量、函数名称、输入输出接口、权重方向和量化参数。若工具链支持前端 BC 执行，则使用 3.1 的固定输入进行数值检查。完整量化参考计算已经实现时，再将其结果作为直接比较对象。

### 3.3 生成 Converted BC

使用 HBDK 将 Exported BC 转换为目标架构计算图。转换阶段根据 `march` 完成算子合法化、目标算子选择、layout 调整、类型转换和图融合，并保存 Converted BC。

```text
Exported BC
  -> HBDK convert / OELLM convert_mlir
  -> Converted BC
```

输入输出边界的 Quantize/Dequantize 是否保留，由 Host 与 HBM 的接口 dtype 决定。任何边界修改都应由统一的运行时接口要求驱动，并同步更新图接口清单。

### 3.4 检查目标图与数值结果

对 Converted BC 执行算子统计，检查量化 Linear、Attention 和其他主要计算的算子类型与后端归属。未合法化、进入非预期后端或触发 fallback 的算子需要单独处理。转换成功本身不等于主要计算已经进入目标后端。

若 Converted BC 可以执行，使用同一份固定输入依次比较：

```text
PyTorch 量化参考（若已实现） -> Exported BC
Exported BC                  -> Converted BC
```

比较只跨过一个转换阶段。这样才能判断变化来自图构造，还是来自目标后端转换。

本阶段应保留：

```text
每张图的 TensorType 与接口清单
Exported BC
Converted BC
图结构与算子统计
BC 数值比较结果
编译器版本与目标 march
```

## 4. HBM 部署模型

本阶段将 Converted BC 编译为目标对象 HBO，再链接为 HBRT 可以加载的 HBM。随后补齐 tokenizer、processor、Embedding 和 Host 调度，完成目标设备上的端到端运行。

完成本阶段后，应形成一套可传输、可校验、可启动的部署包，并保存固定测试条件下的精度和性能结果。

### 4.1 配置目标编译参数

编译配置至少包括：

```text
目标架构 march
优化等级 opt
编译并行度 jobs
每张图的 BPU core_num
片上内存上限与内存策略
input/output padding 方式
HPC 或其他目标优化开关
编译器与运行时版本
```

`jobs` 控制编译机上的 CPU 并行任务数，`core_num` 控制生成程序使用的目标 BPU 核数，两者不能混用。Vision、Prefill 和 Decode 的计算特征不同，可以分别设置 core 数，但最终配置必须与板端资源一致。

OELLM 的统一入口通常以如下参数组织构建：

```bash
python -m leap_llm.apis.oellm_build \
  --model_name <model_name> \
  --input_model_path <checkpoint_dir> \
  --output_model_path <output_dir> \
  --march <target_march> \
  --chunk_size <prefill_chunk> \
  --cache_len <kv_cache_len> \
  --calib_text_path <calibration_manifest> \
  --calib_image_path <calibration_image_root> \
  --w_bits <4_or_8> \
  --vit_core_num <1_or_2_or_4> \
  --prefill_core_num <1_or_2_or_4> \
  --decode_core_num <1_or_2_or_4> \
  --jobs <compile_jobs>
```

校准参数由具体 Model API 决定。LLM 通常使用 `--calib_text_path` 或 `--calib_json_path`；VLM 还需提供图像规格、图像目录或多模态清单，不应机械填写所有校准参数。不同模型 API 也可能增加 Decode 长度或专用参数。命令行只是构建入口，实际工作仍按前述四个阶段展开。

### 4.2 编译 HBO

编译器对每张 Converted BC 执行代码生成、调度、tiling、片上内存分配和多核切分，输出 HBO：

```text
visual_convert.bc  -> visual.hbo
prefill_convert.bc -> prefill.hbo
decode_convert.bc  -> decode.hbo
```

每张 HBO 都应能够被工具链重新加载。编译日志需保留目标架构、core 数、内存配置、开始与完成状态。HBO 是待链接的目标对象，不是板端最终加载文件。

### 4.3 链接 HBM

使用 HBDK linker 将相关 HBO 组合为 HBM。通常 Vision 单独链接为 Vision HBM，Language 的 Prefill 和 Decode 链接为 Language HBM；具体组合由运行时的加载方式决定。

链接完成后读取 HBM 图目录，逐图确认：

```text
graph name
输入输出数量、顺序、shape 和 dtype
目标 core 数
模型文件大小
SHA256
```

HBM 能够被加载只说明文件和接口可解析，不代表数值结果已经正确。

### 4.4 准备配套资源并接入 HBRT

部署包除 HBM 外，还可能包含：

```text
tokenizer 与 processor
Host 侧 Embedding 权重
模型和生成配置
图接口清单
预处理与后处理代码
HBRT/HB DNN 调度程序
```

Host 按固定流程准备 Vision、Prefill 和 Decode 输入，调用 HBRT 加载 HBM、分配输入输出 buffer、提交图任务并读取结果。VLM 还需在 Host 侧保持图像预处理、视觉 token 注入和 Language 输入位置一致；LLM 运行时需要正确维护 position、attention mask 和 KV 有效区间。

运行时实现应以 3.1 保存的图接口清单为唯一接口依据，避免在代码中重新推测 tensor 顺序或 shape。

### 4.5 验证板端精度与性能

板端验证分为图级数值验证和端到端任务验证。

图级验证使用 3.1 的固定输入，比较 Converted BC 与 HBM 在目标设备上的输出。Vision 比较视觉特征，Language 比较 logits 和 K/V update。端到端验证则使用独立任务数据，分别计算浮点模型和 HBM 模型相对于 Ground Truth 的任务指标。

完整比较顺序为：

```text
原生浮点模型
  -> OELLM 浮点模型
  -> PyTorch 量化仿真（若已实现）
  -> Exported BC
  -> Converted BC
  -> 目标设备 HBM
```

工具链支持编译侧执行或仿真时，可在 Converted BC 与目标设备 HBM 之间增加一次同输入比较。

性能测试固定图片、prompt、输入长度、输出长度、batch、生成模式和采样参数。分别记录模型加载、图像预处理、Vision、Prefill、Decode、Host 采样与端到端时间，并区分首次运行和模型常驻后的重复运行。

最终部署包应包含：

```text
Vision / Language HBM
必要的 Embedding、tokenizer 和 processor
Host 运行时与配置
checkpoint、校准数据和构建参数记录
BC、HBO、HBM 的文件校验值
板端精度与性能结果
```

这套产物闭合了从 checkpoint 到目标设备的完整 OELLM 量化流程。后续更换模型时，应先替换第 1 章的模型适配，再按相同顺序重新完成量化、BC 和 HBM 阶段，而不是直接复用旧模型的量化参数或编译产物。
