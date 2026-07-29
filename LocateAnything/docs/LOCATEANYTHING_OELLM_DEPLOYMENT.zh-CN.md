# 基于 OE_LLM 的 LocateAnything-3B 部署记录

本文记录 LocateAnything-3B 从浮点模型适配、量化配置、BC 计算图、HBO 目标代码、HBM 板端模型到 D-Robotics S600 推理的完整实现。内容聚焦 OE_LLM 拆图方式和精度配置的选择依据。板端部分说明如何保持并行框解码（PBD）、自回归解码（AR）、KV 历史缓存和坐标输出的原始语义。

与本文配套的三篇文档分别承担不同内容：

- [LocateAnything 原始浮点模型技术架构](LOCATEANYTHING_FLOAT_MODEL.zh-CN.md)说明 MoonViT、Language Decoder、六类定位任务和 PBD/AR 原理。
- [LLM-VLM 量化流程整理文档](LLM_VLM_QUANTIZATION_PIPELINE.zh-CN.md)说明 Float、量化仿真、BC、HBO、HBM 的通用编译阶段。
- [LLM-VLM 量化技巧整理](LLM_VLM_QUANTIZATION_TRICKS.zh-CN.md)说明 W4/W8、Static/Dynamic A8、正交旋转、Value 中心化和混合精度。

本文集中说明 LocateAnything 的具体实现和部署结果。文中的状态分为两类：三图
Language HBM 是已经过板端功能与性能检查的历史稳定版本；当前源码发布合同为
13 张 Language 图。13 图版本只有在重新完成 BC/HBM 编译、图目录检查、S600
数值对齐和六任务评测后，才可升级为新的板端稳定版本。

~~~text
原始浮点模型
  -> 固定 672x672 Vision、1024 Prefill、4096 KV 配置
  -> Vision 与 Language 量化仿真
  -> Exported BC -> Converted BC -> HBO -> HBM
  -> S600 专用运行时
  -> 图像、文本、坐标和性能验证
~~~

## 1. 原始浮点模型与固定部署配置

### 1.1 部署模块与数据流

LocateAnything 被拆成四个部署部分：

| 部分 | 产物或实现 | 作用 |
|---|---|---|
| Vision | MoonViT visual HBM | 把图像编码为 576 个视觉 token |
| Language | 13 图 Language 候选 HBM | 建立 KV cache，执行 Prefill、PBD、AR 与融合 KV 提交 |
| Embedding | FP16 Token Embedding | 在 Host 侧把 token ID 映射为 2048 维向量 |
| Runtime | LocateAnything 专用 Host 程序与 HBRT | 完成预处理、特征注入、解码调度和坐标还原 |

端到端数据流为：

~~~text
JPEG / PNG
  -> 保持宽高比缩放并填充到 672x672
  -> 14x14 patchify
  -> Vision input [1,2304,588] FP16
  -> visual HBM
  -> Vision feature [1,576,2048] FP16

task command / text query
  -> LocateAnything prompt template
  -> tokenizer
  -> input_ids with 576 <IMG_CONTEXT> positions
  -> Host 查询 Token Embedding
  -> 用 Vision feature 替换图像占位 embedding
  -> Prefill HBM
  -> q=1 AR，或 q=6 PBD + q=1 AR 回退
  -> <ref> / <box> / coordinate tokens
  -> [0,1000] 坐标还原到原图像素
~~~

视觉特征不是追加到文本末尾。Host 先生成 576 个图像占位 token，再逐位置替换对应 embedding，因此序列长度保持不变。图文融合发生在 36 层 Language self-attention 中，没有独立的 cross-attention 图。

### 1.2 LocateAnything 的部署差异

LocateAnything 不能直接套用通用 Qwen2.5-VL 运行时。两者在视觉编码、位置编码、空间词表和生成方式上均不同。

| 部分 | LocateAnything 实现 |
|---|---|
| Vision | 27 层 MoonViT，二维位置表和二维 RoPE |
| Vision 合并 | 2x2 相邻 token 通道拼接，不是平均池化 |
| Language | 36 层 Qwen2.5-3B Decoder，Language 使用一维 RoPE |
| 空间输出 | 1001 个坐标 token，坐标范围为 [0,1000] |
| 快速解码 | PBD q=6，一次预测 `<box> x1 y1 x2 y2 </box>` |
| 回退解码 | AR q=1，按原始状态机接续未完成结构 |

固定图中的 Vision 计算简化为：

~~~text
[1,2304,588]
  -> patch Linear, 588 -> 1152
  -> 48x48 二维位置编码
  -> 27 x MoonViT Block
  -> 2x2 token 拼接, 4 x 1152 = 4608
  -> projector, 4608 -> 2048 -> 2048
  -> [1,576,2048]
~~~

Patch Embedding 的 Conv2d 权重在导出时展平为 Linear 权重，二维位置表和 RoPE 表按 48x48 网格预先生成。详细模型结构见浮点模型技术架构文档，本文不再逐层展开。

Language 保留以下关键参数：

| 参数 | 数值 |
|---|---:|
| Hidden size | `2048` |
| Decoder layers | `36` |
| Query / KV heads | `16 / 2` |
| Head dimension | `128` |
| MLP intermediate size | `11008` |
| Vocabulary | `152681` |
| RoPE theta | `1000000` |
| `max_position_embeddings` | `32768` |
| Tied embedding | `true` |

`max_position_embeddings=32768` 是浮点模型的位置编码上限，不是当前 HBM 的实际上下文容量。板端容量由 Prefill 长度、KV cache 和输出预算共同决定。

### 1.3 固定部署参数

| 部分 | 参数 | 设置 |
|---|---|---:|
| Vision | 输入分辨率 | `672x672` |
| Vision | Patch size | `14` |
| Vision | Patch 数量 | `48x48=2304` |
| Vision | 输出 token | `576` |
| Vision | 输入 / 输出 | `[1,2304,588] / [1,576,2048] FP16` |
| Vision | 权重量化 | `W8` |
| Language | Prefill chunk | `1024` |
| Language | Prefill 剩余位置 | 最多 `448`，还需包含模板和特殊 token |
| Language | KV cache | `4096` |
| Language | PBD / AR query length | `6 / 1` |
| Language | Decoder / `lm_head` 权重量化 | `W8 / W8` |
| Language | 候选图数量 | `13` |
| Language | `max_new_tokens` | 默认 `2048` |
| Calibration | 发布样本 / 收敛检查点 | `1200 / 512` |
| Evaluation | Box IoU 阈值 | `0.90` |
| Runtime | Batch size | `1` |
| BPU | Vision / Prefill / PBD / AR cores | `4 / 4 / 4 / 4` |
| Runtime | 默认模式 | `hybrid`（q=6 PBD，必要时回退 q=1 AR） |

图像按原宽高比缩放并填充，填充值为 128。后处理保留缩放比例和四边 padding，用于把模型坐标逆变换回原图。例如 `<495><214><682><469>` 是 [0,1000] 归一化坐标，不是原图像素。

`chunk_size=1024` 表示 Prefill 图一次接收 1024 个位置，其中 576 个位置已由视觉 token 占用。`cache_len=4096` 表示可保存的历史 KV 长度。`max_new_tokens=2048` 是生成上限，不改变 HBM 的固定 shape。

### 1.4 Checkpoint 适配与浮点验证

Vision 权重来自 `vision_model.*` 和 `mlp1.*`，Language 权重来自 `language_model.*`。适配时需要保持四项内容一致：

1. Patch 权重的展平顺序和二维位置表；
2. 27 个 Vision Block 与 36 个 Language Layer 的完整加载；
3. Embedding 与 `lm_head` 的 tied-weight 关系；
4. Vision token 顺序、RoPE、PBD mask 和位置 ID。

完成固定图改写后，先比较原始 PyTorch 与适配后的 PyTorch：

~~~text
Original Float
  -> Fixed-shape Adapted Float
  -> Vision output、logits、KV、argmax 和任务输出
~~~

这一步只验证模型适配。浮点结果不一致时，后续量化和编译结果没有可解释性。

## 2. 量化仿真与精度方案

### 2.1 最终量化配置

量化方法的数学原理见《LLM-VLM 量化技巧整理》。LocateAnything 最终采用的配置如下：

| 路径 | 当前配置 |
|---|---|
| Vision Linear | 动态 A8 + W8 |
| Vision QK | 动态 S8 x S8 |
| Vision WV Attention | 动态 S8 |
| Vision WV Value | token 均值中心化后动态 S8 |
| Vision WV 补偿项 | FP16 |
| Language 252 个 Decoder Linear | 动态 A8 + W8 |
| Language `lm_head` | 动态 A8 + W8 |
| Language QK / WV | 校准得到的对称 A8 范围 |
| Language KV cache 量化边界 | 校准得到的对称 A8 范围 |
| Embedding | FP16 |

这里有两种激活量化方式。Scale 是浮点值映射到整数码值时使用的缩放系数。Dynamic A8 在每次推理时按当前输入计算 Scale；校准 A8 则在校准数据上确定固定范围，并写入编译图。Language 并不是所有路径都使用 Dynamic A8。

### 2.2 隐藏空间统一

Vision、Language、Embedding 和 `lm_head` 必须处于同一个 2048 维隐藏空间。部署使用带符号并归一化的 Sylvester-Hadamard 正交矩阵 Q：

~~~text
Q.T @ Q = I
~~~

Q 只改变隐藏向量的坐标基，不改变浮点模型函数。它被离线折叠到 Embedding、Vision projector、Language 输入/输出投影和 `lm_head`，运行时不增加一个新的 2048x2048 MatMul。

该方法的通用推导见量化技巧文档。部署时，三份产物必须使用同一矩阵和同一折叠顺序；否则各模块可以单独运行，组合后的 token 语义仍会错误。

### 2.3 校准前向运行

校准前向运行是指让真实校准输入实际经过模型，以收集每个固定量化位置的激活范围。它不是一个精度指标，也不是把旧输出重新播放一遍。

当前发布编译合同使用 1200 条输入：Detection 620 条，GUI 180 条，Referring 120 条，
OCR 120 条，Layout 100 条，Pointing 60 条；512 条仅用于与完整统计结果比较 Scale
收敛。每条输入均按 672x672 处理，并实际执行：

~~~text
Vision
  -> Language Prefill
  -> PBD q=6
  -> AR q=1
~~~

Language 图中共有 289 个需要收集范围的位置。`289/289` 表示这些位置都在真实前向运行中被执行，并得到有限、非零的量化参数；它不表示有 289 条样本，也不表示精度已经通过。

编译时使用两份辅助文件：

- **量化参数清单**（scale manifest）：保存每个量化位置的 Scale 及数据版本；
- **执行覆盖记录**（coverage）：记录 Vision、Prefill、PBD 和 AR 是否实际运行。

这两个文件用于防止在统计缺失时直接导出 BC。量化精度仍需由 PyTorch Q/DQ 仿真、BC 和 S600 结果判断。

### 2.4 Vision 精度方案

Vision 的主要精度瓶颈位于 Attention WV。W8 Linear 的局部误差较小，而 Value 在 token 维可能带有公共偏移。直接使用对称 S8 会浪费部分表示范围，因此部署采用中心化后再量化：

~~~text
Vmean = mean_token(V)
Vc    = V - Vmean
A @ V = A @ Vc + sum_token(A) * Vmean
A @ V ≈ Quant(A) @ Quant(Vc) + sum_token(Quant(A)) * Vmean
~~~

均值项在 MatMul 后以 FP16 加回，浮点信息没有被删除。FP16 WV 和 A16xA8 也做过对比，但 Converted BC 会退回 CPU 执行，无法形成完整的 BPU 路径，因此没有进入最终配置。

历史三图板端版本曾以 820 条校准输入得到以下 Vision 数值记录：

| 指标 | 结果 |
|---|---:|
| Float -> HBM mean cosine | `0.971394` |
| P05 cosine（第 5 百分位） | `0.957296` |
| Min cosine | `0.920964` |
| cosine < 0.95 | `14/820` |

该历史记录的整体均值达到当时使用的 0.95 数值检查目标；最低值和 14 个尾部样本仍作为已知风险保留。

### 2.5 Language 精度方案

Language 的 36 层 Decoder 每层包含 Q/K/V/O 和 gate/up/down 七个 Linear，共 252 个 Linear，全部使用 W8。`lm_head` 同样使用 W8，避免大词表投影中的 W4 排序误差。

Linear 输入采用逐行 Dynamic A8。QK、WV 和 KV cache 量化位置使用校准前向运行得到的固定对称 A8 范围。当前候选必须由完整 1200 条发布校准集重新产生这些范围；历史三图版本使用的 820 条统计只作为历史数值证据。W8/W8 只说明 Decoder 与 `lm_head` 的权重量化，不能据此判断所有激活都使用 Dynamic A8。

PyTorch Q/DQ 量化仿真用于检查当前量化数学。Q/DQ 是 Quantize/Dequantize 的缩写，即先把浮点值映射到整数码值，再反量化为浮点近似值继续计算。

仿真结果用于观察 Vision 张量、Language logits、PBD/AR 路径、坐标 IoU/PCK 和最终任务输出。它不包含 HBDK 的后端图改写，因此不能替代 BC 和 S600 验证。

## 3. OE_LLM 计算图与 HBM 编译

### 3.1 工具链分工

| 层级 | 负责内容 |
|---|---|
| LocateAnything PyTorch 适配 | Checkpoint 加载、固定 shape、校准和 Q/DQ 量化仿真 |
| OE_LLM / LEAP / HBIR | 描述 Vision、Prefill、PBD 和 AR 计算图 |
| HBDK | 转换图、选择 Nash-P 算子、融合、生成 HBO 并链接 HBM |
| HBRT / HB DNN | 在 S600 上加载和调度 HBM |
| LocateAnything Host Runtime | Tokenizer、Embedding 查询、KV 管理、PBD/AR 状态机和坐标恢复 |

OE_LLM 负责把模型计算表达为可编译图，不负责 LocateAnything 的完整交互逻辑。PBD 片段接收、AR 回退和坐标解析仍由专用运行时完成。

### 3.2 图划分与编译产物

当前已部署 Language HBM 包含三张图：

~~~text
prefill     q=1024
decode      q=6, PBD
decode_ar   q=1, AR
~~~

Vision 单独生成 `visual` 图。每张图经过相同的产物链：

~~~text
Exported BC
  -> Converted BC
  -> HBO
  -> HBM
~~~

这些名称的通用含义见量化流程文档。在本项目中：

- Exported BC 保存模型适配层导出的固定图；
- Converted BC 完成 Nash-P 算子替换、量化和融合；
- HBO 是单张图的目标代码；
- HBM 把同一模块的多张 HBO 链接为板端模型。

Converted BC 阶段需要确认算子仍在 BPU 路径上。文中所说“退回 CPU 执行”是指某个算子没有转换为 Nash-P 算子，而由 Host CPU 执行；这种方案即使数值更好，也不作为当前 BPU 部署结果。

融合 PBD 候选以历史三图为基础，增加 `decode_pbd_q7` 至 `decode_pbd_q12`、`decode_ar_q2` 至 `decode_ar_q5` 共 10 张变体，所以候选 Language 图集合共有 13 张图。它仍处于 `workspace/builds/` 的编译候选阶段，不能与已部署三图 HBM 混用，也不能写成已获得板端加速结果。

### 3.3 编译参数

| 参数 | 设置 |
|---|---|
| Target | `nash-p` |
| Optimization | `opt=2` |
| Compile jobs | `16` |
| BPU cores | `4` |
| BPU 本地内存上限（L2M） | `25165824` bytes |
| S600 四核 L2M 划分 | `6:6:6:6` |

`nash-p` 是 S600 BPU 的目标架构。`jobs=16` 是编译机 CPU 并行度，`cores=4` 才是目标 BPU 核数。HBO 代码生成主要消耗 CPU、内存和磁盘，不应根据 GPU 利用率判断是否仍在编译。

### 3.4 编译前检查

以下命令核对校准样本数量、量化参数清单、执行覆盖记录和固定图参数，不执行长时间 HBO 编译：

~~~bash
cd ~/oe_locateanything/LocateAnything
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

python compiler/quantize.py verify --component language --level contract
~~~

校准清单、Scale、图覆盖和固定 profile 均由 `compiler/config.yaml` 统一指定；切换
发布配置时修改配置文件或使用统一入口提供的路径参数，不再通过零散环境变量拼装合同。

### 3.5 Vision 与 Language 编译

Language：

~~~bash
python compiler/quantize.py build \
  --component language --target hbm --resume
~~~

Vision：

~~~bash
python compiler/quantize.py build \
  --component vision --target hbm --resume
~~~

Vision 模型类中的 Linear 权重固定为 W8。历史目录名中的 `w4` 字段不能代表实际 BC 中的位宽，最终配置以模型代码、量化参数清单和 BC 算子为准。

## 4. S600 运行时与验证结果

### 4.1 专用运行时

SDK 的通用 `vlm/libxlm` 路径不包含 LocateAnything 的 MoonViT、Language 一维 RoPE、PBD mask 和空间 token 协议。本项目因此直接使用 HBRT/HB DNN 调用 HBM，同时保留可复用的 tokenizer。

Host 与 BPU 的分工为：

| Host | BPU / HBM |
|---|---|
| 图像解码、letterbox、patchify | MoonViT Vision |
| Tokenizer、prompt template | Language Prefill |
| FP16 Embedding 查询与视觉特征替换 | PBD q=6 / AR q=1 |
| Position IDs、mask、KV ring 管理 | 量化 Attention、MLP 和 `lm_head` |
| 采样、PBD 状态机和坐标解析 | 输出 logits 与新 K/V |

Language 每层有 K/V 两份 cache，共 72 个 cache tensor。运行时使用镜像环形缓冲，避免每个 token 搬移整个历史。PBD 预测出的草稿只有被状态机接受后才能写入有效历史；未接受草稿的 K/V 不能继续保留。

### 4.2 Slow、Hybrid 与融合 PBD

`hybrid` 是当前默认模式：先使用 q=6 PBD，结构合法时直接接收；结构不完整时转入 q=1 AR，完成当前框后再回到 PBD。`slow` 仅使用 q=1 AR，保留为显式对照模式。

已部署的三图 Hybrid 为了提交被接受 token 的正确 K/V，需要在 PBD 预测后重新执行一次图。这个重复调用限制了实际加速。

融合候选把“提交上一轮前缀”和“预测下一组 q=6 token”合并：

~~~text
first PBD       -> q6
next PBD        -> q(6+k), k in [1,6]
PBD -> AR       -> causal bridge qk, k in [1,6]
AR -> PBD       -> q(6+k)
~~~

q7-q12 图覆盖接收 1 到 6 个 token 的情况，q2-q5 causal bridge 处理错误 box 的合法前缀转 AR。q6 PBD 仅用于独立 Prefill 后的首轮启动；完整六 token block 不触发 AR，因此不需要 `decode_ar_q6`。

### 4.3 分层验证方法

| 对比 | 目的 | 主要指标 |
|---|---|---|
| Original Float -> Adapted Float | 验证模型适配 | cosine、max diff、logits/KV、argmax |
| Float -> PyTorch Q/DQ | 验证量化数学 | cosine、relative L2、margin、结构、IoU/PCK |
| Q/DQ -> Exported BC | 验证图导出 | 最终输出、关键边界、图接口 |
| Exported BC -> Converted BC | 验证后端转换 | 输出差、BPU 算子位置、shape |
| Converted BC -> S600 HBM | 验证设备执行 | 张量差异和端到端输出 |
| Float / Q/DQ / HBM -> Ground Truth | 验证任务效果 | Precision、Recall、F1、matched IoU、PCK |

Float 与量化结果的直接比较回答“量化改变了多少”，Ground Truth 比较回答“结果是否正确”。Detection 和 Grounding 按标签及 IoU 做一对一集合匹配，不能按生成顺序配框。

验证覆盖 Vision 数值一致性、Language slow/Hybrid 真实图片推理，以及标签、坐标和停止条件。量化版本与 Float 的差异用于定位数值变化；两者分别与 Ground Truth 比较，才能判断任务结果是否正确。

### 4.4 已部署模块

| 模块 | 部署内容 | 执行位置 | 状态 |
|---|---|---|---|
| Vision | 672x672、W8 MoonViT 图 | S600 BPU | 历史三图版本已部署 |
| Language | W8 Decoder、W8 `lm_head`，`prefill`/`decode`/`decode_ar` | S600 BPU | 历史三图版本已部署 |
| Embedding | FP16 Token Embedding | Host | 历史三图版本已部署 |
| Runtime | LocateAnything 专用 CLI、KV 管理、生成状态机和坐标后处理 | Host + HBRT | 历史三图版本已部署 |
| 融合 Language | 三图加 10 张融合/bridge 图 | 候选 HBM | 未完成 S600 发布验证 |

### 4.5 S600 启动

~~~bash
cd /home/sunrise/oe_locateanything/LocateAnything

cmake --build deploy/build \
  --target vision_hbm_runner language_hbm_runner \
  -j4

sh deploy/install_locateanything_cli.sh
export PATH="$HOME/.local/bin:$PATH"
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6

LocateAnything \
  -i workspace/samples/check_cat2.jpg \
  --max-new-tokens 2048
~~~

不传 `--generation-mode` 时默认使用 Hybrid。需要纯 q=1 AR 对照时显式选择 `slow`：

~~~bash
LocateAnything \
  -i workspace/samples/check_cat2.jpg \
  --generation-mode slow \
  --max-new-tokens 2048
~~~

交互模式中的任务命令包括 `/detect`、`/ground`、`/gui`、`/text`、`/layout` 和 `/point`。CLI 只负责把这些命令展开为模型原本使用的任务提示，不改变模型权重或量化配置。

### 4.6 小结

LocateAnything 部署的核心是保持整条模型语义一致：MoonViT 的二维视觉编码、576 个视觉 token 的注入、Language 隐藏空间、PBD q=6、AR q=1、KV 提交和坐标还原必须同时正确。

W8/A8 三图版本已经形成可运行的历史 S600 部署证据。当前源码将 13 图融合 PBD
定义为发布编译合同，但它仍需完成 BC/HBM 图目录、S600 数值对齐和六任务评测，才可
提升到 `workspace/artifacts/release/` 并替代三图版本。

## 5. 量化与部署问题记录

本章不重复命令、产物名称和过程状态，只讨论会改变模型数值、计算图语义、解码结果或端到端时延的技术问题。每一项均按问题表现、原因分析、解决方案和验证依据展开。后一级验证不能掩盖前一级错误：浮点适配未对齐时，量化误差没有解释基础；量化仿真通过时，也不能直接推断 BC 或 S600 结果正确。

本章中的 448、W4、2048 cache 和 820 条统计均为按日期保留的排查或历史三图证据，
用于说明问题与修复依据，不是当前发布默认值。当前候选以 672、W8/W8、1024/4096、
1200 条校准和 0.90 IoU 为合同。

### 5.1 MoonViT 固定图改写破坏浮点等价性

**问题表现。** 在关闭量化后，编译侧 MoonViT 已从前几个 Transformer Block 开始偏离原始浮点模型。二维位置向量的检查结果同时显示，静态位置表保持为零。此时提高位宽、扩大校准集或更换量化方法都不会改善结果，因为误差发生在量化之前。

**原因分析。** 固定分辨率部署需要把动态位置计算提前固化，但固化过程改变了两处原始语义。

第一处是二维 RoPE 的通道配对。MoonViT 将相邻通道解释为一个复数对：

~~~text
(a, b) -> (-b, a)
~~~

早期适配沿用了 Language RoPE 的前半区、后半区配对方式。两种实现使用相同的 cos 和 sin 公式，却作用于不同的通道组合，因此 Q/K 在进入 Attention 前已经不同。

第二处是位置表加载。`pos_emb_static` 是 `persistent=False` 的 Buffer，不属于常规 checkpoint 状态。把插值结果写入 `state_dict` 后再用 `strict=False` 加载，只会忽略这个额外键，不会更新实际 Buffer。静态图最终使用的是初始化零值，而不是插值后的 48x48 位置表。

**解决方案。** 二维 RoPE 改为相邻通道配对，每个复数频率通过 `repeat_interleave(2)` 对齐到两个实数通道。64x64 学习位置表按目标网格插值到 48x48 后，直接写入模型 Buffer，再进入 BC 导出。672x672 和 448x448 的 patch 数、视觉 token 数及位置表均不同，两个 profile 不共享输入张量和量化参数。

**验证依据。** 修复后，适配 Float 与原始 Float 的逐层比较为：

| 位置 | Cosine |
|---|---:|
| Patch Embedding | `0.99999994` |
| Block 0 | `0.99999988` |
| Block 26 | `0.99993134` |
| Final Norm | `0.99978495` |
| Projected Visual | `0.99989551` |

这些结果说明主要偏差不再来自位置编码和静态化改写。只有先完成这一层验证，后续 Vision 差异才能归因于量化或后端转换。

### 5.2 PyTorch 动态实现无法直接映射为静态 BC 图

**问题表现。** Vision 和 Language 在导出、转换及 HBO 编译阶段出现了不同形式的失败：嵌套 KV 返回值无法确定类型，原生 LayerNorm 无法处理编译器中间值，Attention MatMul 维度不匹配，Decode 编译进程在没有 Python traceback 的情况下退出。

**原因分析。** PyTorch 允许 Python 容器、运行时分支和隐式张量语义，BC 图则要求输出数量、shape、dtype、转置方式和内存布局在编译时确定。具体冲突包括：

1. Language 原始返回结构为 `(logits, list[K], list[V])`。导出器只能跟踪 Tensor 叶节点，不能把嵌套列表自动展开成固定输出。
2. `torch.nn.LayerNorm` 的执行入口只接收 `torch.Tensor`。LEAP 跟踪时传入的是表示图节点的 `OpResult`，必须使用能够生成 LEAP/HBIR 算子的实现。
3. `DynamicQuantMatmul` 已按 Trans-RHS 方式解释右操作数。若调用侧再次转置 K，就会形成双重转置；`FakeQuantMatmul` 则需要调用侧显式转置。两类封装不能共用同一套 shape 处理。
4. 词表大小为 152681。FP16 logits 每行占 `152681 x 2 = 305362` bytes，不能被 64 整除。强制关闭输出 padding 后，HBDK 在 Decode 编译路径中进入异常分支。

**解决方案。** Language 返回值展开为 `logits + 36 K + 36 V`，固定为 73 个 Tensor；MoonViT 归一化替换为 LEAP 可跟踪实现；QK 和 WV 分别按所用 MatMul 封装的转置规则组织输入。Logits 允许 HBDK 在设备布局中补齐到 305408 bytes，Host 根据输出 stride 读取每行，并只保留前 152681 个有效词表值。

**验证依据。** 修正后，Vision、Prefill、PBD 和 AR 均可完成 BC 导出、后端转换和 HBO 编译。Language 图稳定保持 73 个输出。词表补齐只改变设备内存布局，没有增加可采样 token，也没有改变模型权重。

### 5.3 跨模块隐藏空间不一致导致输出失去语义

**问题表现。** 旧版 HBM 能够加载，部分真实视觉输入也能产生非零 logits，但端到端输出没有可信语义。单独替换 Embedding、Vision projector 或 `lm_head` 只能改变症状，不能恢复完整推理链。

**原因分析。** 原始 checkpoint 的各模块处于同一隐藏空间，问题出现在量化重参数化阶段只对部分边界折叠了 Q。2048 维隐藏向量不仅有长度，还有确定的坐标基。若 Embedding 输出 `hQ`，而下一层仍按原坐标基解释 `h`，即使 Q 是正交矩阵，线性投影结果也会改变。正交变换保持内积的前提，是变换前后的相邻权重同时重参数化。只旋转一个模块会破坏 Vision、Embedding、Decoder residual 和 `lm_head` 之间的接口。

**解决方案。** 选用带符号、归一化的 Sylvester-Hadamard 正交矩阵 Q，并把坐标变换离线折叠进所有跨隐藏空间的权重。以行向量实现为例：

~~~text
Q.T @ Q = I
h'       = h @ Q
E'       = E @ Q
W_in'    = (W_in * gamma) @ Q
W_out'   = Q.T @ W_out
W_lm'    = (W_lm * gamma_final) @ Q
W_vis'   = Q.T @ W_vis
b_vis'   = b_vis @ Q
~~~

其中 `* gamma` 表示沿输入列吸收 RMSNorm 权重。Q/K/V 和 MLP 的内部语义不变，输出投影再把结果写回旋转后的 residual。该处理不在运行时增加 2048x2048 MatMul。

**验证依据。** Q 的正交误差为 `5.96046448e-08`。重参数化前后的 FP32 Language logits cosine 为 `0.999999999986`，KV 最大差值为 `6.109476e-05`；FP16 Vision 输出 cosine 为 `0.999999927`。这些数值验证的是浮点函数等价性。它们说明 Q 的折叠顺序正确，但不能单独代替量化后或板端验证。

### 5.4 校准程序没有覆盖真实量化路径

**问题表现。** 早期编译入口能够接收校准数据路径，也能够导出 BC，但日志中没有真实样本前向。另一次检查发现，Vision 的 108 个 Attention 统计位置全部为零。图可以继续编译，却会把初始范围或默认归一化参数写入量化图。

**原因分析。** 固定 A8 需要先运行真实输入，再从每个量化位置收集激活范围。`ConstFakeQuant.absmax` 只在前向执行时更新，RMSNorm 的数值保护参数也依赖实际 hidden energy。早期独立 Vision/Language API 在加载权重后直接进入 `compile_mode(True)`，没有执行任务前向。Vision Eager 路径又直接调用 `torch.matmul`，绕过了 QK/WV 的量化统计模块。传入数据路径、构造模型和执行量化统计是三个不同动作，前两个成功并不表示第三个已经发生。

672x672 发布配置还会改变统计分布。它产生 2304 个 patch 和 576 个视觉 token，早期 448x448 配置只有 1024 个 patch 和 256 个视觉 token。Attention 长度、图像占位数、Prefill 组成及位置编码均已变化，因此旧 profile 的范围不能直接复用。

**解决方案。** 在隐藏空间折叠和浮点等价性验证之后，使用实际 672x672 输入依次运行 Vision、Language Prefill、PBD q=6 和 AR q=1。Eager 与导出路径统一经过同一组 QK/WV 模块，确保统计节点看到的就是编译图将使用的张量。量化参数同时绑定输入 profile、模型配置和校准数据版本；任一项改变时重新采集。

这组历史统计覆盖为：Vision、Prefill 和 PBD 各 820 条，AR 128 条；Vision 108/108、Language 289/289 个统计位置均被执行，未出现零范围或无效归一化参数。它不能替代当前 1200 条发布校准的 Scale 清单。

**验证依据。** 从 512 条扩展到 820 条后，平均变化不大，但少数敏感位置仍有明显漂移：

| 模块 | Scale 平均相对变化 | 最大变化 | 变化超过 10% 的位置 |
|---|---:|---:|---:|
| Vision | `1.7865%` | `14.1956%` | 4 |
| Language | `1.2660%` | `35.3708%` | 10 |

平均值会掩盖尾部量化位置。历史三图编译使用完整 820 条统计，而不是用较小快照替代。这只能说明该 820 条快照更适合当时输入版本，不能证明量化范围已经对未知数据分布收敛，也不能替代当前 1200 条发布合同。

### 5.5 Language 低比特量化破坏坐标 token 排序

**问题表现。** 早期 W4 方案的敏感性检查在中后段 Decoder Block 观察到明显输出漂移，少量坐标 token 的候选顺序随之交换。另一个早期现象是 Prefill 可以产生非零 logits，而 Decode 接收较小幅度的文本 Embedding 后接近全零。这两类现象都发生在 Language 侧，但对应的量化位置不同。

**原因分析。** 对称权重量化的步长近似由张量范围除以整数正向最大值决定。相同范围下，W4 的有效网格远稀于 W8。误差经过 36 层 residual 传播后进入 152681 维 `lm_head`。坐标由 1001 个离散 token 表示，最终选择依赖候选 logits 的相对排序，而不是单个 hidden tensor 的平均误差。Top-1 与 Top-2 margin 较小时，很小的投影误差也足以交换坐标 token。

输入边界还有另一类风险。若整段多模态 Embedding 共用一个固定量化范围，幅度较小的文本向量可能只落在少量整数码值上，极端情况下接近零。该问题发生在第一层之前，后续层无法恢复已经丢失的方向信息。

**解决方案。** 删除 Language 输入边界的统一 QuantStub，让 FP16 Embedding 直接进入第一层。36 层 Decoder 的 252 个 Linear 和 `lm_head` 使用 W8；Linear 输入在每次执行时按行计算 Dynamic A8 Scale。Language QK、WV 和 KV cache 边界仍使用真实校准前向得到的固定对称 A8 范围，不能把当前方案笼统写成“全部 Dynamic A8”。

`lm_head` 使用 W8 的作用是避免再引入一层较粗的权重误差，不代表它是剩余 AR 尾部的唯一原因。定向 Float rescue 中，仅恢复 `lm_head` 并未消除低 IoU 尾部，因此没有依据把 `lm_head` 单独提高到 FP16。

**验证依据。** 早期 W4/W8 Block 对比显示，W8 在代表性敏感层更接近 Float，因此最终没有保留 W4 Language 候选。当前仓库没有保存一份可复现的完整 W4/W8 逐层报告，W8 应视为保守的精度配置，而不是已经完成全量任务评估的结论。现有 S600 功能证据来自固定 cat 样本：q=1 路径能够重复生成合法标签和坐标。这说明输入边界与主干量化不再导致整体失效，但不能由单个样本推断全部精度通过。

### 5.6 Vision Attention WV 对公共偏移敏感

**问题表现。** MoonViT 的 Linear 使用 W8 后，主要尾部误差集中在 Attention 的 WV MatMul。直接把 Attention probability 和 Value 都映射到对称 S8 时，部分样本的输出 cosine 明显低于总体水平。把 WV 临时改成 FP16 可以减小局部误差，却无法生成纯 BPU 执行图。

**原因分析。** Value 在 token 维常带有公共偏移。对称 S8 以绝对最大值确定 Scale：

~~~text
scale = max(abs(V)) / 127
q     = round(V / scale)
~~~

公共偏移会增大 `max(abs(V))`，使有限整数码值用于表示整体平移，而真正区分 token 的变化量只能使用更稀的网格。WV 误差由此增大。Attention probability 虽然非负，但简单改为 U8 只改变局部表示范围，不保证完整 Attention、residual 和后续 Block 的误差单调下降。

**解决方案。** 只沿 token 维中心化 Value，不平移 Attention：

~~~text
Vmean = mean_token(V)
Vc    = V - Vmean
A @ V = A @ Vc + sum_token(A) * Vmean
~~~

`A @ Vc` 使用逐行动态 S8，均值补偿以 FP16 加回。实现中采用实际量化后 Attention 的行和，不假设 Softmax 行和在有限精度下严格等于 1。该分解保持原始浮点公式，同时缩小进入 S8 MatMul 的 Value 动态范围。

没有采用 Attention 与 Value 双中心化，因为它需要更多补偿项，Converted BC 对补偿路径的取整更敏感。FP16 WV 在转换后出现 `native::MatMul` CPU fallback，A16xA8 也未通过目标后端转换，因此两者只用于定位，不属于最终 BPU 方案。

**验证依据。** 固定动态 QK/WV 后，未中心化 Eager 的 mean/min cosine 为 `0.972189/0.937993`；只中心化 Value 后提高到 `0.980400/0.952326`。历史三图 S600 HBM 的 820 条结果为：

| 指标 | 结果 |
|---|---:|
| Float -> HBM mean cosine | `0.971394` |
| P05 cosine | `0.957296` |
| Min cosine | `0.920964` |
| cosine < 0.95 | `14/820` |

整体均值达到当时采用的 0.95 数值检查门槛，但尾部样本仍然存在，不能表述为每一条样本都通过。这 820 条输入同时参与历史量化范围采集，因此该结果主要验证量化与设备执行链，不属于独立保留集上的泛化评估。

### 5.7 PBD 转入 AR 后的误差会沿生成历史放大

**问题表现。** 普通 PBD 样本的 Float 与量化坐标大多接近，较大的坐标差却集中在至少一侧进入 AR 的样本。这个现象容易被解释成“PBD 已经停止，AR 为什么还受 PBD 影响”，或被误判为 PBD 到 AR 的实现接错。

**原因分析。** PBD 停止的是后续并行草稿，不是整个生成请求。原始状态机保留已经确认合法的 token，只丢弃未接受草稿及其无效 K/V。AR q=1 从最后一个确认 token 继续生成当前结构，因此仍然继承相同的文本历史、position、mask 和有效 KV cache。它不会从 `<box>` 重新生成整个框。

当 Float 与量化 logits 在低 margin 位置选择了不同 token，两条路径从该步开始拥有不同输入和 KV。此后的差异包含两部分：当前步量化误差，以及不同历史经自回归反馈产生的放大。直接比较已经分叉的后续 logits，无法定位最初是哪一个算子或哪一步造成了变化。两条路径进入不同 PBD/AR 分支也不必然表示量化结果错误，因为自由生成可能选择不同但合法的目标。

**解决方案。** Float 与量化路径分别维护独立 Prefill cache。定位单步误差时，固定相同的历史 token、position、mask 和 KV，以 q=1 逐 token 回放已接受前缀，并记录完整词表的 rank、Top-K、margin 和坐标候选。该方法比较的是同一条件下的下一步分布，能够区分量化数学、状态切换和自回归放大。绝对正确性另行与 Ground Truth 比较。

AR 定向试验只改变一个局部模块，并保持其他权重、cache 和解码规则不变。WV Float 是最敏感的单项，但 QK、KV、全 Attention、全 Linear 和 `lm_head` 的局部恢复均未单独消除尾部。因此 WV 是优先排查位置，不是已经证明的唯一根因。

**验证依据。** 100 条 Layout 一致性审计中，IoU 小于 0.5 的 43 个框有 37 个位于至少一侧进入 AR 的路径。非 AR 子集包含 1080 个框，Float 与量化结果的 mean IoU 为 `0.98008`；双 AR 子集包含 292 个框，mean IoU 为 `0.89192`。该审计为了定位量化数学，将 Attention QK/WV/KV 切换为逐行 Dynamic A8；当前编译图在这些位置使用校准得到的固定 A8。因此这些数字只能说明诊断路径中的 AR 误差更集中，不能作为当前 HBM 的精度报告，也不是 Ground Truth 准确率。

### 5.8 按生成顺序配框会制造虚假的坐标误差

**问题表现。** 多目标图片中，Float 的第 N 个框与量化模型的第 N 个框可能相差很大。按序计算 IoU 会把目标顺序变化、漏检、增检和真正的坐标漂移混为一类，从而夸大量化误差。

**原因分析。** LocateAnything 输出的是一个目标集合，但通过自回归序列表示。目标的生成顺序不是训练任务规定的稳定主键。同一图片上，两次推理可以先后识别不同实例；只要它们分别与标注中的合法目标匹配，就不能仅因序列位置不同判错。Float 输出本身也不是 Ground Truth，用它作为唯一参照会把 Float 的漏检或随机性固化成“正确答案”。

**解决方案。** 数值回归和任务精度采用两套比较：

1. Float、量化仿真、BC 和 HBM 之间，在固定输入及固定前缀下比较 logits、KV、结构和坐标差，用于定位部署引入的变化。
2. Float、量化仿真和 HBM 分别与同一 Ground Truth 比较，用于判断任务结果是否正确。

Box 任务先按标签限制候选，再进行一对一最大 IoU 匹配。IoU 达到阈值的配对计为 TP，未匹配预测计为 FP，未匹配标注计为 FN，由此计算 Precision、Recall 和 F1；matched IoU 只统计成功配对后的定位质量。Pointing 使用归一化距离和 PCK。OCR、Layout 还需先检查输出结构和标签是否合法。

**验证依据。** 100 条 Layout 的 Float 与量化一致性审计得到结构一致率 `99.23%`、Box mean IoU `0.95480`。该结果说明两条执行路径总体接近，却不能替代六任务 Ground Truth 评估。保存的上游自由生成结果也不作为 Ground Truth 使用。

### 5.9 生成长度上限污染验证结果

**问题表现。** 早期全量验证从旧元数据继承 `max_new_tokens=512`。目标较多的 Detection、OCR 和 Layout 样本在达到长度上限时停止，输出缺少闭合标签或结束 token。若直接统计格式有效率、框数量和 F1，这些人为截断会被错误计入模型失效。

**原因分析。** Prefill 长度、KV cache 长度和最大生成长度控制不同阶段。Prefill 1024 决定一次建立初始上下文时可容纳的视觉 token、模板和查询；KV cache 4096 决定推理过程中可以保留的总历史；`max_new_tokens` 是 Host 允许模型继续生成的 token 数。把旧的 512 生成上限当成模型固定能力，会在 KV 仍有空间时提前终止输出。

**解决方案。** 当前验证显式设置 `max_new_tokens=2048`，不再从旧校准元数据继承生成上限。报告必须记录停止原因；因长度上限终止的样本先标为截断并重新运行，不能与正常 `<|im_end|>` 结束的样本一起计算格式和任务指标。运行前分别检查 Prefill 实际长度、`prompt + generated` 的 cache 占用和 Host 生成预算。

**验证依据。** 当前固定配置为 Prefill 1024、KV cache 4096、生成上限 2048。显式生成上限消除了从旧元数据继承 512 后造成的提前停止条件；长样本是否完整仍由停止原因和最终结构逐条确认。该处理修正的是验证条件，不代表模型精度本身得到提升。

### 5.10 通用 VLM 运行时无法保持 LocateAnything 推理语义

**问题表现。** 通用 `vlm/libxlm` 可能直接拒绝模型类型，也可能成功加载 HBM、产生 BPU 活动和非零 logits，却输出乱码、无关标签或重复坐标。专用 CLI 的早期版本还会把 `find cat` 或中文问句直接送入模型，得到 `None` 或无关标签；完整任务提示则能够得到合法框。

**原因分析。** LocateAnything 不是通用图文聊天模型。它要求一组相互配套的执行规则：MoonViT 672x672 输入和 576 个视觉 token 注入、Language 一维 RoPE、PBD 专用 attention mask 与 position、已接受 token 的 KV 提交，以及 `<ref>`、`<box>`、坐标 token 和结束 token 的状态机。通用 Qwen-VL 路径使用另一套视觉编码、位置编码和生成循环，文件能够加载并不意味着上述规则得到执行。

提示词也属于模型输入协议。六类任务在训练时使用确定模板；未经模板化的自由文本不是等价输入。短中文提示失败不能据此归因于“中文校准数据不足”，因为校准只决定量化范围，不会把定位模型改造成聊天模型。

**解决方案。** 使用 LocateAnything 专用 Host 运行时直接通过 HBRT/HB DNN 调用 Vision、Prefill、PBD 和 AR 图。Host 负责图像 letterbox、patchify、Tokenizer、视觉 token 替换、position、mask、KV ring、采样和空间 token 解析。CLI 的 `/detect`、`/ground`、`/gui`、`/text`、`/layout` 和 `/point` 只把简短命令展开为模型原始任务模板；当前版本拒绝没有任务类型的普通自由文本，不再把它作为定位请求执行。

正确性按层检查：原始 Float 与适配 Float 验证模型改写，Float 与 Q/DQ 验证量化数学，Exported BC 与 Converted BC 验证图转换，Converted BC 与 S600 HBM 验证设备执行，各阶段再分别与 Ground Truth 计算任务指标。HBM 加载成功、BPU 利用率变化和 logits 非零，只能证明图执行过。

**验证依据。** 专用运行时在固定 cat 图片和完整 Detection 模板上，可以重复输出合法的 `<ref>cat</ref><box>...</box>` 序列，并将 `[0,1000]` 坐标正确还原到原图像素。相同模型在缺少任务模板时输出无关标签，说明该样例的问题位于输入和解码协议，而不是由中文权重量化直接造成。这是一项功能验证，不替代六任务 Ground Truth 结果。

### 5.11 Host 数据搬运和重复 KV 提交限制了 Hybrid 加速

**问题表现。** Prefill 的 BPU 等待时间只有百毫秒量级，早期界面却多出约 1.27 秒；q=1 单 token 仍需数十毫秒。CPU 和 BPU 的平均利用率都不高，Hybrid 也没有达到 q=6 表面上对应的 2 至 3 倍端到端加速。

**原因分析。** 低平均利用率不等于存在一段可直接并行化的空闲计算。自回归生成存在严格的数据依赖，下一轮必须等待上一轮 logits、采样结果和 KV 更新。每轮还包含输入打包、cache flush、图提交、BPU 等待和输出回读，短图之间的同步间隙会降低监控工具显示的平均利用率。

早期 Prefill 导出完整 `[1,1024,152681]` FP16 logits，约 298.2 MiB；72 个 INT8 KV 更新另占约 18 MiB，全部 Host 输出合计约 316.2 MiB。Host 为调试摘要复制并扫描整块 logits，形成主要额外时延。KV 每轮整体移动、Tokenizer 重复加载和全词表排序又增加了非 BPU 时间。

Hybrid 还有一个结构性限制。第一次 q=6 使用草稿 token 预测候选；状态机接受其中 k 个 token 后，三图实现必须再次执行这些真实 token，才能把正确 K/V 提交到历史。一次 q=6 预测并不等于六个 token 都已经成为有效 cache，因此不能只删除第二次调用。

**解决方案。** 已部署运行时采用以下处理：Prefill 只导出最后一个有效 logits row；Host 只物化需要的输出；KV 改为镜像环形缓冲；Tokenizer 常驻；PBD 使用固定容量 Top-K；q=1 使用四核 AR 图。这些修改减少了无效搬运和 Host 计算，不改变 token 选择规则。

进一步加速需要修改并重新编译 Language 图。q7 至 q12 图把“提交上一轮已接受的 k 个 token”和“计算下一组六槽 PBD 窗口”合并；q2 至 q5 图处理错误 box 的合法前缀转 AR。当前能够核实的是图定义、编译脚本和运行时状态机单元测试；真实 BC、融合 HBM 与 S600 时延尚未形成归档验证，因此不能写成已经取得板端加速结果。另一条可行路线是在 BPU 图内完成 token 选择和 KV 提交，减少 Host 与 BPU 的往返。

**验证依据。** 在固定图片和提示下，运行时优化后的缓存图片 slow q=1 为 `514.8 至 517.7 ms`，Hybrid 为 `418.0 至 418.7 ms`；四核 AR 的 q=1 BPU wait 约 `22.0 至 22.3 ms`，单 token 总时间约 `34.4 ms`。这些结果说明 Host 优化已经消除了主要的无效扫描，但 Hybrid 的剩余下限仍由 Prefill、串行图提交和重复 KV 提交共同决定。要继续降低到约 200 ms，必须改变编译图的数据流，而不是继续优化微小的 Host 循环。上述时延只来自固定 cat 图片和固定提示，并非通用性能基准；当前 CLI 默认采用 Hybrid，`slow` 保留为纯 q=1 AR 对照模式。

上述问题具有明确的依赖顺序：先保证 Float 模型改写等价，再统一隐藏空间和校准路径；随后验证 Q/DQ、BC 与 S600 数值；最后才讨论任务精度和性能。跳过前一层，会使后一层的结果失去可归因性。
