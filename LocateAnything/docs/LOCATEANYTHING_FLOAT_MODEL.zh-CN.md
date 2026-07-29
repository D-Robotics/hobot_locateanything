# LocateAnything 原始浮点模型架构

本文说明 LocateAnything-3B 的任务定义、视觉编码、图文融合、空间词表和 Parallel Box Decoding (PBD)。内容以原始模型配置、PyTorch 实现、处理器和生成代码为准，不引入量化或固定部署参数。

本文中的 Float 指未插入整数 Q/DQ 的原始计算语义，不表示所有算子都使用 FP32。当前 checkpoint 的 Vision 与 Language 配置均声明为 BF16。

```text
统一定位任务与生成式输出
  -> 六类任务共用一套 prompt、词表和模型权重
  -> 框、点和无目标结果统一写成 token 序列

  -> MoonViT 视觉编码与媒体序列化
       -> 动态图像网格与 14x14 patch
       -> 27 层 MoonViT 与 2x2 patch merger
       -> 多图和视频帧转换为视觉 token 序列

       -> Qwen2.5 Decoder 与空间 Token 协议
            -> 视觉特征替换 <IMG_CONTEXT> embedding
            -> 36 层 Decoder 完成图文 self-attention
            -> 专用坐标 token 表达框和点

            -> PBD q=6 与 Hybrid 生成状态机
                 -> 6-token 结构块训练与推理
                 -> 合法片段按实际长度接收
                 -> 异常框由 q=1 AR 从已验证前缀补全
```

## 1. 统一定位任务与生成式输出

### 1.1 六类定位任务

LocateAnything 将视觉定位写成条件生成问题。六类任务共用 MoonViT、Language Decoder、词表和输出解析器，任务差异由 prompt 与目标序列表示，不对应六个独立检测头。

| 任务 | 条件输入 | 生成结果 |
|---|---|---|
| 通用目标检测 | 一个或多个类别 | 类别名称与全部实例框 |
| 指代表达定位 | 自然语言描述 | 一个或多个对应区域 |
| GUI 元素定位 | 元素描述或操作意图 | 元素框或点 |
| 文本定位 | 全图文本或指定文本 | 文本内容与区域框 |
| 版面定位 | 标题、段落、表格、图片等类别 | 版面类别与区域框 |
| 点定位 | 目标描述 | 二维点坐标 |

多类别检测使用 `</c>` 分隔类别。版面定位复用检测格式，GUI 定位可按数据协议输出框或点。任务统一发生在文本条件和空间 token 层，视觉主干不切换任务分支。

### 1.2 空间结果的序列表示

模型没有独立的分类头、框回归头或 NMS。标签与位置均由 Language Decoder 生成：

```text
边界框: <ref>label</ref><box><x1><y1><x2><y2></box>
点坐标: <ref>label</ref><box><x><y></box>
无目标: <box>none</box>
```

一个答案可以连续包含多个 `<ref>...</ref><box>...</box>` 片段。输出顺序由生成过程决定，不等同于传统检测器按置信度排序后的候选框数组。

### 1.3 模型数据流

LocateAnything 由 MoonViT、视觉到语言投影器和 Qwen2.5 Decoder 组成：

```text
image / video frames
  -> image processor
  -> MoonViT
  -> 2x2 patch merger
  -> multimodal projector
  -> visual embeddings

prompt
  -> media placeholders + chat template
  -> tokenizer
  -> text embeddings

visual embeddings
  -> replace <IMG_CONTEXT> embeddings
  -> Qwen2.5 Decoder
  -> PBD q=6 and/or AR q=1
  -> label and coordinate tokens
```

视觉特征不是追加到文本序列末尾。处理器先创建与视觉 token 数量一致的 `<IMG_CONTEXT>`，模型再替换这些位置的 embedding。替换前后序列长度不变，图文融合由 Language self-attention 完成。

## 2. MoonViT 视觉编码与媒体序列化

### 2.1 图像网格与 Patch 输入

原始图像处理器使用动态分辨率。设处理后的图像尺寸为 `H' x W'`，处理顺序为：

1. 将图像转换为 RGB。
2. 当 patch 数超过 `in_token_limit` 时，按面积比例缩小图像。
3. 将高和宽调整为 `merge_kernel_size * patch_size` 的整数倍。
4. 将像素转换到 `[0,1]`，再按 `mean=(0.5,0.5,0.5)`、`std=(0.5,0.5,0.5)` 归一化。
5. 按 `14x14` 划分 patch，生成 `[N_patch,3,14,14]` 和二维网格 `(H'/14,W'/14)`。

当前处理器配置的 `in_token_limit` 为 `25600`。它限制预合并 patch 数，不是 Language 上下文长度。

### 2.2 MoonViT 编码器

Patch Embedding 使用 kernel 与 stride 均为 `14` 的 Conv2d，将每个 RGB patch 映射到 1152 维。输入同时加入两类二维位置信息：

- 可学习二维 Position Embedding，根据实际网格执行双三次插值；
- 共享于各层的二维 RoPE，在 Attention 中旋转 Q/K 的横纵坐标通道。

MoonViT 包含 27 个 pre-norm Transformer block。每层计算为：

```text
x
  -> LayerNorm
  -> QKV projection
  -> 2D RoPE on Q/K
  -> visual self-attention
  -> output projection + residual
  -> LayerNorm
  -> MLP 1152 -> 4304 -> 1152
  -> residual
```

Attention 使用 16 个 head，每个 head 为 72 维。多张图或多个视频帧的 patch 可以在同一个扁平 tensor 中连续存放，`cu_seqlens` 保证每个媒体网格只在自身序列内执行视觉 self-attention。27 层结束后再执行一次 LayerNorm。

### 2.3 Patch Merger 与多模态投影器

`patch_merger` 将空间相邻的 `2x2` patch 特征按通道拼接：

```text
4 x 1152 -> 4608
```

该操作是拼接，不是平均池化。对合法网格 `(H'/14,W'/14)`，进入 Language 的视觉 token 数为：

```text
N_visual = (H' / 14) * (W' / 14) / 4
```

每个 4608 维合并特征再经过：

```text
LayerNorm(4608)
  -> Linear(4608, 2048)
  -> GELU
  -> Linear(2048, 2048)
```

最终接口是 `[N_visual,2048]`。这与 Language hidden size 一致，因此视觉特征可直接替换占位 token 的 embedding。

### 2.4 多图与视频序列化

处理器使用 `<image-N>` 和 `<video-N>` 标记媒体位置。单张图被展开为：

```text
<image N><img><IMG_CONTEXT> ... <IMG_CONTEXT></img>
```

占位数量等于该图的 `N_visual`。多图输入为每张图分别建立占位区，再将所有 patch tensor 与网格信息按媒体顺序拼接。

视频先按目标帧率和最大帧数采样。当前默认目标帧率为 2 FPS，默认最多采样 64 帧；这些值可由输入配置覆盖。每帧继续复用同一图像处理器和 MoonViT。文本模板为帧加入 `Frame-N`，存在时间戳时写入 `Frame-N-timestamp`，随后插入该帧的视觉占位区。

原始模型没有独立的时序视觉编码器。跨图和跨帧关系由媒体标记、统一 token 序列和 Language self-attention 表达。坐标 token 本身不包含媒体 ID，调用侧需依据模板、输出约定或后处理保留媒体对应关系。

## 3. Qwen2.5 Decoder 与空间 Token 协议

### 3.1 Language Decoder

Language 部分基于 Qwen2.5-3B 的 decoder-only Transformer。每层保持 Qwen2 结构：

```text
hidden states
  -> RMSNorm
  -> GQA self-attention + residual
  -> RMSNorm
  -> SiLU-gated MLP + residual
```

主要参数如下：

| 参数 | 数值 |
|---|---:|
| Hidden size | `2048` |
| Decoder layers | `36` |
| Query heads | `16` |
| KV heads | `2` |
| Head dimension | `128` |
| MLP intermediate size | `11008` |
| Vocabulary size | `152681` |
| Maximum position embeddings | `32768` |
| RoPE theta | `1000000` |
| Tied input/output embedding | `true` |
| Sliding-window attention | disabled |

GQA 的 K/V head 少于 Query head，Attention 前按组扩展到 Query head 数。Language 使用一维 RoPE。`use_sliding_window=false` 表示原始配置未启用滑动窗口；PBD 块内可见性由专用 Attention Mask 决定，与 sliding window 无关。

### 3.2 视觉 Token 注入

模型先查询完整 `input_ids` 的 token embedding，再选择 `input_ids == image_token_index` 的位置，以投影后的视觉特征逐项替换：

```text
image_token_index = 151665
count(<IMG_CONTEXT>) = count(visual embeddings)
```

文本、特殊 token 和视觉 token 随后位于同一个 2048 维 residual stream 中。36 层 Decoder 同时处理这些位置，没有独立 cross-attention。

多图输入不等于模型 batch 大于 1。当前自定义 `generate()` 实现要求 `batch_size=1` 和 `use_cache=True`，一个请求内仍可包含多张图或多个视频帧。

### 3.3 空间词表与坐标还原

当前 checkpoint 提供 1001 个坐标 token，表示整数 `0..1000`：

| 语义 | Token ID |
|---|---:|
| Image context | `151665` |
| Box start / end | `151668` / `151669` |
| Ref start / end | `151672` / `151673` |
| Text mask | `151676` |
| Coordinate `<0>` ... `<1000>` | `151677` ... `152677` |
| PBD null padding | `152678` |
| `none` | `4064` |

坐标 token 的词表 ID 与坐标值不是同一个数。设 `token_id` 位于坐标区间，则：

```text
v = token_id - coord_start_token_id

x_px = v_x / 1000 * original_image_width
y_px = v_y / 1000 * original_image_height
```

`none_token_id=4064` 表示无目标输出中的 `none`。`null_token_id=152678` 是 PBD 结构块的补齐 token，也参与 PBD 终止判定，不表示无目标。两者不能共用解析逻辑。

## 4. PBD q=6 与 Hybrid 生成状态机

### 4.1 六位置结构块

标准边界框由六个结构位置组成：

```text
<box>, <x1>, <y1>, <x2>, <y2>, </box>
```

因此当前 `block_size=6`。PBD 一次预测一个框的六个位置，不是一次生成图中的所有框。点结构只有四个有效位置：

```text
<box>, <x>, <y>, </box>
```

状态机识别到 `</box>` 后只接收前四个 token，其余预测不进入生成历史。

### 4.2 PBD 训练块

训练数据构造从一个已知 anchor token 开始，读取其后最多 6 个目标 token。若提前遇到 `</ref>`、`</box>` 或回复结束位置，当前目标块在该处结束；不足 6 位的目标以 `<null>` 补齐。

输入块与监督块分别为：

```text
input : [anchor, <text_mask>, <text_mask>, <text_mask>, <text_mask>, <text_mask>]
target: [next_1, next_2, next_3, next_4, next_5, next_6]
```

Position IDs 与原输出位置对齐。已确认前缀保持因果关系，当前块能够读取已确认前缀。当前配置 `causal_attn=false`，因此六位置块内部双向可见，使四个坐标在同一次前向中交换信息；当前块不能读取后续未确认块。

PBD 的 q=6 由训练目标、位置编码和 Attention Mask 共同定义，不等价于把普通 AR 图的 query length 从 1 直接改成 6。

### 4.3 PBD 推理与片段接收

PBD 推理复制最后一个已确认 token 作为 anchor，并附加 5 个 `<text_mask>`，构造长度为 6 的预测窗口。Language Decoder 一次返回六个位置的 logits，结构解析器再决定实际接收长度。

| 预测结构 | 接收结果 |
|---|---|
| 结束 token 或首位 `<null>` | 生成结束 |
| `<box> none </box>` | 接收 3 个 token |
| `<box> x1 y1 x2 y2 </box>` | 接收 6 个 token |
| `<box> x y </box>` | 接收 4 个 token |
| `<ref> ...` 文本片段 | 截止 `<null>` 或有效片段末尾 |
| 框结构不完整 | Hybrid 接收合法前缀并切换 AR |

框坐标位置只从坐标词表中选择。Hybrid 还检查低置信且候选跨度过大的坐标歧义。结构合法性与坐标候选共同决定当前块是否直接提交。

### 4.4 Fast、Slow 与 Hybrid 模式

原始生成实现提供三种模式：

| 模式 | 生成路径 | 异常框处理 |
|---|---|---|
| `fast` | 始终使用 PBD q=6 | 不切换 AR，保持块生成 |
| `slow` | 始终使用 AR q=1 | 不使用 PBD |
| `hybrid` | 默认使用 PBD q=6 | 对当前异常框切换 AR q=1 |

Hybrid 状态机为：

```text
已验证历史
  -> PBD q=6
       -> 合法结构: 接收有效片段，继续 PBD
       -> error_box:
            保留当前块中已验证的 <box> 和连续合法坐标
            丢弃其余并行预测
            切换 AR q=1
            生成到 </box>
            恢复 PBD q=6
```

AR 接管后，PBD 不再决定后续 token。回退也不会从 `<box>` 之前重新生成整个答案。它从当前块最后一个已验证 token 继续补全当前框。

### 4.5 KV 提交语义

PBD 前向会计算整个六位置窗口的 KV，但生成循环在采样前把 cache 截断到旧的已确认历史长度。未接受的并行预测不会直接进入持久 KV。

结构解析完成后，合法片段被追加到 `generated`。下一次 PBD 或 AR 调用重新处理这些已接受 token，并把它们写入因果历史。发生 `error_box` 时，AR 因此从已验证前缀重新计算当前局部片段，而不是复用无效 PBD 窗口的 KV。

AR 生成 `</box>` 后恢复 PBD。Hybrid 的 AR 阶段若生成既不是坐标、`none`、也不是 `</box>` 的 token，当前实现终止本次生成，避免不完整框继续扩散。

Attention backend 只负责执行上述可见性关系。它可以改变计算效率，但不改变模型权重、空间词表、PBD 接收规则或 Hybrid 状态机。

LocateAnything 的原始 Float 语义由四项约束共同确定：媒体顺序与视觉网格正确，MoonViT 输出与 `<IMG_CONTEXT>` 一一对应，空间 token 按 `0..1000` 解析，PBD 与 AR 只共享已确认的生成历史。量化、静态图编译和设备运行均需保持这四项语义。
