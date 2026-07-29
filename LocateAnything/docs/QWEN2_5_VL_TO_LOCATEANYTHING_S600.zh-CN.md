# 从 Qwen2.5-VL-3B 到 LocateAnything-3B：浮点架构、量化原理与 S600 部署

本文从模型本身开始，而不是从编译命令开始。阅读顺序固定为：

~~~text
Qwen2.5-VL-3B 浮点架构
  -> LocateAnything-3B 浮点架构
  -> 两个模型的差异与可复用边界
  -> Vision 中量化哪些权重和激活
  -> Language 中量化哪些权重和激活
  -> Scale 如何计算和共享
  -> Q 域旋转、Value 中心化等方法解决什么问题
  -> 如何逐层排查量化误差
  -> 如何生成 BC、HBO、HBM 并在 S600 验证
~~~

本文中的模型配置、量化方案和验证状态以 2026-07-28 的当前仓库为准。历史实验用于解释方法选择，不会被写成当前正式配置。

配套资料：

- [Qwen2.5-VL-3B S600 基线](../../Qwen-2.5-VL-3B/docs/QWEN2_5_VL_BASELINE.md)
- [LocateAnything 原始浮点架构](LOCATEANYTHING_FLOAT_MODEL.zh-CN.md)
- [LocateAnything OELLM 部署记录](LOCATEANYTHING_OELLM_DEPLOYMENT.zh-CN.md)
- [OELLM 算子算法](OELLM_OPERATOR_ALGORITHMS.zh-CN.md)
- [S600 运行与同步](S600_RUNTIME.md)

## 第一部分：从浮点模型架构开始

### 1. Qwen2.5-VL-3B 的整体架构

Qwen2.5-VL-3B 是一个 Vision Encoder 加 Decoder-only Language Model 的多模态模型。它没有单独的 Cross-Attention 模块，视觉特征通过图像占位 Token 注入 Language 序列。

~~~text
图像
  -> Qwen Vision Transformer
  -> Spatial Merger
  -> 2048 维视觉特征

文本
  -> Tokenizer
  -> 2048 维 Token Embedding

视觉特征替换 image placeholder 的 Embedding
  -> 多模态 inputs_embeds
  -> Qwen2.5 Language Decoder
  -> logits
  -> AR q=1 逐 Token 生成
~~~

这里有三个必须保持一致的接口：

| 接口 | 要求 |
|---|---|
| Vision -> Language | 最后一维都为 2048，且处于相同隐藏坐标域 |
| Tokenizer -> Embedding | Token ID、Vocabulary 和 Embedding 行必须一致 |
| Language -> Runtime | position、mask、KV cache 和采样规则必须一致 |

张量 shape 一致只是必要条件。隐藏域、Token 顺序或位置编码错误时，模型仍会输出非零 logits，但语义已经不可信。

### 2. Qwen2.5-VL-3B 的 Vision

受控基线固定输入为 448x448：

~~~text
448x448 RGB
  -> 14x14 Patch
  -> 32x32 = 1024 个 Patch
  -> Patch Embedding: 588 -> 1280
  -> 32 个 Vision Transformer Block
  -> 2x2 Spatial Merge
  -> 256 个视觉 Token
  -> Projector: 5120 -> 2048
~~~

其中：

~~~text
588 = 3 x 14 x 14
~~~

Qwen Vision 的主要参数：

| 项目 | 数值 |
|---|---:|
| Vision hidden | 1280 |
| Blocks | 32 |
| Window Attention | 28 层 |
| Full Attention | 4 层 |
| Full Attention 层号 | 7、15、23、31 |
| Patch 数 | 1024 |
| Merge 后 Token | 256 |
| Language hidden | 2048 |

#### 2.1 Patch Embedding

原始 Qwen2.5-VL Patch Embedding 是 Conv3D，包含 temporal kernel。静态图片使用两个相同 temporal slice，可以将时间维权重离线求和：

~~~text
y = x * W0 + x * W1
  = x * (W0 + W1)

W2D = sum_temporal(W3D)
~~~

这是一项有前提的浮点等价改写：两个 temporal 输入必须相同。它只适用于当前 Qwen 静态图片 profile，不能照搬到 MoonViT。

#### 2.2 Window Attention 与 Token 顺序

大部分 Vision Block 只在局部 Window 内计算 Attention，少量 Full Block 建立全局联系。内部会按 window_index 重排 Patch。

Vision HBM 输出仍遵循编译图内部顺序。注入 Language 前，Host 必须使用 reverse index 恢复文本占位位置所期待的视觉 Token 顺序。

因此下面两种错误都可能产生正常 shape：

~~~text
正确特征 + 错误 Token 顺序
错误特征 + 正确 Token 数量
~~~

它们都不能通过“输出不是零”来发现。

### 3. Qwen2.5-VL-3B 的 Language

Language 是 36 层 Decoder-only Transformer：

| 参数 | 数值 |
|---|---:|
| Hidden size | 2048 |
| Layers | 36 |
| Query heads | 16 |
| KV heads | 2 |
| Head dimension | 128 |
| MLP intermediate | 11008 |
| Vocabulary | 151936 |

每层由两个大子层组成：

~~~text
Residual Stream
  -> RMSNorm
  -> Self-Attention
  -> Residual Add
  -> RMSNorm
  -> Gated MLP
  -> Residual Add
~~~

#### 3.1 Language Attention

一层 Attention 的浮点顺序是：

~~~text
Xn = RMSNorm(X)

Q = Xn @ Wq.T + bq
K = Xn @ Wk.T + bk
V = Xn @ Wv.T + bv

Q', K' = RoPE(Q, K, position)

S = Q' @ K'.T / sqrt(head_dim)
S = S + attention_mask
A = softmax(S)

C = A @ V
O = C @ Wo.T
Y = X + O
~~~

Q 和 K 决定“关注谁”，V 携带“取回什么内容”，Softmax 后的 A 是 Attention Weight。

#### 3.2 Language MLP

Qwen2.5 的 Gated MLP 为：

~~~text
Xn = RMSNorm(X)

G = Xn @ Wgate.T
U = Xn @ Wup.T
M = SiLU(G) * U
D = M @ Wdown.T

Y = X + D
~~~

它不是普通的单路 Linear -> Activation -> Linear。Gate 和 Up 两条分支相乘，任一分支的量化误差都会进入乘积。

#### 3.3 三轴 MRoPE

Qwen2.5-VL 使用三轴 MRoPE：

~~~text
t：时间或文本序列轴
h：图像高度轴
w：图像宽度轴
~~~

Q/K 的旋转通道被分配给不同坐标轴。两个 Token 的 Attention 内积最终包含相对时间差、相对行差和相对列差。

文本 Token 没有独立二维位置，三个轴可以共享文本顺序位置；图像或视频 Patch 则使用各自的 t、h、w 坐标。

Qwen 基线 Language HBM 包含：

~~~text
prefill：建立初始 KV cache
decode：q=1，自回归生成一个 Token
~~~

### 4. LocateAnything-3B 是怎样从 Qwen 主干发展出来的

LocateAnything 保留了 Qwen2.5 3B Language 主干，但把模型改造成统一空间定位模型。它不是换了提示词的 Qwen2.5-VL。

主要变化为：

~~~text
Qwen Vision
  -> MoonViT

通用图文生成
  -> Detection / Referring / GUI / OCR / Layout / Pointing

Qwen 三轴 MRoPE Language
  -> LocateAnything 一维 RoPE Language

普通 AR q=1
  -> PBD q=6 + AR q=1 回退

Vocabulary 151936
  -> Vocabulary 152681，加入空间和状态 Token
~~~

### 5. LocateAnything-3B 的 MoonViT Vision

当前发布 profile 使用 672x672 letterbox：

~~~text
原图
  -> 保持宽高比缩放
  -> 填充到 672x672
  -> 14x14 Patchify
  -> 48x48 = 2304 个 Patch
  -> Patch Linear: 588 -> 1152
  -> 27 个 MoonViT Block
  -> Final LayerNorm
  -> 2x2 Patch Merger
  -> 24x24 = 576 个视觉 Token
  -> Projector: 4608 -> 2048 -> 2048
~~~

关键参数：

| 项目 | 数值 |
|---|---:|
| Vision hidden | 1152 |
| Blocks | 27 |
| Heads | 16 |
| Head dimension | 72 |
| Vision 输入 | [1,2304,588] FP16 |
| Vision 输出 | [1,576,2048] FP16 |

当前 27 层全部使用全局 Attention，没有 Qwen Vision 的 Window/Full 切换、window_index 和 temporal Conv3D。

#### 5.1 MoonViT Block

MoonViT Block 的基本浮点顺序为：

~~~text
X
  -> LayerNorm
  -> QKV Linear
  -> 2D RoPE
  -> QK MatMul
  -> Softmax
  -> WV MatMul
  -> Output Linear
  -> Residual Add
  -> LayerNorm
  -> Linear
  -> GELU
  -> Linear
  -> Residual Add
~~~

Vision 与 Language 都有 Attention 和 MLP，但 Norm 类型、MLP 结构、位置编码、Head 维度和生成状态不同，不能因为算子名称相似就共用模型代码。

#### 5.2 二维 RoPE

2304 个 Patch 原本是一维存储，但它们对应 48x48 网格：

~~~text
x = patch_index % 48
y = patch_index // 48
~~~

Q/K 通道对交替使用 x 和 y 的旋转频率：

~~~text
pair 0 -> x frequency 0
pair 1 -> y frequency 0
pair 2 -> x frequency 1
pair 3 -> y frequency 1
~~~

原始复数旋转在 LEAP 中展开为实数运算：

~~~text
real' = real*cos(theta) - imag*sin(theta)
imag' = real*sin(theta) + imag*cos(theta)
~~~

RoPE 只改变 Q/K 的方向，不旋转 V。Attention 因此感知 Patch 之间的相对横向和纵向距离。

### 6. LocateAnything-3B 的 Language

LocateAnything Language 仍是 36 层、2048 hidden、16 Query Head 和 2 KV Head，但接口已经变化：

| 项目 | LocateAnything |
|---|---:|
| Vocabulary | 152681 |
| 坐标 Token | 1001 个，对应 [0,1000] |
| Position encoding | 一维 RoPE |
| Tied Embedding | true |
| Prefill chunk | 1024 |
| KV cache | 4096 |
| PBD query length | 6 |
| AR query length | 1 |

#### 6.1 为什么是 1024 Prefill

一张图片产生 576 个视觉 Token。Prefill 图固定容纳 1024 个位置：

~~~text
1024 - 576 = 448
~~~

剩余 448 个位置还要容纳系统模板、任务提示、查询和特殊 Token。这里的 1024 是固定 Prefill 图长度，不是原始模型的 RoPE 理论上限。

#### 6.2 为什么同时需要 q=6 和 q=1

一个 Box 结构包含六个位置：

~~~text
<box> x1 y1 x2 y2 </box>
~~~

PBD q=6 一次预测整组候选。状态机接受合法部分；结构不完整或候选不满足规则时，AR q=1 从最后一个已确认 Token 继续生成。

PBD 与 AR 共享：

~~~text
同一组 Language 权重
同一条有效 Token 历史
同一套 position 规则
同一个 KV cache 语义
~~~

它们只是不同固定 shape、不同 Mask 的编译图。

### 7. 两个模型的可复用边界

| 内容 | 可从 Qwen 复用 | 必须为 LocateAnything 重写 |
|---|---|---|
| LEAP/HBIR 描图方式 | 是 |  |
| BC/HBO/HBM 编译链 | 是 |  |
| HBRT 基本调用 | 是 |  |
| 分层数值验证方法 | 是 |  |
| 隐藏域正交重参数化方法 | 是 | 折叠位置必须按 LA 结构重新确定 |
| Vision 模型和 HBM |  | MoonViT |
| temporal 权重折叠 |  | LA 不使用该 Qwen 规则 |
| window_index |  | LA 当前全局 Attention |
| Language MRoPE |  | LA 使用一维 RoPE |
| Vocabulary |  | 必须保留 152681 |
| Decode Runtime |  | 必须支持 PBD q=6、AR q=1 和 KV 提交 |
| 任务后处理 |  | 必须解析空间 Token 和 inverse letterbox |

## 第二部分：量化到底量化哪些内容

### 8. 先区分权重、激活和累加结果

模型中的数值可分为三类：

| 类型 | 产生时间 | 例子 |
|---|---|---|
| 权重 Weight | 训练后固定 | Wq、Wk、Wv、Wo、Wgate、Wup、Wdown |
| 激活 Activation | 每次输入动态产生 | X、Q、K、V、Attention Weight、MLP 中间值 |
| 累加和输出 | MatMul/Conv 计算产生 | QK score、WV output、Linear output |

W8 表示权重以 8 bit 表示；A8 表示进入低比特计算的激活以 8 bit 表示。它们不是同一件事。

Bias、Norm、Softmax、Residual 和最终输出不一定全部强制成 INT8。编译器会根据算子合同保留 FP16/FP32 中间值或使用内部累加精度。不能把“模型是 W8A8”理解成图中的每一个张量都是 INT8。

### 9. 整体量化对象树

~~~text
LocateAnything
|
+-- Vision
|   |
|   +-- 权重
|   |   +-- Patch Embedding Linear
|   |   +-- 27 x QKV Linear
|   |   +-- 27 x Attention Output Linear
|   |   +-- 27 x MLP Linear 1 / Linear 2
|   |   +-- Patch Merger / Projector Linear
|   |
|   +-- Attention 激活
|   |   +-- Norm 输出
|   |   +-- Q / K / V
|   |   +-- RoPE 后 Q / K
|   |   +-- QK score
|   |   +-- Softmax Attention Weight
|   |   +-- WV output
|   |
|   +-- MLP 激活
|       +-- Linear 1 输入和输出
|       +-- GELU 输出
|       +-- Linear 2 输入和输出
|
+-- Language
    |
    +-- 权重
    |   +-- 36 x Q/K/V/O Projection
    |   +-- 36 x Gate/Up/Down Projection
    |   +-- lm_head
    |   +-- Token Embedding 当前保留 FP16
    |
    +-- Attention 激活
    |   +-- RMSNorm 输出
    |   +-- Q / K / V
    |   +-- RoPE 后 Q / K
    |   +-- KV cache
    |   +-- QK score
    |   +-- Softmax Attention Weight
    |   +-- WV output
    |
    +-- MLP 激活
        +-- Gate / Up 输出
        +-- SiLU 输出
        +-- SiLU(Gate) * Up
        +-- Down Projection 输入和输出
~~~

量化分析应沿这棵树逐层进行，而不是只说“Vision W8”或“Language A8”。

### 10. Vision 权重量化

当前 Vision Linear 使用 W8。对一个 Linear：

~~~text
Y = X @ W.T + b
W.shape = [out_features, in_features]
~~~

OELLM 当前按输出通道计算权重 Scale：

~~~text
weight_absmax[o] = max_k(abs(W[o,k]))
weight_scale[o]  = weight_absmax[o] / 127
qW[o,k]          = round(W[o,k] / weight_scale[o])
~~~

每一个输出通道有独立 Scale，避免整个大矩阵被单个异常通道共同拉宽范围。

Vision 中被 W8 化的主要权重：

| 模块 | 权重作用 |
|---|---|
| Patch Linear | 把 588 维 Patch 映射到 1152 hidden |
| QKV Linear | 生成每个 Head 的 Q、K、V |
| Output Linear | 把多 Head Attention 结果写回 residual |
| MLP Linear 1 | 1152 -> intermediate |
| MLP Linear 2 | intermediate -> 1152 |
| Merger/Projector | 把 4 个相邻 Patch 拼接后映射到 2048 |

权重量化是静态的：权重固定，因此 Scale 可在编译前计算并写入图，不需要校准图片决定 W8 Scale。

### 11. Vision 激活量化

#### 11.1 Linear 输入激活

Vision 的 DynamicQuantLinear 在运行时对输入最后一维逐行计算 Scale。

对于：

~~~text
X.shape = [B,T,D]
~~~

每个 Token 行使用：

~~~text
x_absmax[b,t] = max_d(abs(X[b,t,d]))
x_scale[b,t]  = x_absmax[b,t] / 127
~~~

因此同一张图片中，不同 Patch Token 可以使用不同 Scale。它比全图共享一个静态 Scale 更能适应 OCR、Layout 和自然图像之间的幅值差异。

#### 11.2 Attention 激活

Vision Attention 的关键路径：

~~~text
Xn
  -> QKV Linear
  -> Q, K, V
  -> 2D RoPE(Q,K)
  -> S = Q @ K.T
  -> A = softmax(S)
  -> C = A @ V
  -> Output Linear
~~~

当前正式方案：

| 位置 | 当前处理 |
|---|---|
| QKV Linear 权重 | W8 |
| QKV Linear 输入 | 逐行动态 A8 |
| QK MatMul 两侧 | 逐行动态 S8 |
| Softmax | 保持高层 Softmax 语义，由后端实现 |
| WV 的 Attention Weight | 逐行动态 S8 |
| WV 的 Value | Token 中心化后逐行动态 S8 |
| Value 均值补偿 | FP16 |
| Output Linear 权重/输入 | W8 + 逐行动态 A8 |

Attention 激活比普通 Linear 更敏感，因为 QK 决定概率分布，WV 决定对所有 Token 内容的加权求和。前者的小误差可能改变关注位置，后者的小概率和小 Value 会直接被舍入。

#### 11.3 Vision MLP 激活

Vision MLP 为：

~~~text
H1 = Linear1(Xn)
H2 = GELU(H1)
H3 = Linear2(H2)
Y  = X + H3
~~~

量化落点为：

| 位置 | 处理 |
|---|---|
| Linear1 权重 | W8 |
| Linear1 输入 Xn | 逐行动态 A8 |
| GELU | 浮点/后端近似激活 |
| Linear2 输入 H2 | 进入 Linear2 前重新逐行动态 A8 |
| Linear2 权重 | W8 |
| Residual Add | 恢复到 Block 输出数据类型后相加 |

这里没有要求 Linear1 输出与 GELU 输出永久保存为 INT8。DynamicQuantLinear 在每个低比特 MatMul 边界重新量化输入，MatMul 后恢复带 Scale 的浮点结果供后续算子使用。

### 12. Language 权重量化

一层 Language Decoder 有七个主要 Linear：

~~~text
Attention:
  q_proj
  k_proj
  v_proj
  o_proj

MLP:
  gate_proj
  up_proj
  down_proj
~~~

36 层共有：

~~~text
36 x 7 = 252 个 Decoder Linear
~~~

除此之外还有大词表投影 lm_head。

当前正式配置：

| 权重 | 配置 |
|---|---|
| 252 个 Decoder Linear | W8 |
| lm_head | W8 |
| Token Embedding | FP16 |
| Norm 权重 | 浮点，并可离线折叠到相邻 Linear |
| Bias | 不作为普通 W8 矩阵处理 |

Qwen 基线曾使用 Language W4，但 LocateAnything 最终选择 W8。原因不是 W4 不能运行，而是坐标输出依赖 1001 个坐标 Token 的 logits 排序。W4 的整数范围更小、网格更粗，容易在低 margin 时交换坐标候选顺序。

### 13. Language Attention 激活量化

Language Attention 的执行和量化位置：

~~~text
X
  -> RMSNorm
  -> Xn
  -> Dynamic A8 + W8 Q/K/V Projection
  -> Q, K, V
  -> 1D RoPE(Q,K)
  -> K/V cache 拼接
  -> Static A8 QK MatMul
  -> Softmax
  -> Static A8 WV MatMul
  -> Dynamic A8 + W8 O Projection
  -> Residual Add
~~~

当前正式编译路径：

| 激活位置 | 当前处理 |
|---|---|
| Q/K/V Projection 输入 | 逐行动态 A8 |
| RoPE 后 Q/K | QK MatMul 前使用校准固定对称 A8 |
| KV cache 输入/输出边界 | 校准固定对称 A8 |
| QK score | MatMul 后恢复并进入 Scale、Mask、Softmax |
| Softmax Attention Weight | WV MatMul 前使用校准固定对称 A8 |
| WV Value | WV MatMul 前使用校准固定对称 A8 |
| O Projection 输入 | 逐行动态 A8 |

当前 FakeQuantMatmul 的固定量化范围来自 ConstFakeQuant：校准期间记录该模块在全部样本上观察到的最大绝对值，导出时写成模块级对称范围。

诊断程序测试过动态 QK/WV/KV 和 WV-U8，但它们是 Eager A/B，不代表当前 HBM 已经改为动态 Attention A8。

#### 13.1 KV cache 为什么也是激活

K/V cache 不是模型权重。它由当前输入和历史 Token 动态生成：

~~~text
new K/V
  -> 写入 cache
  -> 后续 q=6 或 q=1 读取
  -> 与当前 Q 共同计算 Attention
~~~

如果 cache 量化发生饱和或分辨率不足，误差会跨 Decode 步持续存在。PBD 与 AR 必须使用相同的 cache 数值约定。

### 14. Language MLP 激活量化

Language Gated MLP：

~~~text
Xn = RMSNorm(X)

G = gate_proj(Xn)
U = up_proj(Xn)
A = SiLU(G)
M = A * U
D = down_proj(M)

Y = X + D
~~~

量化位置：

| 位置 | 当前处理 |
|---|---|
| gate_proj / up_proj 输入 Xn | 各自逐行动态 A8 |
| gate/up 权重 | W8 |
| SiLU 和逐元素乘法 | 浮点或后端支持的数据类型 |
| down_proj 输入 M | 进入 down_proj 前重新逐行动态 A8 |
| down_proj 权重 | W8 |
| Residual Add | Block 输出类型下相加 |

MLP 常见风险：

1. Gate 和 Up 分支幅值分布不同，共用错误 Scale 会损失其中一支；
2. SiLU 会产生小值和长尾；
3. 两支相乘会放大相对误差；
4. Down Projection 输入可能出现比 Xn 更大的动态范围；
5. 36 层 Residual 会累积小误差。

因此排查 Language 时，不能只检查 Attention。应分别记录 Gate、Up、SiLU、乘积和 Down 输出。

## 第三部分：Scale 是怎样计算出来的

### 15. 对称整数量化

对 b bit signed 整数：

~~~text
qmin = -2^(b-1)
qmax =  2^(b-1) - 1
~~~

S8 通常使用：

~~~text
[-128,127]
~~~

为了用同一个正负 Scale 对称覆盖浮点范围，常用：

~~~text
scale = absmax / 127
q     = clip(round(x / scale), -128, 127)
x_hat = q * scale
~~~

x_hat 是反量化近似值。量化误差来自：

~~~text
rounding error：落到最近整数网格
clipping error：超过范围后饱和到 qmin/qmax
zeroing error ：小于半个 Scale 的值可能变成 0
~~~

### 16. W8 和 W4 的 Scale

#### 16.1 W8

当前 DynamicQuantLinear 对每个输出通道独立计算：

~~~text
W.shape = [N,K]

w_absmax[n] = max_k(abs(W[n,k]))
w_scale[n]  = w_absmax[n] / 127
qW[n,k]     = round(W[n,k] / w_scale[n])
~~~

因此 W8 是 per-output-channel，而不是整个权重矩阵共用一个 Scale。

#### 16.2 W4

W4 只有大约 16 个整数码值：

~~~text
[-8,7]
~~~

Scale 近似为：

~~~text
w_scale[n] = w_absmax[n] / 7
~~~

同样范围下，W4 的步长约为 W8 的 18 倍：

~~~text
127 / 7 ≈ 18.14
~~~

这就是 W4 更省存储和带宽，但更容易改变小 margin logits 排序的直接原因。

### 17. 动态激活 Scale

DynamicQuantLinear 使用：

~~~text
x_q, x_scale = dynamic_quantize(x, blockSize=-1)
~~~

对 Linear 输入 [B,T,D]，最后一维 D 共享一个 Scale：

~~~text
x_scale[b,t,1] = max_d(abs(x[b,t,d])) / 127
~~~

于是：

~~~text
x_q.shape     = [B,T,D]
x_scale.shape = [B,T,1]
~~~

不同 Token 行各自适配范围。对 Attention MatMul，仍然沿张量最后一维生成 Scale，但“这一行”可能对应 Q 的一个 Token 向量、Attention 的一行概率，或转置后 K 的一行，取决于当时的布局。

动态量化的运行步骤：

~~~text
abs
  -> reduce_max
  -> 除以 127
  -> 数值保护
  -> reciprocal
  -> x / scale
  -> round-even
  -> saturate to S8
~~~

它减少跨样本范围冲突，但增加运行时 Reduce、Scale 计算和量化操作。

### 18. 静态激活 Scale

ConstFakeQuant 在校准前初始化：

~~~text
absmax = 0
~~~

每次真实前向：

~~~text
current = max(abs(x))
absmax  = max(absmax, current)
~~~

校准结束后：

~~~text
scale = absmax / 127
~~~

这个 Scale 固定写入 Exported BC。当前 ConstFakeQuant 是一个量化位置一个全局 absmax，并不是每张图片或每个 Token 单独保存 Scale。

优点：

~~~text
运行时不重新统计
图更简单
开销更低
~~~

风险：

~~~text
少数极值把 Scale 拉大
常见小值只能使用很少的整数码
未知输入超过校准范围时发生饱和
~~~

### 19. 两个输入的 MatMul 如何恢复 Scale

假设：

~~~text
qx ≈ x / sx
qy ≈ y / sy
~~~

整数 MatMul：

~~~text
acc = qx @ qy
~~~

反量化近似：

~~~text
out_hat = acc * sx * sy
~~~

如果 sx 或 sy 粒度不同，编译器需要按照对应行、通道或 Block 广播 Scale。很多量化错误不发生在整数乘法本身，而发生在 Scale 的 shape、布局、广播轴或恢复顺序。

### 20. U8 与 S8

Softmax Attention Weight 满足：

~~~text
A in [0,1]
~~~

用 S8 对称量化会浪费负半轴。理论上 U8 可使用：

~~~text
q in [0,255]
scale = max(A) / 255
~~~

同一非负范围可以获得更密的网格。

但“局部表示更细”不等于“整网精度一定更好”。当前 Language WV-U8 Eager A/B 有局部收益，也有样本退化；HBDK/S600 的 U8 x S8 支持和端到端收益未形成正式部署证据，因此 U8 仍是诊断候选。

## 第四部分：量化优化方法与解决的问题

### 21. 先区分 Value 中心化和 Q 域旋转

这两个方法经常被混在一起，但它们完全不同：

| 方法 | 做什么 | 作用位置 |
|---|---|---|
| Value 中心化 | 减去 Token 维均值，再补偿回来 | Vision Attention 的 WV |
| 正交 Q 域旋转 | 整个 2048 维隐藏向量更换正交坐标基 | Vision/Embedding/Language 边界和 residual stream |

Value 中心化真的执行减均值；Q 域旋转不减均值，所以不能称为“Q 域中心化”。

另外，Q 域矩阵 Q_domain 也不是 Attention 中的 Query 张量 Q_attn。本文后面用 R 表示正交域矩阵，避免混淆。

### 22. Vision Value 中心化

Vision WV：

~~~text
O = A @ V
~~~

其中：

~~~text
A.shape = [B,H,T,T]
V.shape = [B,H,T,D]
~~~

沿 Token 维计算：

~~~text
Vmean = mean_token(V)
Vc    = V - Vmean
~~~

因为：

~~~text
V = Vc + Vmean
~~~

所以：

~~~text
A @ V
= A @ Vc + sum_token(A) * Vmean
~~~

部署时：

~~~text
主 MatMul：Quant(A) @ Quant(Vc)
补偿项  ：sum_token(Quant(A)) * Vmean
~~~

补偿使用实际 Attention 行和，不直接假设有限精度下严格等于 1。

#### 22.1 它解决什么问题

如果 V 的一行范围为：

~~~text
[-2,6]
~~~

S8 对称量化必须按 absmax=6：

~~~text
scale = 6 / 127
~~~

假设均值约为 2，中心化后：

~~~text
Vc ≈ [-4,4]
scale = 4 / 127
~~~

步长变小，Token 之间的差异使用更多整数码表示。公共偏移没有丢失，而是在补偿项中加回。

#### 22.2 为什么没有直接使用 FP16 WV

FP16 WV 数值更接近 Float，但当前 HBDK Convert 出现 native::MatMul CPU fallback。A16 x A8 候选也没有形成稳定的 Nash-P BPU 图。

因此最终选择可部署的 Centered-Value S8，而不是数值更高但落到 CPU 的方案。

#### 22.3 当前证据

同一动态 QK/WV 基础上：

| 方案 | Eager mean cosine | Eager min cosine |
|---|---:|---:|
| 未中心化 | 0.972189 | 0.937993 |
| 只中心化 V | 0.980400 | 0.952326 |

最终 S600 820 张：

| 指标 | 结果 |
|---|---:|
| Float -> HBM mean cosine | 0.971394 |
| P05 cosine | 0.957296 |
| Min cosine | 0.920964 |
| cosine < 0.95 | 14/820 |

这证明整体平均达到当前 0.95 门槛，不代表每张图片都超过 0.95，也不等于独立测试集泛化精度。

### 23. 正交 Q 域旋转

设原始 residual hidden 为行向量 h，选择正交矩阵 R：

~~~text
R.T @ R = I
h'      = h @ R
~~~

正交变换保持：

~~~text
||h'||2 = ||h||2
(h1 @ R) · (h2 @ R) = h1 · h2
~~~

所以它不改变浮点空间中的长度和内积。

#### 23.1 为什么可能有利于量化

原始隐藏空间可能存在少数幅值特别大的通道：

~~~text
某些 channel 很大
多数 channel 很小
~~~

对称量化 Scale 被大通道控制，小通道只能落到少量整数码。Hadamard 类正交变换可以把通道能量重新混合，使异常能量不再长期集中在固定通道。

它不保证每个输入的 absmax 一定下降，因此必须通过量化 A/B 验证，而不能只根据正交性判断收益。

#### 23.2 为什么必须折叠所有边界

如果：

~~~text
h' = h @ R
~~~

下一个 Linear 原本为：

~~~text
y = h @ W.T
~~~

要保持 y 不变，需要同步改写权重。当前工程将 R 离线折叠到：

~~~text
Token Embedding
Vision Projector
Language Q/K/V 输入投影
Language MLP Gate/Up 输入投影
Attention O Projection
MLP Down Projection
Final RMSNorm / lm_head
~~~

运行时不会新增 2048x2048 MatMul。

只旋转 Vision 输出、不旋转 Language 输入，或只旋转 Embedding、不旋转 lm_head，都会破坏隐藏空间接口。正交矩阵保持几何关系的前提是相邻模块使用同一坐标约定。

#### 23.3 Q 域和 Attention Query 的区别

~~~text
R / Q_domain：2048x2048 隐藏坐标变换
Q_attn       ：[B,H,T,D] Attention Query 激活
~~~

它们名字相似，但维度、作用和生命周期完全不同。

### 24. RMSNorm 数值保护

RMSNorm：

~~~text
rms(x) = sqrt(mean(x^2) + eps)
y      = weight * x / rms(x)
~~~

FP16 计算 x^2 和归约可能溢出。OELLM 根据校准期间观察到的最大平方和选择保护系数 s：

~~~text
z    = x / s
eps' = eps / s^2
~~~

因为：

~~~text
z / sqrt(mean(z^2) + eps')
= x / sqrt(mean(x^2) + eps)
~~~

z 只在 RMSNorm 内部使用。Residual 分支仍保留原始 x，因此下一个 Block 不会收到被缩小 s 倍的状态。

它解决的是 FP16 数值溢出，不是低比特量化分辨率问题。

### 25. 动态 A8、静态 A8 与混合精度怎么选择

| 方案 | 适合的问题 | 代价 |
|---|---|---|
| 静态 A8 | 分布稳定、运行开销敏感 | 依赖校准覆盖，容易受极值影响 |
| 动态 A8 | 样本和 Token 范围变化大 | 增加 Reduce 和 Scale 计算 |
| W8 替代 W4 | 权重误差影响 logits 排序 | 模型更大、带宽增加 |
| U8 | 输入天然非负 | 需要后端支持和端到端验证 |
| FP16 局部恢复 | 定位敏感算子 | 可能 CPU fallback 或性能下降 |
| Value 中心化 | WV 的公共偏移浪费 S8 范围 | 增加均值和补偿计算 |
| 正交 Q 域 | 隐藏通道能量不均 | 必须全链路一致折叠并重新校准 |

选择顺序应为：

~~~text
先用 Float rescue 定位敏感位置
  -> 再用单变量 Eager A/B 验证候选
  -> 再检查 Converted BC 是否全 BPU
  -> 最后在 S600 做全量数值和任务验证
~~~

不能因为某个 FP16 探针 cosine 更高，就跳过 CPU fallback 检查直接采用。

## 第五部分：校准、排查与部署

### 26. 校准在计算什么

校准不是训练，不更新模型权重。它执行真实前向并记录：

~~~text
每个 ConstFakeQuant 的最大绝对值
静态激活 Scale
RMSNorm 数值保护统计
Observer 覆盖
输入 profile 和数据版本
~~~

当前数据职责：

~~~text
JSON/JSONL：任务、路径、标注和追溯信息
PT        ：张量加元数据和参考结果
NPY       ：连续数值输入，方便 C++/HBRT 读取
~~~

模型消费的是数值张量，不要求“校准数据必须是 JSON”。

LocateAnything 校准覆盖六类任务：

~~~text
Detection
Referring
GUI
OCR
Layout
Pointing
~~~

当前记录中 Vision 108/108、Language 289/289 个统计位置被执行。覆盖率只证明统计路径运行过，不证明 Scale 最优或任务精度通过。

### 27. 最有效的量化排查顺序

#### 27.1 第一步：确认浮点架构

~~~text
Original Float -> Adapted OELLM Float
~~~

Vision 检查：

~~~text
Input
Patch Embedding
每个 MoonViT Block
Final LayerNorm
Patch Merger
Projector
Final Output
~~~

Language 检查：

~~~text
Input Embedding
每层 RMSNorm
Attention
MLP
KV
Final Norm
lm_head logits
PBD/AR token
~~~

浮点已经不一致时，不讨论 Scale 和校准集。

#### 27.2 第二步：权重与激活分离

对同一输入执行单变量 A/B：

~~~text
Float
Weight-only quantization
Activation-only quantization
Weight + Activation quantization
~~~

如果 Weight-only 已明显下降，优先检查 W4/W8、per-channel Scale 和权重布局。

如果 Activation-only 下降，继续拆分：

~~~text
Attention activation only
MLP activation only
KV activation only
lm_head input only
~~~

#### 27.3 第三步：Attention 内部拆分

~~~text
Q/K/V Projection
RoPE 后 Q/K
QK MatMul
Mask + Softmax
Attention Weight A
WV MatMul
O Projection
~~~

重点记录：

| 指标 | 用途 |
|---|---|
| cosine | 方向相似度 |
| MAE / RMSE | 平均误差大小 |
| Max Abs | 极端误差 |
| relative L2 | 相对能量误差 |
| Scale | 量化网格宽度 |
| saturation rate | 超出整数范围比例 |
| zero rate | 小值被量化为 0 的比例 |
| min/max/mean/std | 分布变化 |
| Top-K margin | logits 候选是否容易翻转 |

局部 cosine 很高不代表最终一定稳定。Attention 和 36 层 residual 会累积小误差，应同时看逐 Block 漂移曲线。

#### 27.4 第四步：MLP 内部拆分

Language 记录：

~~~text
RMSNorm output
Gate Projection
Up Projection
SiLU output
Gate-activated x Up product
Down Projection
Residual output
~~~

Vision 记录：

~~~text
LayerNorm output
Linear1
GELU
Linear2
Residual output
~~~

如果误差在乘法前较小、乘法后突然增大，应检查两分支 Scale、零值率和幅值相关性，而不是只调整 Down Projection。

#### 27.5 第五步：PBD 和 AR 分开

PBD q=6 和 AR q=1 使用相同权重，但输入 shape、Mask、历史和误差传播方式不同。

应分别维护 Float 与 Quantized KV cache，并固定同一历史逐 Token 比较。已经分叉后的两条自由生成序列不能直接用于定位首个量化错误。

坐标任务同时记录：

~~~text
结构有效率
坐标 Token exact
Float Token 在 Quantized Top-K 中的 rank
Top-1/Top-2 margin
Box IoU
Point PCK
PBD -> AR 回退次数
~~~

### 28. 全量验证和精细诊断如何分工

全量 820 张适合：

~~~text
最终输出
关键 Block 边界
逐样本 cosine / MAE / relative L2
结构和坐标指标
六任务分层统计
~~~

少量最差样本适合：

~~~text
每个叶子 Linear
Q/K/V
Norm
QK
Softmax
WV
Gate/Up/Down
Scale、饱和率和零值率
~~~

全量程序应边计算、边聚合、边写 JSON/CSV，然后释放张量。保存 820 张所有叶子算子 NPY 会产生数百 GB 文件，但不会改善总体统计。

### 29. 从误差首次出现的位置判断原因

| 首次明显偏离 | 优先检查 |
|---|---|
| Original Float -> Adapted Float | Patch 顺序、RoPE、Mask、权重加载、Merger |
| Float -> Weight-only | W4/W8、权重 Scale、轴和布局 |
| Float -> Activation-only | 静态范围、动态粒度、极值和饱和 |
| QK 后下降 | Q/K Scale、RoPE 后范围、Mask |
| Softmax 后下降 | QK score 误差、LUT、低 margin |
| WV 后下降 | Attention 小概率、Value 偏移、零值率 |
| MLP 乘法后下降 | Gate/Up Scale 与乘法放大 |
| Eager -> Exported BC | forward/build 不等价 |
| Exported -> Converted BC | HBDK 融合、布局、Scale 恢复、LUT |
| Converted BC -> S600 | 输入打包、HBRT、dtype、设备执行 |
| 张量一致但任务错误 | Prompt、Tokenizer、PBD 状态机、坐标还原 |

### 30. 从量化图到 S600

量化方案通过 Eager 门控后，才进入：

~~~text
PyTorch/OELLM build()
  -> LEAP/HBIR
  -> Exported BC
  -> HBDK Convert
  -> Converted BC
  -> HBO
  -> HBM
  -> S600 HBRT/BPU
~~~

| 产物 | 含义 |
|---|---|
| Exported BC | 模型 build() 导出的高层固定图 |
| Converted BC | 完成量化、融合、布局和 Nash-P 算子选择 |
| HBO | 一张图的 Nash-P 目标程序 |
| HBM | 一张或多张 HBO 加元数据形成的板端模型包 |

LocateAnything 图划分：

~~~text
Vision HBM:
  visual

Language HBM:
  prefill
  decode       q=6
  decode_ar    q=1
~~~

Converted BC 中出现 native::MatMul 等 Host 算子，说明候选发生 CPU fallback。即使数值更好，也不能称为完整 BPU 方案。

### 31. 4090 与 S600 分别验证什么

4090 编译机：

~~~text
CUDA：
  Original Float
  Adapted Float
  Quantized Eager
  校准前向

CPU / 编译器：
  Exported BC Host 执行
  HBDK Convert
  HBO 编译
  Nash-P 指令仿真
~~~

S600：

~~~text
Host CPU：
  图像解码、letterbox、patchify
  Tokenizer、Embedding 查询
  Position、Mask、KV ring
  PBD 接受、AR 回退
  坐标解析和 inverse letterbox

BPU：
  visual
  prefill
  decode q=6
  decode_ar q=1
~~~

4090 上启动 HBM 仿真不等于 NVIDIA GPU 在执行 BPU 模型。S600 HBRT/BPU 才是最终设备路径。

### 32. 最终验收必须分成三类

#### 32.1 数值一致性

~~~text
Float -> Q/DQ
Q/DQ -> Exported BC
Exported BC -> Converted BC
Converted BC -> S600 HBM
~~~

#### 32.2 任务精度

Float、Quantized Eager 和 HBM 分别对 Ground Truth 计算：

~~~text
Detection/Referring/Layout/GUI：Precision、Recall、F1、Matched IoU
Pointing：归一化距离、PCK
OCR：文本、结构和坐标
~~~

Float 不是 Ground Truth，历史自由生成结果也不是 Ground Truth。

#### 32.3 性能和资源

~~~text
Vision latency
Prefill latency
PBD/AR latency
Host packing 和 output copy
KV commit
HBM size
Host/BPU memory
CPU/BPU utilization
thermal
~~~

精度通过前不应为了速度静默改变模型语义；性能数据必须固定图片、Prompt、生成长度和计时范围。

### 33. 当前结论边界

已经有明确证据支持：

1. Qwen2.5-VL-3B 的自编译 Vision、Language 和 Embedding 可以在 S600 完成图文推理；
2. LocateAnything 必须保留 MoonViT、152681 Vocabulary、一维 Language RoPE、PBD q=6 和 AR q=1；
3. LocateAnything Vision 的主要历史误差位于激活量化，特别是 WV，而不是 W8 Linear；
4. Vision Value 中心化改善了动态 S8 Eager，并形成全 BPU HBM；
5. 最终 Vision HBM 在 820 张上的 mean cosine 为 0.971394；
6. Language 正式权重为 Decoder W8、lm_head W8，Linear 输入为动态 A8；
7. Language QK/WV/KV 正式编译路径仍使用校准固定对称 A8；
8. AR-WV 是坐标尾部的敏感路径，但还没有一个单一低比特替代被证明能消除全部尾部；
9. HBM 加载、非零 logits 和单张图片成功不能替代六任务 Ground Truth 验收。

尚不能声称：

1. 820 张校准输入等同于独立泛化测试集；
2. 100 个 Layout 的历史预测等同于官方标注；
3. U8 或局部 Float 已成为当前正式 HBM；
4. 每一张 Vision 样本 cosine 都超过 0.95；
5. Language 六任务绝对精度已因固定图片运行成功而全部通过。

### 34. 源码阅读顺序

~~~text
浮点结构与配置
  -> compiler/leap_llm/models/locateanything/config/
  -> compiler/leap_llm/models/locateanything/vision_model_leap.py
  -> compiler/leap_llm/models/locateanything/text_model_leap.py

Vision
  -> compiler/leap_llm/models/locateanything/utils/rope_2d.py
  -> compiler/leap_llm/models/locateanything/blocks/vision_attention_leap.py
  -> compiler/leap_llm/models/locateanything/blocks/vision_block_leap.py
  -> compiler/leap_llm/models/locateanything/blocks/vision_patch_merger_leap.py

Language
  -> compiler/leap_llm/models/locateanything/blocks/text_attention_leap.py
  -> compiler/leap_llm/models/locateanything/blocks/text_mlp_leap.py
  -> compiler/leap_llm/models/locateanything/blocks/text_block_leap.py

量化算子
  -> compiler/leap_llm/nn/modules/linear.py
  -> compiler/leap_llm/nn/modules/matmul.py
  -> compiler/leap_llm/nn/modules/const_fake_quant.py
  -> compiler/leap_llm/nn/modules/rms_norm.py

编译接口
  -> compiler/leap_llm/apis/model/locateanything_vision.py
  -> compiler/leap_llm/apis/model/locateanything_language.py

S600 Runtime
  -> deploy/src/hbm_session.cpp
  -> deploy/run_locateanything.py::prepare_image
  -> deploy/src/kv_cache_ring.cpp
  -> deploy/src/hybrid_decoder.cpp
  -> deploy/example/language_hbm_runner.cpp
~~~

### 35. 总结

这套量化部署应该按如下逻辑理解：

~~~text
先理解 Qwen2.5-VL 的 Vision、Language 和多模态注入
  -> 再理解 LocateAnything 为什么替换 MoonViT、词表和解码状态机
  -> 把模型拆成 Vision 与 Language
  -> 每部分再拆成权重、Attention 激活和 MLP 激活
  -> 为每个低比特边界明确 Scale 的计算粒度
  -> 用 Q 域旋转解决隐藏通道能量不均
  -> 用 Value 中心化解决 WV 公共偏移浪费 S8 范围
  -> 用 RMSNorm 保护解决 FP16 平方归约溢出
  -> 用权重/激活/Attention/MLP 单变量 A/B 找到首个误差
  -> 最后才导出 BC、编译 HBM 并在 S600 验证
~~~

部署的目标不是把尽可能多的张量改成 INT8，而是在 S600 可执行约束下，为每一个真正影响精度和性能的边界选择合适的数据类型、Scale 粒度和验证方法。
