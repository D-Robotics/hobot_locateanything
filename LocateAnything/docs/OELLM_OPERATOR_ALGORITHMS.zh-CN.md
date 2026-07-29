# OELLM 算子算法与 Nash-P 执行原理

本文从一次模型计算的实际运行顺序出发，解释 OELLM 如何把 PyTorch 模型变成 S600
上的 Nash-P 程序。内容按“整体路径 → 量化 → 核心计算 → Transformer/Vision → 编译
→ BPU 执行 → 验证”组织，先建立全局认识，再进入每类算子的细节。

文中使用三种证据等级：

- **源码事实**：可由 `compiler/leap_llm`、`llm_compression` 的 Python 源码直接确认；
- **IR 事实**：可由 Exported BC、Converted BC 的算子、类型和属性直接确认；
- **实现未知**：HBDK pass、Nash-P 指令选择和固件调度的闭源细节，不能凭名称推测。

阅读主线如下：

```text
PyTorch 数学模型
  -> OELLM 浮点/校准路径
  -> LEAP 建图路径
  -> HBIR 高层算子
  -> Nash-P backend 算子
  -> HBO 编译与 HBM 链接
  -> HBRT 提交
  -> S600 BPU 执行
```

## 第一部分：OELLM 的整体运行逻辑

### 1.1 同一个 Module 的两条执行路径

OELLM 的 `Module` 同时包含 `forward()` 和 `build()`：

| 模式 | 实际入口 | 输入 | 作用 |
|---|---|---|---|
| `compile_mode(False)` | `forward()` | 真实 PyTorch Tensor | 浮点运行、校准统计、数值验证 |
| `compile_mode(True)` | `build()` | 静态 shape/dtype 的 LEAP Value | 创建 HBIR 计算图 |

`ModuleMeta` 会根据 `compile_mode` 切换路径。普通模式执行 PyTorch 数值计算；编译模式
不再产生真实结果，而是执行 `leap.add`、`leap.dynamic_quantize`、`leap.rms_norm` 等
建图接口，并把模块名和源码位置写入 IR。

这意味着一个 OELLM 算子必须从三个层次理解：

```text
forward()       PyTorch 中怎样计算或收集统计量
build()         Exported HBIR 中创建哪些算子
Converted BC    这些算子最终落到哪个 Nash-P backend
```

### 1.2 从 PyTorch 到 HBM 的编译阶段

| 阶段 | 入口 | 主要产物 | 本阶段回答的问题 |
|---|---|---|---|
| LEAP 导出 | `leap_export()` | Exported BC | 高层数学图和量化点是什么 |
| LLM 图改写 | `llm_convert()` | 改写后的 HBIR | RMSNorm、Softmax、动态量化如何目标化 |
| Backend 转换 | `convert()` | Converted BC | 算子选择 BPU、VPU、SPU 还是 fallback |
| 目标编译 | `compile()` | HBO | 如何 tiling、规划内存、生成多核程序 |
| 链接 | `link()` | HBM | 如何把一个或多个 graph 组成部署容器 |
| 运行 | HBRT/HBUCP | BPU task | 如何分配内存、提交并等待硬件执行 |

后续各部分都沿着这个顺序解释算子，不把 PyTorch 数学、量化规则和硬件 lowering 混在
同一层讨论。

### 1.3 OELLM 算子的五个大类

| 大类 | 典型算子 | 核心职责 |
|---|---|---|
| 量化算子 | `const_fake_quant`、`dynamic_quantize` | 浮点数与低比特整数之间映射 |
| 核心计算 | Linear、MatMul、Conv | 完成主要乘加计算 |
| 模型结构 | RMSNorm、Softmax、RoPE、Attention、MLP | 实现 Transformer/Vision 数学结构 |
| 布局算子 | reshape、transpose、slice、concat | 调整 shape、布局和数据流 |
| 编译运行 | fusion、tiling、DMA、HBRT task | 把计算图变成 Nash-P 可执行程序 |

## 第二部分：量化算子的完整体系

### 2.1 浮点数如何映射为整数

对浮点张量 `x` 做 signed `b` bit 对称量化时，基本公式为：

```text
qmax  = 2^(b-1) - 1
scale = absmax(x) / qmax
q     = clamp(round(x / scale), -qmax, qmax)
x_hat = q * scale
```

- `q` 是真正参与低比特计算的整数；
- `scale` 是相邻两个可表示浮点值之间的距离；
- `x_hat` 是整数 `q` 所代表的近似浮点值；
- `round` 产生舍入误差，`clamp` 产生饱和误差。

scale 越大，可覆盖的浮点范围越大，但量化网格越稀；scale 越小，分辨率越高，但更容易
因超出范围而饱和。

### 2.2 scale 按什么范围共享

| 粒度 | scale 数量 | 常见对象 | 特点 |
|---|---:|---|---|
| per-tensor | 整个张量 1 个 | 静态激活、算子输出 | 成本低，容易受全局极值影响 |
| per-channel | 每个输出通道 1 个 | Linear/Conv 权重 | 能适应不同通道的权重范围 |
| per-row | 每个输入行 1 个 | 动态 A8 激活 | 随每次输入变化，分辨率更细 |
| per-block | K 维每个 block 1 个 | Block Quant MatMul | 精度更高，但 scale 和计算更多 |

“静态/动态”说明 scale 何时产生；“per-tensor/per-row”说明一个 scale 覆盖多少数据。这是
两个不同维度。

### 2.3 静态量化：`ConstFakeQuant`

#### 2.3.1 校准阶段

`ConstFakeQuant.forward()` 对每批校准输入执行：

```python
curr_absmax = x.abs().max()
self.absmax = max(self.absmax, curr_absmax)
return x
```

它只累计历史最大绝对值，返回的仍是原浮点张量。因此普通 OELLM 校准 Eager 路径是
observer 统计，不是严格的量化-反量化数值仿真。

#### 2.3.2 导出阶段

`ConstFakeQuant.build()` 把校准范围写入 HBIR：

```text
qnt.const_fake_quant(
    min=[-absmax],
    max=[ absmax],
    bits=b,
    narrowRange=true
)
```

`axis=None` 表示整个张量共用范围；`axis=0` 配合一组 min/max，表示每个输出通道独立
量化。HBDK Convert 再把这种 fake-quant 语义转成真实的 quantize/dequantize、整数输入
或融合 kernel 参数。

#### 2.3.3 静态量化的误差来源

```text
校准范围太小 -> 推理值超范围 -> saturation
校准范围太大 -> scale 过大    -> 常见值分辨率不足
异常极值过多 -> 大量整数码值用于覆盖少数极值
分布发生变化 -> 固定 scale 不再匹配当前输入
```

所以“observer 全部执行”只证明收集流程运行过，不证明 scale 合理。

### 2.4 动态量化：`dynamic_quantize`

#### 2.4.1 HBIR 接口和 shape

OELLM 常用：

```python
q, scales = leap.dynamic_quantize(x, blockSize=-1)
```

在已核验的 HBIR 中，当 `x=[...,M,2304]` 时，`blockSize=-1` 被推导为完整 K 维：

```text
x      : [..., M, 2304] f16
q      : [..., M, 2304] si8
scales : [..., M,    1] f16
```

即每一行一个 scale。若显式设置更小 block size，K 维会被切成多个量化块；具体输出
shape 必须以相应 HBIR 为准。

#### 2.4.2 Nash-P 上的实际算法

Converted IR 将动态量化展开为：

```text
x
  -> abs
  -> reduce_max(axis=K, keepdim=true)
  -> multiply approximately 1/127
  -> add a small positive guard
  -> reciprocal through LUT
  -> x * reciprocal_scale
  -> round-to-even
  -> saturate to signed int8
```

主干公式为：

```text
scale ~= max(abs(x_row)) / 127 + epsilon
q     = saturate_s8(round_even(x / scale))
```

保护项避免全零行或极小范围造成除零。动态量化不依赖固定激活 scale，但每次执行都要
付出 abs、reduce-max、倒数、乘法和整数转换的额外成本。

### 2.5 静态量化和动态量化如何选择

| 对比项 | 静态量化 | 动态量化 |
|---|---|---|
| scale 来源 | 离线校准 | 当前输入实时计算 |
| 对分布变化的适应性 | 较弱 | 较强 |
| 运行时额外开销 | 较低 | 较高 |
| 主要风险 | 饱和或 scale 被极值拉大 | reduce/LUT 开销及 backend 支持 |
| 最适合 | 分布稳定、性能敏感的张量 | 行间范围差异大或输入变化明显的张量 |

选择必须依据逐算子精度与真机性能，不应简单地把所有 A8 都改成动态或全部固定为静态。

## 第三部分：低比特核心计算算子

### 3.1 `DynamicQuantLinear`：W4/A8 Linear

设输入 `X=[...,M,K]`，权重 `W=[N,K]`。OELLM 的 W4/A8 路径依次执行：

```text
1. X 按行动态量化：X -> Xq(S8), Sx
2. W 按输出通道计算 scale：
       Sw[n] = max_k(abs(W[n,k])) / 7
3. Wq = clamp(round(W / Sw), -8, 7)
4. block_quantized_matmul(Xq, Wq, Sx, Sw)
5. 如果存在 bias，执行浮点 bias add
```

目标结果为：

```text
Y[m,n] ~= sum_k(Xq[m,k] * Wq[n,k]) * Sx[m] * Sw[n] + bias[n]
```

这里：

- W4 表示权重使用 4-bit 有效码值范围；用 `torch.int8` 暂存不等于 W8；
- A8 表示输入激活映射为 signed int8；
- 权重是离线静态量化，激活是运行时动态量化；
- `has_scale=True` 时复用外部提供的权重 scale，否则从当前权重计算。

### 3.2 `block_quantized_matmul`：分块量化矩阵乘

#### 3.2.1 trans-RHS shape 约定

OELLM 使用的形状是：

```text
lhs: [..., M, K]
rhs: [..., N, K]
out: [..., M, N]
```

逻辑运算为 `lhs @ rhs^T`。RHS 已经按 `[N,K]` 传入，调用方不能再先转成 `[K,N]`，
否则会出现双重转置语义或 K 维不匹配。

#### 3.2.2 block 数学

K 维被切成多个 block 时：

```text
Y[m,n] ~= sum_b (
    sum_{k in block b} Xq[m,k] * Wq[n,k]
  ) * Sx[m,b] * Sw[n,b]
```

完整 K 维一个 block 就退化成 per-row/per-output-channel 的单 scale 形式。

#### 3.2.3 Nash-P backend 实现

已核验的 Converted IR 使用：

```text
int8 lhs/rhs
  -> reshape + transpose
  -> b30.conv2d integer MAC primitive
  -> reshape + transpose back
  -> multiply lhs scale
  -> multiply rhs scale and mmaAlpha-related factor
  -> float16 output
```

因此 Converted BC 中出现 `b30.conv2d` 不代表原模型含有 Conv2D。Nash-P backend 复用
Conv/MAC primitive 承载矩阵乘。

`mmaAlpha=1024` 是 HBIR/backend 属性，不是位宽，也不能按普通乘法常量理解。它如何
参与累加范围、系数编码和误差控制没有公开算法，只能通过 Converted IR 与数值 A/B
探针验证。

### 3.3 静态 `FakeQuantLinear`

静态 Linear 的完整路径是：

```text
输入激活 -> ConstFakeQuant(per-tensor)
权重     -> ConstFakeQuant(per-output-channel) 或固定 W4 scale
         -> hbir.linear
输出     -> 可选 ConstFakeQuant，通常 16 bit
```

它省去运行时 abs/reduce/LUT，但输入 scale 依赖校准分布。

### 3.4 `FakeQuantMatmul` 与 `DynamicQuantMatmul`

| 算子 | 左输入 | 右输入 | 主计算 |
|---|---|---|---|
| `FakeQuantMatmul` | 静态 observer scale | 静态 observer scale | `hbir.matmul` |
| `DynamicQuantMatmul` | 实时 per-row/block S8 | 实时 per-row/block S8 | `block_quantized_matmul` |

Softmax 概率、Q/K、V、KV Cache 在不同阶段可能具有完全不同的分布。使用静态 MatMul 时
必须分别检查饱和率、量化后零值率和局部 cosine；使用动态 MatMul 时还要检查运行时
reduce/LUT 成本。

### 3.5 Conv：另一类 MAC 核心

Conv2D 的数学形式为：

```text
Y[n,h,w,o] = bias[o] +
  sum_{kh,kw,c} X[n,h+kh,w+kw,c] * W[o,kh,kw,c]
```

OELLM Conv 通常采用输入 per-tensor、权重 per-output-channel 的静态量化。由于 Nash-P
的核心乘加 primitive 以 Conv 形式表达，原始 Conv 和 lower 后的 MatMul 都可能看到
`b30.conv2d`，必须结合前后 shape 和源码位置判断来源。

## 第四部分：Transformer 算子的运行顺序

一个典型 Decoder Block 按以下顺序执行：

```text
输入 hidden states
  -> RMSNorm
  -> Q/K/V Linear
  -> RoPE
  -> QK MatMul
  -> scale + mask + Softmax
  -> WV MatMul
  -> Output Linear
  -> Residual Add
  -> RMSNorm
  -> Gate/Up Linear + Activation + Down Linear
  -> Residual Add
```

下面沿这条路径解释每类算子。

### 4.1 RMSNorm：归一化并保护 FP16 范围

RMSNorm 定义为：

```text
rms(x) = sqrt(mean(x^2) + eps)
y      = weight * x / rms(x)
```

OELLM 在浮点/校准路径记录最大的 `sum(x^2)`，计算保护系数：

```text
s = max(1, 2 * sqrt(max_sum_x2 / 65504))
```

导出时使用 `x/s` 和 `eps/s^2`。由于：

```text
(x/s) / sqrt(mean((x/s)^2) + eps/s^2)
= x / sqrt(mean(x^2) + eps)
```

该变化数学等价，目的是避免 FP16 平方与归约溢出。

RMSNorm 可以导出为融合 `hbir.rms_norm`，也可以拆成：

```text
pow -> reduce_mean -> add eps -> rsqrt -> mul -> mul weight
```

具体选择由模型 `build()`、输入 shape 和 `llm_convert(rmsnorm_version=...)` 决定。

### 4.2 LayerNorm：比 RMSNorm 多一步中心化

```text
mu  = mean(x)
var = mean((x - mu)^2)
y   = weight * (x - mu) / sqrt(var + eps) + bias
```

OELLM 同时提供融合 `leap.layernorm` 和拆解 `LayerNormSplit`。拆解版本也可以使用等价
缩放避免 FP16 中间值溢出。

### 4.3 RoPE：给 Q/K 注入位置信息

标准 1D RoPE 先计算：

```text
inv_freq[i] = 1 / base^(2i/d)
theta[p,i]  = p * inv_freq[i]
```

再对 Q/K 的成对通道旋转：

```text
rotate_half([x1,x2]) = [-x2,x1]
q' = q*cos(theta) + rotate_half(q)*sin(theta)
k' = k*cos(theta) + rotate_half(k)*sin(theta)
```

cos/sin 通常离线预计算，运行时按 `position_ids` gather，再通过 mul、neg、concat/add
完成旋转。

### 4.4 Attention：QK、Softmax 与 WV

完整计算为：

```text
Q = X Wq, K = X Wk, V = X Wv
Q,K = RoPE(Q,K)
score = Q K^T / sqrt(head_dim)
score = score + additive_mask
P = softmax(score)
O = P V
Y = O Wo
```

这里包含两个性质不同的 MatMul：

| MatMul | 左输入 | 右输入 | 分布特征 |
|---|---|---|---|
| QK | Q | K | 有正有负，范围受 hidden state 和 RoPE 影响 |
| WV | Softmax 概率 P | V | P 非负、和为 1，通常含大量小概率 |

因此 QK 与 WV 不应默认共用同一种量化策略。WV 尤其需要观察概率量化后的零值率。

### 4.5 Softmax 与 LUT

数值稳定的 Softmax 为：

```text
m   = max(x)
e_i = exp(x_i - m)
y_i = e_i / sum_j(e_j)
```

OELLM 导出 `leap.softmax`，`llm_convert` 用 `softmax_version` 选择目标实现。Nash-P 常将
exp、reciprocal、rsqrt 等非线性运算变成 `b30.lut` 与 reduce/eltwise 组合。

Converted IR 能显示 LUT 常量、round mode 和 saturation；LUT 的生成算法及严格误差界
没有在 Python API 中公开，所以应通过输入范围扫描和 Float 对比评估。

### 4.6 KV Cache：保存什么以及如何参与 Decode

每层 KV Cache 保存历史 token 经过 K/V 投影后的张量。Decode 时：

```text
历史 K/V cache
  + 当前 token 的 K/V
  -> slice/concat/transpose
  -> 当前 Attention 的完整 K/V
```

静态 cache shape 是最大容量，不表示所有位置都有效。真实有效长度由 host 维护的
position、slice 和 attention mask 决定。Prefill、PBD 和 AR 可以复用同一组权重，但因
q_len、mask 和 cache 有效区间不同，通常需要不同静态 graph。

### 4.7 MLP 与激活函数

常见 gated MLP 为：

```text
gate = activation(X W_gate)
up   = X W_up
out  = (gate * up) W_down
```

常用激活定义：

```text
SiLU/Swish: x * sigmoid(x)
GELU      : x * Phi(x)，backend 可能使用近似
Tanh      : (exp(2x)-1)/(exp(2x)+1)
```

`FakeQuantSoftmax/Swish/GELU` 可以在输出后追加 `ConstFakeQuant`。这会同时包含非线性
近似误差和量化舍入误差，分析时应分开测量。

## 第五部分：Vision 与张量布局算子

### 5.1 Patch Embedding

Patch Embedding 本质上是：

```text
kernel_size = patch_size
stride      = patch_size
```

的卷积。每个互不重叠的图像 patch 被线性投影为一个视觉 token。3D Patch Embedding
则把 temporal patch 一并纳入卷积核。

在 OELLM 中，Patch 输入和权重可先经过静态 fake quant，权重通常按输出通道量化，随后
调用 `leap.conv2d/conv3d`。

### 5.2 reshape 与 transpose

`reshape` 改变逻辑 shape，`transpose` 改变维度顺序。高层看似只改变视图，但到了
Nash-P backend，是否需要真实数据重排取决于前后算子的物理 layout。

典型 Attention 布局变化为：

```text
[B,T,H*D]
  -> reshape [B,T,H,D]
  -> transpose [B,H,T,D]
```

### 5.3 slice、concat 与 tile

- `slice`：选择 cache 或 token 的有效区间；
- `concat`：连接历史 KV 与当前 KV；
- `tile`：在 GQA/MQA 中把较少的 KV head 逻辑扩展给多个 Q head。

这些算子通常不修改数值定义，但可能产生显著 DDR 搬运。是否被融合、是否零拷贝，必须
查看 Converted IR 和 perf，不能由高层算子名称判断。

## 第六部分：HBDK 如何把算子变成 BPU 程序

### 6.1 Exported BC：保留高层算法

`leap_export()` 产生的图主要包含：

```text
hbir.*    高层计算：matmul、softmax、rms_norm、conv、reduce
qnt.*     量化语义：const_fake_quant、dynamic_quantize
func.*    graph 接口和返回值
```

这一层最适合确认模型结构、量化点、shape 和源码位置是否正确。

### 6.2 `llm_convert()`：LLM 专用改写

`llm_convert()` 接收 `march`、`rmsnorm_version`、`softmax_version` 等参数，负责对 LLM
特有图模式做目标化改写。公开接口能确认它被调用以及改写前后的 IR；具体 pass 顺序和
内部启发式未公开。

### 6.3 `convert()`：选择 Nash-P backend

常见映射如下：

| 高层 HBIR | Converted BC 常见表达 |
|---|---|
| `qnt.dynamic_quantize` | abs + reduce(max) + reciprocal LUT + `b30.quantize` |
| `hbir.block_quantized_matmul` | layout ops + `b30.conv2d` + scale restore |
| `hbir.reduce_*` | `b30.reduce` |
| `hbir.add/mul/sub` | `b30.binary_eltwise` |
| `hbir.rsqrt/exp` | `b30.lut` 或 VPU 路径 |
| reshape/transpose/slice | `hbdk.*` layout/data-movement ops |
| 不支持的 native 算子 | `hbtl.call: native::*` |

`b30vpu.*` 表示 VPU backend，不能因为名字中有 `call` 就认定为 CPU fallback。明确的
主机 fallback 通常表现为 `hbtl.call: native::*`，还应结合 compiler advice 和真机 perf
确认。

### 6.4 算子融合

融合可能把以下操作组合为一个 kernel：

```text
quantize -> MAC -> scale restore -> bias -> activation -> requant
```

融合还可能改变 layout，使中间结果直接留在片上存储中。算子数减少通常有利于性能，
但不自动代表数值正确；Exported BC 与 Converted BC 仍需比较。

### 6.5 `compile()`：tiling、内存和多核

Converted BC 进入 HBO 编译后，大型 MatMul/Conv 通常经历：

```text
1. 沿 M/N/K 或 H/W/C 切 tile
2. 从 DDR 搬运输入和权重到 L2M
3. 执行整数 MAC 或 VPU 运算
4. 合并 K tiles 或分块结果
5. 恢复 scale、执行 bias/activation/requant
6. 写回 DDR 或直接交给融合后继算子
7. 在多个 BPU core 间分配和同步任务
```

相关公开参数包括：

| 参数 | 作用 |
|---|---|
| `core_num` | 目标 BPU core 数量 |
| `max_l2m_size` | 允许使用的片上 L2M 上限 |
| `opt` | 编译优化等级 |
| `jobs` | 编译主机并行任务数，不是 BPU core 数 |
| `input/output_no_padding` | graph I/O padding 约定 |
| `enable_hpc` | 启用相应高性能编译路径 |

具体 tile 尺寸、double buffering、DMA 时序、core 切分和指令选择属于闭源实现，只能由
compiler dump/perf 证明。

### 6.6 HBO、HBM 与 S600 运行

```text
Converted BC
  -> compile
HBO：单个已编译目标对象
  -> link
HBM：包含一个或多个 graph 的最终部署容器
  -> HBRT/HBUCP
BPU task
```

S600 运行时的核心顺序是：

```text
加载 HBM
  -> 查询 graph 和 tensor contract
  -> 分配设备可访问内存
  -> CPU 写入输入并 flush cache
  -> 创建并提交 BPU task
  -> 等待完成
  -> invalidate cache
  -> CPU 读取输出
```

HBM 能加载、能运行、输出非零，只证明接口和运行时基本兼容，不证明量化精度正确。

## 第七部分：按算子层级验证是否正确

### 7.1 六层验证边界

| 层级 | 对比对象 | 主要检查 |
|---|---|---|
| 1 | 原生 Float vs OELLM Float | 模型移植、权重、shape、语义 |
| 2 | Float vs 严格 QDQ Eager | 量化算法自身的误差 |
| 3 | QDQ Eager vs Exported BC | LEAP 导出是否忠实 |
| 4 | Exported BC vs Converted BC | backend lowering、融合、fallback |
| 5 | Converted BC vs HBM 仿真 | HBO 编译、链接和最终数值 |
| 6 | HBM 仿真 vs S600 | 仿真与真实 BPU 是否一致 |

### 7.2 不同算子应该观察什么

| 算子 | 关键统计 |
|---|---|
| 静态量化 | absmax、scale、饱和率、量化后零值率 |
| 动态量化 | 每行 scale 分布、额外耗时、零值率 |
| Linear/MatMul | cosine、MAE、relative L2、最大误差 |
| Softmax | 行和、概率零值率、top-k 概率保持情况 |
| RMSNorm | 输出范数、极值、NaN/Inf、融合与拆解一致性 |
| KV Cache | 每层 K/V cosine、有效区间、跨步累积误差 |
| 最终 logits | cosine、top-1/top-k token 一致率 |

### 7.3 能确认和不能确认的边界

```text
源码能够确认：数学公式、量化点、scale 粒度、graph shape
HBIR 能够确认：实际导出的算子、类型、属性和源码定位
Converted IR 能确认：backend 选择、layout、部分 round/saturation 行为
HBO/HBM 能确认：graph contract、目标架构、编译和链接产物
perf/dump 能确认：具体图的 tile、内存和部分调度结果

不能仅靠 Python 接口确认：
闭源 pass 的完整算法、LUT 生成细节、指令级调度启发式、固件内部线程模型
```

### 7.4 源码入口

| 主题 | 文件 |
|---|---|
| 双路径与编译入口 | `compiler/leap_llm/nn/utils.py` |
| 静态 observer | `compiler/leap_llm/nn/modules/const_fake_quant.py` |
| Linear | `compiler/leap_llm/nn/modules/linear.py` |
| MatMul | `compiler/leap_llm/nn/modules/matmul.py` |
| RMSNorm | `compiler/leap_llm/nn/modules/rms_norm.py` |
| LayerNorm | `compiler/leap_llm/nn/modules/layer_norm.py` |
| 激活函数 | `compiler/leap_llm/nn/modules/activation.py` |
| Conv/Patch | `compiler/leap_llm/nn/modules/conv.py`、`vision_embedding.py` |
| 通用编译转换 | `compiler/llm_compression/converters/compile_converter.py` |
| HBDK API | 安装包内 `hbdk4/compiler/docs/hbdk_api_reference.md` |
