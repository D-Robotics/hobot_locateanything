# LLM-VLM 量化技巧整理

后训练量化的核心，是用有限个整数码值近似原模型中的连续浮点数，并让这种近似能够落到目标设备的矩阵乘、卷积和缓存算子上。位宽只是量化配置的一部分。整数范围、Scale 粒度、零点、舍入、饱和、累加精度和图内实现共同决定最终误差。

本文讨论 LLM 与 VLM 量化中常用的数值方法。完整的 Float、PyTorch Q/DQ 量化仿真、BC、HBO、HBM 和设备执行流程见[《LLM-VLM 量化流程整理文档》](./LLM_VLM_QUANTIZATION_PIPELINE.zh-CN.md)。本文不重复编译步骤，而是沿着下列技术关系说明每种方法的定义、收益来源、实现方式和适用边界。

~~~text
浮点张量
  -> 1. 量化表示与 Scale 设计
       -> 整数码域、仿射映射、舍入与饱和
       -> 权重和激活的 Scale 粒度
       -> Static A8、Dynamic A8、W4/W8、S8/U8
  -> 2. 浮点等价变换与分布整形
       -> 静态图等价改写
       -> 隐藏空间正交变换
       -> Value 中心化
  -> 3. Attention 与 Language 敏感路径
       -> QK、Softmax、WV、KV cache、lm_head
       -> 局部混合精度
  -> 4. 误差分析与方案选择
       -> 张量误差、决策误差、任务误差
       -> 固定历史比较、单模块回退、设备成本
~~~

## 1. 量化表示与尺度设计

### 1.1 有限整数码域与可表示范围

整数码域是量化张量允许使用的有限整数集合。位宽决定码值数量，signedness 决定这些码值如何分配到正负区间。b bit 的 signed 和 unsigned 整数域分别为：

~~~text
Q_b^S = {-2^(b-1), ..., 2^(b-1)-1}
Q_b^U = {0, ..., 2^b-1}
~~~

8 bit 可以编码 256 个不同值。S8 通常把码值分布在零点两侧，U8 则全部位于非负整数区间。

| 表示 | 完整整数域 | 常用对称窄范围 | 典型用途 |
|---|---:|---:|---|
| S4 | [-8, 7] | [-7, 7] | W4 权重 |
| S8 | [-128, 127] | [-127, 127] | W8 权重、signed A8 |
| U8 | [0, 255] | [0, 255] | 非负激活或非对称 A8 |

整数码值本身没有浮点量纲。Scale s 规定相邻码值之间的浮点间隔，zero point z 规定浮点零对应哪个整数码值。给定 s、z 和整数边界，可表示的浮点范围为：

~~~text
R_q = [s * (qmin - z), s * (qmax - z)]
~~~

因此，整数码域回答“有多少个位置可用”，Scale 和 zero point 回答“这些位置落在浮点数轴的哪里”。同样的 S8 码域可以覆盖 [-1,1]，也可以覆盖 [-100,100]，但两个区间的量化步长完全不同。

### 1.2 仿射映射与量化误差

浮点张量 x 的仿射量化和反量化为：

~~~text
q     = clip(round(x / s) + z, qmin, qmax)
x_hat = (q - z) * s
~~~

q 是写入整数码域的值，x_hat 是该码值代表的浮点近似。目标整数算子直接使用 q；PyTorch Q/DQ 仿真通常显式构造 x_hat，再用浮点算子观察量化误差。

对称量化令 z=0，正负值共享同一个 Scale。它适合零点附近近似对称的权重和激活，并能简化整数 MatMul。非对称量化允许 z 非零，使码值区间贴近偏移分布。常用参数计算为：

~~~text
s = (xmax - xmin) / (qmax - qmin)
z = clip(round(qmin - xmin / s), qmin, qmax)
~~~

非对称量化提高了偏移分布的码值利用率，但整数 MatMul 需要处理零点修正项，目标 kernel 也必须支持相应语义。

量化误差主要由两部分组成。

1. **舍入误差**：浮点值位于两个网格点之间，只能映射到其中一个。未发生 clipping 且采用 round-to-nearest 时，单值绝对误差满足近似上界：

   ~~~text
   abs(x_hat - x) <= s / 2
   ~~~

2. **饱和误差**：浮点值超出 R_q，只能被截到边界。该误差不受 s/2 约束，可能远大于普通舍入误差。

减小 s 会让网格更密，降低范围内的舍入误差，同时缩小可覆盖区间并增加饱和风险。增大 s 可以覆盖更大的异常值，却会让主体分布使用更稀疏的网格。量化范围估计需要在两类误差之间选择工作点。

Absmax 是直接的对称范围估计：

~~~text
s = max(abs(x)) / Q
~~~

Q 是对称正半轴的最大码值，例如窄范围 S8 的 127。Absmax 不主动裁剪极值。若少数异常值远离主体分布，百分位或误差最小化方法可以裁剪尾部，以换取主体区域更细的网格。此时必须同时观察真实截断率和任务结果，不能只依据局部 cosine 选择阈值。

同一位宽不代表同一量化规则。量化仿真、BC 中的 Quantize/Dequantize、目标整数算子必须固定相同的 qmin、qmax、舍入和饱和规则。S8 的 [-128,127] 与 [-127,127] 只差一个负码值，却会改变 Scale 计算和边界行为。

全零 tensor 或全零 row 的动态范围为零。实现需要按目标后端约定把 Scale 设为正值，例如 1 或固定 epsilon，使所有量化结果仍为零并避免除零。

### 1.3 权重、激活与累加器的数据合同

W8A8 描述的是 MatMul 两侧输入的主要整数位宽，不代表整条算子都以 8 bit 执行。整数乘积需要在更宽的累加器中求和，Bias、Scale、输出反标定和后续残差也可能使用其他精度。

| 数值对象 | 分布特点 | 关键配置 |
|---|---|---|
| Weight | 部署前固定，可离线处理 | W4/W8、channel/group Scale、布局 |
| Activation | 随样本、token 和阶段变化 | A8/A16、static/dynamic、row/block Scale |
| Bias | 与输出通道对应 | 浮点加入或 accumulator 域整数化 |
| Accumulator | 汇总 K 个整数乘积 | 位宽、分块、溢出和饱和 |
| Output | 进入残差或下一算子 | 反标定 dtype、Requantize 位置 |

权重适合离线量化，因为部署后数值不再变化。激活只能在校准阶段估计分布，或者在推理时根据当前输入计算 Scale。累加器则要覆盖多个整数乘积之和。若两个输入分别受限于 Q_x 和 Q_w，长度为 K 的累加器绝对值上界可粗略写为：

~~~text
abs(Acc) <= K * Q_x * Q_w
~~~

真实上界还受分块、稀疏性和符号抵消影响，但该式说明累加器不能直接沿用输入位宽。累加位宽不足会造成溢出或饱和，这类误差不会出现在只模拟输入 Q/DQ 的浮点 MatMul 中。

### 1.4 矩阵乘法的整数累加与反标定

以 Linear 为例，设权重 W[O,K] 使用每输出通道 Scale，激活 X[M,K] 使用每行 Scale：

~~~text
s_w[o]   = max_k(abs(W[o,k])) / Q_w
s_x[m]   = max_k(abs(X[m,k])) / Q_x

W_q[o,k] = clip(round(W[o,k] / s_w[o]), -Q_w, Q_w)
X_q[m,k] = clip(round(X[m,k] / s_x[m]), -Q_x, Q_x)

A_int[m,o] = sum_k(X_q[m,k] * W_q[o,k])
Y_hat[m,o] = A_int[m,o] * s_x[m] * s_w[o]
~~~

s_x[m] * s_w[o] 把无量纲的整数累加结果恢复到浮点输出域，这一步称为反标定。若下一算子仍接收整数，还需要把 Y_hat 按新的输出 Scale 再量化，形成 Requantize。每增加一次 Requantize，就增加一个舍入和饱和位置。

Per-group 或 per-block Scale 不能直接从总累加器外乘一次 Scale。每组必须分别累加并反标定：

~~~text
Y_hat[m,o]
  = sum_g s_x[m,g] * s_w[o,g] * Acc[m,o,g]
~~~

这也是细粒度 Scale 需要专用 kernel 的原因。若先把所有组累加到同一个整数值，组间 Scale 差异已经无法恢复。

Bias 可以在反标定后以浮点加入，也可以量化后进入累加器。Dynamic per-row s_x 下，Bias 的整数表示随 m 变化：

~~~text
q_b[m,o] = round(b[o] / (s_x[m] * s_w[o]))
~~~

它不能预存为唯一的 q_b[o]。后端若不支持逐行 Bias Scale，通常会在反标定后以较高精度加入 Bias。

PyTorch Q/DQ 仿真与目标整数 kernel 的代数关系可以相同，执行细节并不等价。前者常把量化输入反量化为浮点，再调用浮点 MatMul；后者可能执行整数乘加、分块累加和硬件特定 Requantize。因此 Q/DQ 仿真适合筛选量化方案，累加位宽、舍入顺序和饱和规则仍需在 Converted BC 或设备侧核对。

### 1.5 Scale 粒度与范围估计

Scale 粒度规定多少个元素共享同一组 s 和 z。共享区域越大，Scale 元数据越少，区域内的局部极值也越容易支配全部元素。缩小共享区域可以让量化网格贴近局部分布，但会增加 Scale 数量、广播和 kernel 复杂度。

| 粒度 | Scale 共享范围 | 数值作用 | 实现代价 |
|---|---|---|---|
| Per-tensor | 整个 tensor | 元数据最少 | 局部极值影响全张量 |
| Per-output-channel | 一个权重输出通道 | 隔离输出通道范围 | 必须固定正确 axis |
| Per-row / Per-token | 一个激活行 | 适应 token 或 head 差异 | 运行时 reduction 和广播 |
| Per-group | K 维中的固定通道组 | 提高低比特权重局部分辨率 | Scale 存储增加，依赖 group kernel |
| Per-block | 固定二维或多维块 | 与硬件分块共同设计 | block shape 必须与后端一致 |

对 PyTorch Linear 的权重 W[O,K]，per-output-channel Scale 的 shape 通常为 [O,1]，沿 K 维统计。Per-group 则把每个输出通道的 K 维切成若干组，每组独立计算 Scale。Axis 写错不一定触发 shape 错误，仍可能生成可执行但数值错误的图。因此 Scale shape、axis 和广播方向应成为算子合同的一部分。

范围算法和 Scale 粒度解决的是两个不同问题。粒度决定哪些元素共享范围，范围算法决定如何从这些元素中选择 xmin 和 xmax。常见算法包括 absmax、min-max、百分位裁剪和以 MSE 为目标的阈值搜索。算法选择应匹配误差目标：保护极值、降低主体误差或减少饱和，不能用一个名称替代具体参数。

粒度更细也不必然形成更好的端侧方案。目标 kernel 若只接收 per-tensor Scale，额外的分组反标定可能拆散融合或触发 Host fallback。有效粒度需同时满足三项条件：量化误差下降、编译器保留目标算子、设备执行成本可接受。

### 1.6 W4 与 W8 权重量化

W4 和 W8 的主要差别是可用码值数量。对称窄范围 W4 的正半轴最大码值为 7，W8 为 127。在覆盖相同浮点范围时，W8 的步长明显更小，因此对小权重和相邻权重的区分能力更强。

W4 的优势是减少权重容量和读取带宽，收益在大 Linear 和词表投影中更明显。代价是更大的舍入误差，也更依赖 per-channel 或 per-group Scale。W8 占用更高，但通常是敏感投影、首尾层和 lm_head 的保守候选。

权重自身的 cosine 只描述 Weight Q/DQ，不能直接代表算子输出。Linear 输出还取决于激活在哪些输入方向上具有能量。评估 W4/W8 时，应在相同激活输入下比较 MatMul 输出、Block 边界、logits 排序和任务结果。

W4/W8 也不是整个模型只能二选一。混合权重精度可以让大多数层保持 W4，仅把明确敏感的 Attention 投影、视觉投影或 lm_head 提高到 W8。是否值得这样做，取决于敏感层数量、目标 kernel 支持和模型读取带宽。

### 1.7 Static A8 与 Dynamic A8

激活不同于权重。激活范围会随输入内容、token 位置、Attention head、Prefill 和 Decode 阶段变化。Static A8 与 Dynamic A8 的区别，是激活 Scale 在校准阶段冻结，还是在推理阶段根据当前输入重新计算。

#### 1.7.1 Static A8

Static A8 在校准数据上统计激活范围，并把 Scale 固定到部署图。以 per-tensor 对称 S8 absmax 为例：

~~~text
a_max = max_over_calibration(abs(x))
s_a   = a_max / 127
~~~

推理阶段直接复用 s_a，不再计算当前输入的最大值。其主要优势是运行路径短，Scale 可作为常量进入编译图，也更容易与固定整数 kernel 融合。

这一优势成立的前提是校准分布能够代表部署分布。部署值超过 a_max 时会饱和；少数校准极值把 a_max 拉大时，常见值的步长会变粗。校准样本数量只说明统计数据量，不能说明覆盖了正确任务、序列阶段和张量边界。

Static A8 的实现需要明确四项内容：

- 统计位置，即量化发生在算子输入、输出还是缓存边界；
- 统计粒度，即 per-tensor、per-channel 或 per-block；
- 范围算法，即 absmax、百分位或误差最小化；
- 合并规则，即多 batch、多个 token 和不同阶段如何聚合统计量。

独立验证数据上的 clip rate、误差分位数和任务指标，才决定固定 Scale 是否可用。统计值非零且有限，只能说明校准代码运行过。

#### 1.7.2 Dynamic A8

Dynamic A8 在每次推理时根据当前输入计算 Scale。以 X[M,K] 的 per-row 对称 S8 为例：

~~~text
s_x[m]   = max_k(abs(X[m,k])) / 127
X_q[m,k] = clip(round(X[m,k] / s_x[m]), -127, 127)
~~~

每一行只受自身最大值影响。不同 token、head 或样本之间的范围差异不会被压进同一个固定 Scale。对于分布随生成阶段变化明显的激活，这种局部自适应可以减少饱和和网格浪费。

Dynamic A8 的收益来自更准确的当前范围，不是更高位宽。代价是每次执行 abs、reduce-max、倒数、量化和 Scale 传播。若后端能把这些步骤融合到 MatMul 前处理，额外成本可能较小；若需要独立算子或 Host 参与，端到端时延可能抵消整数 MatMul 的收益。

多次收集 dynamic Scale 后再取均值、最大值或分位数，会得到一个新的 static Scale。它不再具有逐输入自适应性，必须按 Static A8 重新测量饱和率和任务误差。

#### 1.7.3 Static 与 Dynamic 的适用条件

| 条件 | 优先候选 | 数值或工程原因 |
|---|---|---|
| 分布稳定、校准覆盖充分、kernel 偏好常量 Scale | Static A8 | 省去在线范围统计 |
| token 或阶段间范围变化明显 | Dynamic per-row A8 | Scale 贴近当前输入 |
| Dynamic 数值更好但后端无法融合 | 更细粒度 Static 或局部高精度 | 避免运行时拆图 |
| 只有少数边界敏感 | 混合 Static/Dynamic | 把在线开销限制在敏感路径 |

选择时不能只比较 PyTorch Q/DQ 仿真的 cosine。还要确认动态量化是否真正落到目标 kernel，以及 Scale 计算是否计入设备时延。

### 1.8 S8 与 U8 的分布适配

S8 适合同时包含正负值的 Q、K、V、residual 和 Linear 输入。Softmax Attention Weight 位于 [0,1]，负半轴不会被使用。若它仍采用对称窄范围 S8，正半轴只有 127 个有效间隔；U8 在 zero point 为 0 时可使用 0 到 255 的全部码值。

对同一个非负最大值 a_max：

~~~text
s_s8 = a_max / 127
s_u8 = a_max / 255
~~~

U8 对非负区间提供更细的量化网格，小概率值更不容易直接舍入为零。这一收益来自码值利用率，不是 U8 天然比 S8 更精确。

U8 需要区分两种语义。非负且下界为零的张量可使用 z=0；跨越零点的偏移分布需要由 xmin、xmax 计算非零 z 和零点修正。目标 MatMul 还必须支持 U8×S8、U8×U8 或编译器可等价变换的 kernel。若 U8 导致算子回退到 CPU，局部数值收益不能转化为部署收益。

## 2. 量化前等价改写与分布整形

量化前改写并非同一种操作。本章区分三类技术：严格保持浮点函数的等价改写、把动态图固定到目标输入配置的接口适配、改变张量分布但保留整体模型函数的重参数化。三者的验证标准不同，不能都用“变换后能运行”代替数值证明。

### 2.1 等价改写、接口适配与分布整形

| 技术类型 | 数学目标 | 典型方法 | 主要收益 | 验证对象 |
|---|---|---|---|---|
| 严格等价改写 | 新旧表达式在规定输入上相等 | 静态帧 kernel 折叠、RMSNorm 参数折叠 | 删除或简化部署算子 | 变换前后 Float 张量 |
| 目标配置适配 | 复现原处理器在固定 profile 下的行为 | Position Embedding 插值、Token 重排 | 把动态输入固化为静态图 | 处理器与部署图接口 |
| 等价重参数化 | 中间坐标改变，端到端函数不变 | 正交隐藏域、Value 中心化补偿 | 改善量化分布 | Float logits、KV 与任务输出 |

严格等价改写必须先在 Float 域成立，再收集校准统计。若改写后继续使用旧图的 Scale，统计范围与实际部署张量已经不一致。

目标配置适配不一定与任意原始输入等价。例如，Position Embedding 从一个网格插值到另一个网格，本身改变了位置表。正确标准是部署实现与原模型处理器在同一目标分辨率下使用相同插值规则。

分布整形的目标不是减少浮点信息，而是让有限整数码值更有效地覆盖张量。正交旋转处理 hidden channel 的能量集中，Value 中心化处理 token 维上的均值偏移。是否有效取决于变换后的实际分布，不能仅从公式推断量化精度必然提高。

### 2.2 静态帧的 Temporal Kernel 折叠

部分 VLM 的 Patch Embedding 使用 Conv3D 处理 temporal 维。静态图片路径可能把同一图像复制到多个 temporal slice。若 temporal kernel 只产生一个输出位置，且每个 slice 都是同一个 x，则：

~~~text
y = sum_t Conv2D(x, W_t) + b
  = Conv2D(x, sum_t W_t) + b
~~~

等效 Conv2D 权重为：

~~~text
W_2d = sum_t W_3d[:,:,t,:,:]
~~~

这一折叠的收益是删除重复的静态帧输入，并把可能不受目标后端支持的 Conv3D 改为 Conv2D 或展平后的 Linear。计算量和输入尺寸下降，checkpoint 中所有 temporal kernel 的贡献仍被保留。

成立原因是卷积对输入和权重均为线性运算。当每个 temporal slice 完全相同，分别卷积再求和等于先对 kernel 求和再卷积。Bias 只在原 Conv3D 输出上加一次，因此折叠后仍保留一次，不能随 slice 数量相乘。

实现时先确认输入构造、temporal stride、padding、dilation 和输出 temporal 长度。若不同 slice 对应不同视频帧，或 kernel 在时间轴上滑动产生多个输出位置，直接求和不再等价。验证应在 Patch Embedding 边界比较原 Conv3D 与折叠路径的逐元素误差，再检查后续 Vision 输出。

### 2.3 Position Embedding 的目标网格适配

可学习二维 Position Embedding 通常绑定训练网格 P[H_0,W_0,D]。部署图使用固定网格 [H_1,W_1] 时，需要按原处理器的规则生成新位置表。hidden 维应作为插值 channel，空间轴仍保持二维结构：

~~~text
P_chw = permute(P, [D,H_0,W_0])
P_new = interpolate(P_chw, size=[H_1,W_1])
P_seq = permute(P_new, [H_1,W_1,D]).reshape(H_1*W_1, D)
~~~

这项适配使 Position Embedding 成为固定权重，运行图不再根据输入尺寸动态插值。优势是图 shape 稳定，位置表可与其他常量共同编译。

插值并非无条件的 Float 等价变换。只有当离线插值与原模型在目标 H_1×W_1 网格上的执行完全一致，部署图才复现该目标 profile。插值模式、align_corners、抗锯齿选项、坐标约定、轴顺序和计算 dtype 都会影响结果。

先把二维网格 flatten，再做一维插值，会把行末与下一行行首错误地视为相邻位置。实现应保留 H、W 两个轴，完成插值后再展平。适配完成后重新执行 Float Vision，并从新图采集量化范围。

### 2.4 Window Attention 的 Token 顺序

Window Attention 为了让同一窗口的 token 连续存储，通常使用索引 idx 把 spatial order 变换为 window order：

~~~text
X_window  = X_spatial[idx]
inv       = argsort(idx)
X_spatial = X_window[inv]
~~~

索引变换是离散置换，不改变 token 数值，只改变它们的位置。其收益是让窗口内 Attention 形成规则分块，便于静态 shape 和局部计算。逆置换则保证后续 Full Attention、spatial merge 或 Language 图像占位仍按原空间顺序接收特征。

错误的 token 顺序不会表现为普通低比特噪声。每个向量本身可能具有正常范数和 cosine，但对应的图像位置已经改变，最终任务结果会出现结构性错误。此类问题不能通过增大位宽或重新校准修复。

实现时需明确每个边界要求的顺序：

~~~text
spatial order
  -> window reorder
  -> Window Attention
  -> optional inverse reorder
  -> Full Attention / spatial merge / VLM injection
~~~

若 Vision 图输出仍为 window order，Host 只执行一次逆重排；若图内已经恢复 spatial order，Host 不再重复处理。验证包含 idx 的置换性、X[idx][argsort(idx)] 与 X 的一致性、merge 后 token 数，以及视觉特征与图像占位的一一对应。

### 2.5 RMSNorm 参数折叠与数值缩放

RMSNorm 先按 hidden 维的均方根归一化，再乘以可学习参数 gamma：

~~~text
n(x) = x / sqrt(mean(x^2) + eps)
y    = (n(x) * gamma) @ W^T
~~~

对于 PyTorch Linear 的 W[out,in]，gamma 可以折叠到 Linear 的输入维：

~~~text
W_fold = W * gamma[None,:]
y      = n(x) @ W_fold^T
~~~

折叠后 RMSNorm 的 weight 可置为 1。这样做可以减少独立逐元素乘法，也便于后续把 hidden-domain 变换统一折叠到 Linear 权重。

等价性来自乘法结合律，但只对所有消费同一 Norm 输出的分支同时成立。Attention 的 Q/K/V 和 MLP 的 gate/up 往往分别共享一个 Norm 输出；漏掉其中一个消费者会使分支语义不同。Residual 旁路使用的是原始 x，不能把 gamma 施加到旁路。

RMSNorm 的数值缩放还可用于避免内部平方溢出。设 s>0，并且 s 在整个归一化维度上共享：

~~~text
z    = x / s
eps' = eps / s^2

z / sqrt(mean(z^2) + eps')
  = x / sqrt(mean(x^2) + eps)
~~~

s 会在分子和分母中约去，不需要在输出端乘回。若 s 是 per-channel 向量，而不是归一化维度共享的标量，这个恒等式一般不成立，因为 mean(x^2) 的相对权重已经改变。

RMSNorm 折叠会改变相邻权重的分布。它可能简化图，也可能让某些输入通道的权重范围增大。应在折叠后重新计算 Weight Scale 和激活校准统计，并先验证 Float 输出等价。

### 2.6 Embedding 与 lm_head 的权重共享

许多 Decoder-only LLM 使用 tied embedding，即输入词嵌入 E 与输出词表投影 W_lm 来自同一份 checkpoint 权重。共享的是浮点模型的参数语义，不一定是部署后的物理存储。

输入侧执行 token ID 查询：

~~~text
h_0 = E[token_id]
~~~

输出侧执行词表投影：

~~~text
logits = h_final @ W_lm^T
~~~

当隐藏域旋转为 h'=hQ 时，输入 Embedding 变为 E'=EQ。若 final RMSNorm 的 gamma_f 同时折叠到 lm_head，则：

~~~text
W_lm' = (W_lm * gamma_f[None,:]) Q
~~~

此时 E' 与 W_lm' 不再逐元素相同，但两者仍来自同一 tied 源权重。Runtime 若强制输入和输出共用同一二进制张量，就不能同时把 gamma_f 只折叠到输出路径；需要保留 final gamma，或调整权重共享实现。

Embedding 与 lm_head 的访问模式也不同。Embedding 是稀疏行查询，lm_head 是大矩阵乘。两者可以使用不同 dtype、Scale 粒度或布局。部署清单应记录它们各自的变换和量化方式，不能根据 tied 名称推断量化后仍共享存储。

### 2.7 隐藏空间正交旋转

隐藏空间正交旋转用正交矩阵 Q 改变 residual stream 的坐标基：

~~~text
Q^T Q = I
h'    = h Q
~~~

正交变换保持二范数和内积：

~~~text
norm(hQ, 2) = norm(h, 2)
(xQ)(yQ)^T = xy^T
~~~

因此它不删除浮点信息。量化收益来自能量重新分配：若原 hidden space 的少数 channel 长期承载大值，共享 Scale 会被这些 channel 支配；合适的 Q 可能把能量分散到更多 channel，使 absmax 与主体分布更接近。

正交性只保证 Float 代数，不保证 absmax 一定下降。旋转可能改善一个张量，同时恶化另一个张量。采用前需要统计旋转前后的 channel absmax、分位数、Scale 和量化误差。

#### 2.7.1 Residual Stream 的权重折叠

对行向量线性关系 y=xW，输入改为 x'=xQ 后取 W'=Q^T W：

~~~text
x'W' = xQQ^T W = xW
~~~

实际 PyTorch Linear 存储 weight[out,in]。设 residual 使用 h'=hQ，并把 RMSNorm gamma 折叠到输入投影，则：

~~~text
W_in' = (W_in * gamma[None,:]) Q
~~~

Attention output projection 和 MLP down projection 把内部结果写回旋转后的 residual 域：

~~~text
W_out' = Q^T W_out
b_out' = b_out Q
~~~

Embedding 使用 E'=EQ。Vision 最后一个 projector 若直接输出 Language hidden，也需要把输出侧 Q 折叠进权重。Q/K/V 的输出仍位于 Attention 内部坐标域；KV cache 不应再次旋转，除非接口明确定义在旋转域。

#### 2.7.2 正交矩阵的实现

归一化 Walsh-Hadamard 变换具有元素幅值均匀、可快速计算的特点。加入确定性符号翻转或置换仍可保持正交。常用快速 Hadamard 实现要求维度为 2 的幂；其他 hidden size 需要分块、padding 或选择其他正交矩阵，并重新证明边界语义。

部署时通常把 Q 离线折叠进已有权重，不在运行图中新增 hidden×hidden MatMul。收益是运行时没有额外旋转开销；代价是要重新物化多组权重，并保证 Embedding、Vision 输出、所有 residual 输入输出和 lm_head 使用同一坐标域。

实施前检查 Q 的 shape、dtype 和 max(abs(Q^TQ-I))。实施后比较旋转前后的 Float Vision 输出、logits、KV、argmax 和任务结果。Float 等价成立后，才比较量化收益。

### 2.8 Value 中心化与补偿计算

Value 中心化处理的是 Attention Value 在 token 维上的均值偏移。设当前实际参与 WV 的 Attention 表示为 A_used[M,T]，Value 为 V[T,D]。沿 token 维计算每个 channel 的均值：

~~~text
mu = mean_T(V)        # [1,D]
Vc = V - 1 * mu       # [T,D]
~~~

WV 可以严格分解为：

~~~text
A_used V
  = A_used Vc + (A_used 1) mu
~~~

若主项使用量化后的 A_hat，补偿项也必须使用同一份 A_hat：

~~~text
A_hat V
  = A_hat Vc + (A_hat 1) mu
~~~

不能在主项使用 A_hat，却在补偿项使用浮点 A 的行和，否则分解对应的不是同一个输入。

中心化的数值收益来自去除每个 Value channel 的公共偏移。若 V 的主要动态范围由均值偏移造成，absmax(Vc) 会小于 absmax(V)，S8 Scale 随之减小，主体值获得更细的网格。均值信息没有丢失，而是通过补偿项加回。

该方法不是方差归一化，也没有训练参数。mu 通常依赖当前输入，因而会新增 reduce、sub、row-sum、补偿乘法和 add。补偿项宜保留较高精度。

在 Decode 的 KV cache 上，token 轴包含历史位置。若每步重新扫描完整 V cache 计算均值，成本随上下文增长。可选实现包括按 cache block 保存统计量，或维护 Value 的累积和并随有效 token 更新。两种实现都必须与 cache position、滑动窗口和无效 padding 保持一致。

以下条件会限制中心化收益：V 已接近零中心；范围由少数 channel 尖峰主导；Attention Weight 量化是主要误差；补偿项本身被低精度量化；token 轴或 transpose 定义错误。采用前后应比较 V 与 Vc 的 absmax、Scale、clip rate、WV 输出和最终任务结果。

## 3. Attention 与 Language 敏感路径

Attention 不是一个孤立 MatMul，而是一条由投影、相关性计算、概率归一化、加权求和和残差写回组成的数值链。不同位置的分布和误差放大机制不同，因此不能用“Attention 全部 A8”概括其量化语义。

### 3.1 Attention 数值链与误差传播

一个 Decoder Block 中与 Attention 相关的主要计算为：

~~~text
residual
  -> RMSNorm
  -> Q / K / V projection
  -> RoPE on Q and K
  -> QK score / sqrt(d)
  -> mask add
  -> Softmax
  -> Attention Weight A
  -> WV MatMul
  -> output projection
  -> residual add

K / V projection
  -> KV cache write
  -> later Decode reads

final residual
  -> final RMSNorm
  -> lm_head
  -> logits
~~~

量化误差沿这条链传播，但不同位置的作用不同：

- Q/K 误差先改变 Attention score，再经过 Softmax 的非线性映射；
- Attention Weight 误差改变各 token 的加权系数；
- V 误差直接改变被聚合的内容向量；
- output projection 误差写回 residual，并进入后续所有 Block；
- KV cache 误差跨生成步持久存在；
- lm_head 误差直接改变最终 token 排序。

对 WV 输出 Y=AV，令量化扰动为 ΔA 和 ΔV，则：

~~~text
DeltaY = DeltaA * V + A * DeltaV + DeltaA * DeltaV
~~~

该式给出直接的定位顺序：分别保持 V 为 Float 只量化 A，保持 A 为 Float 只量化 V，再同时量化两侧。若只观察最终 Y 的 cosine，无法判断误差来自哪一侧。

### 3.2 Q/K 投影与 QK Score

Q、K 由 residual 经 Linear 投影得到，并在进入 QK MatMul 前应用位置编码。Linear 的 Weight/Activation 量化误差、RoPE 的 dtype 和 QK 算子的输入量化都会共同影响 score。

Attention score 为：

~~~text
S = Q K^T / sqrt(d)
~~~

设 Q_hat=Q+ΔQ、K_hat=K+ΔK，score 扰动为：

~~~text
DeltaS
  = (DeltaQ K^T + Q DeltaK^T + DeltaQ DeltaK^T) / sqrt(d)
~~~

该式说明 Q/K 两侧误差会交叉进入 score。Q 和 K 的整体 cosine 较高，并不保证每个 query 对所有 key 的相对排序不变。尤其当两个 score 接近时，小的局部扰动就可能交换它们的顺序。

Q/K 通常包含正负值，可从对称 S8 开始。若不同 token 或 head 的范围差异明显，dynamic per-row 或 per-head Scale 可以避免一个大值支配整张 score 输入。代价是两侧都需要在线范围统计和 Scale 传播。

实现时需区分三个量化边界：

1. Q/K projection 的 Linear 输入与权重；
2. RoPE 前后 Q/K 的 dtype；
3. QK MatMul 自身的 operand Scale。

只把投影 Linear 提高到 W8，并不代表 QK operand 已经高精度；反过来，QK 使用高精度也不能消除投影阶段已经产生的误差。

### 3.3 Mask、Softmax 与 Attention Weight

Mask 在 score 上加入不可见位置的限制：

~~~text
S_masked = S + M
A        = softmax(S_masked)
~~~

M 常包含 0 和大负值，用于让被屏蔽位置的概率接近 0。若把 M 与有效 score 共同纳入普通 activation range，大负 mask 值会支配 Scale，使有效 score 的网格过粗。部署图应明确 mask add、dtype cast 和 Softmax 的先后顺序，并让 mask 使用后端规定的专用表示。

Softmax 通常先减去每行最大值以提高数值稳定性：

~~~text
A_i = exp(S_i - max(S)) / sum_j exp(S_j - max(S))
~~~

Softmax 对整体平移不敏感，但对 score 之间的相对差异敏感。其局部 Jacobian 为：

~~~text
J_softmax(A) = diag(A) - A A^T
~~~

因此，score 扰动对概率的影响取决于当前概率分布。无法用一个固定 QK cosine 推断所有 head 和 query 的 Softmax 误差。

Attention Weight A 非负，每行和在标准 Softmax 下接近 1。部分 head 可能包含许多小概率值。对称 S8 会把一半码值留给不会出现的负区间，U8 可以提高非负区间的分辨率。量化后新产生的零值可定义为：

~~~text
quantized_zero_rate = mean(A_hat == 0)
new_zero_rate       = mean((A != 0) and (A_hat == 0))
~~~

new_zero_rate 描述原本非零的概率有多少被量化为零，不能与模型本身的稀疏性混为一谈。U8 是否采用仍由 kernel 支持、Scale 粒度和设备成本决定。

### 3.4 WV MatMul 与 Value 分布

WV MatMul 把 Attention Weight 作为系数，对 Value token 求加权和：

~~~text
Y[m,d] = sum_t A[m,t] * V[t,d]
~~~

两侧分布明显不同。A 非负并受概率归一化约束，V 通常包含正负值，其范围还可能带有 token 维均值偏移。给两侧机械地使用同一 static S8 策略，无法利用这些结构。

候选设计可以分侧选择：

| A 侧 | V 侧 | 数值依据 | 主要代价 |
|---|---|---|---|
| Dynamic S8 | Dynamic S8 | 两侧范围随输入变化 | 双侧 reduction 与 Scale 传播 |
| U8 | S8 | A 非负 | 依赖 U8×S8 kernel |
| S8 | Centered S8 | V 的均值偏移占据范围 | 中心化与补偿计算 |
| A16 | A8/S8 | A 侧为主要敏感源 | 带宽、算力或 fallback 风险 |
| FP16 | FP16 | 目标后端支持高精度 WV | 资源成本最高 |

诊断时先比较四组输出：

~~~text
Float A,     Float V
Quantized A, Float V
Float A,     Quantized V
Quantized A, Quantized V
~~~

第一组是局部参考，第二、三组分别隔离 A 和 V，第四组观察交叉项。若 V 的均值偏移是主要问题，可采用第 2.8 节的中心化分解；若 A 的小概率大量归零，优先调整 A 的 signedness 或 Scale；若两侧都稳定，再检查 output projection 和 residual 写回。

### 3.5 Output Projection 与 Residual 写回

WV 输出通常先经过 output projection，再与 Block 输入相加：

~~~text
H_attn = WV @ W_o^T + b_o
H_next = H_residual + H_attn
~~~

Output projection 将多个 head 的结果混合回 hidden space。它的权重量化误差会把局部 Attention 误差重新分布到所有 residual channel。若 residual 使用正交隐藏域，W_o 还承担从 Attention 内部坐标写回旋转域的职责。

Residual add 具有两个不同动态范围的输入。直接把两侧压到同一低精度 Scale，可能让较小分支失去分辨率。常见实现是在较高精度中执行 add，或先把两侧 Requantize 到明确的共同输出 Scale。选择取决于后端残差融合算子的合同。

定位 Block 误差时，应分别观察 WV 输出、output projection 输出和 residual add 后输出。若 WV 稳定而 Block 输出下降，继续调整 WV 位宽不会解决后级投影或残差 Scale 问题。

### 3.6 KV Cache 的量化合同

KV cache 保存历史 token 的 K/V，并在后续 Decode 中重复读取。它不是普通临时激活，而是跨生成步持久存在的模型状态。一次写入误差会影响所有读取该位置的后续 Attention。

KV 量化合同至少包含：

- K/V 的 dtype 与整数码域；
- Scale 粒度及 Scale 元数据布局；
- head、token、channel 的物理顺序；
- cache position、有效长度和更新范围；
- Prefill 写入与 Decode 读取的一致性；
- PBD 草稿与已接受 token 的提交规则。

Static KV Scale 可以在覆盖 Prefill 和 Decode 的校准数据上冻结，运行时管理简单，但需要面对序列阶段和 token 位置分布变化。Dynamic KV 有多种语义，不能笼统归为一种方案。

1. **Per-token Scale**：每个新 K/V token 单独计算并保存 Scale。已有 cache 不需要重标定，但 Scale 元数据随序列增长。
2. **Per-block Scale**：一组 token 共享 Scale。追加 token 超出当前范围时，需要开启新 block，或重标定该 block 内已有值。
3. **全 cache 共享动态 Scale**：范围扩张时必须重新量化历史 cache，否则新旧 token 不在同一合同下。这种方案通常带来较高更新成本。

Prefill 与 Decode 若使用不同 Scale，边界必须显式转换。只比较最终 logits 会掩盖最早的 cache 写入误差，因此分析应按 layer、K/V、head 和 token position 分组。

### 3.7 Final Norm 与 lm_head

Decoder 最后通过 final RMSNorm 和 lm_head 将 hidden 映射到全词表 logits：

~~~text
h_final = RMSNorm(h)
logits  = h_final @ W_lm^T
~~~

lm_head 的输出维度等于词表大小，权重矩阵通常很大。W4 可以降低词表投影的权重容量和读取带宽，W8 提供更细的权重网格。由于 logits 直接参与 token 排序，少量权重误差既可能只影响接近并列的候选，也可能改变高置信 token。

Cosine 衡量全量 logits 的整体方向，但词表中大量非候选维度会稀释局部排序变化。lm_head 还应比较：

- top-1 agreement；
- top-k overlap；
- top-1 与 top-2 margin；
- 参考 token 的 rank 和 logit 差；
- 最终任务序列。

若 Float 的 top-1/top-2 margin 本身很小，量化后的交换不等于高置信错误。应继续观察该 token 是否为坐标、结构控制符或普通文本，以及自由生成是否改变最终任务结果。lm_head 使用 W8 是常见保守候选，但是否需要由任务结果和设备权重读取成本共同决定。

### 3.8 Prefill、块解码与 AR Decode

同一 Language Block 在不同生成阶段看到的输入分布并不相同：

| 阶段 | Query length | 历史状态 | 主要分布差异 |
|---|---:|---|---|
| Prefill | 多 token | 通常从空 cache 建立历史 | 文本与视觉 token 混合，mask 较大 |
| 块解码 | 多个候选 token | 读取已有 KV | 块内 mask 和位置规则由算法定义 |
| AR Decode | 1 或少量 token | cache 持续增长 | 单步输入，历史长度不断变化 |

Static Scale 若只从 Prefill 收集，不能自动代表 AR Decode；反之亦然。不同编译图可以使用各自的 static Scale，也可以共享 dynamic Scale。独立 Scale 增加配置和产物管理，dynamic Scale 增加运行时范围计算。

局部精度比较必须固定历史 token、position IDs、mask 和输入 KV。若 Float 与量化模型先自由生成出不同历史，后续 logits 差异同时包含当前步量化误差和先前 token 反馈，无法定位具体算子。

自由生成仍然必要，但用途不同。固定历史回答“当前一步的量化误差在哪里”，自由生成回答“这些误差经过反馈后是否影响最终任务”。

### 3.9 局部混合精度与敏感模块定位

混合精度不是按层号机械切分，而是把有限的高精度资源分配给首个明显误差来源。候选层级包括：

| 方案 | 主要收益 | 主要代价 |
|---|---|---|
| W4 | 降低权重容量与读取带宽 | 权重舍入误差较大 |
| W8 | 提高权重分辨率 | 模型容量与带宽高于 W4 |
| Static A8 | 省去在线范围计算 | 依赖校准覆盖 |
| Dynamic A8 | 适应当前输入范围 | 增加 reduction 和 Scale 传播 |
| A16 | 扩大激活表示能力 | 需明确整数、FP16 或 BF16 语义 |
| FP16 operator | 保留局部浮点路径 | 可能增加资源或产生 Host fallback |

单模块高精度回退用于识别敏感位置。在 PyTorch Q/DQ 量化仿真中，只把一个 Linear、QK、WV、KV 边界、Norm 或 lm_head 恢复到较高精度，其余配置保持不变。输出误差明显下降，说明该模块对当前误差敏感；它仍不自动说明设备端应该采用同样精度。

扫描可从最终输出反向追踪。Block 输出出现明显漂移时，先分离 Attention 和 MLP；Attention 内继续分离 Q/K/V projection、QK、Softmax、WV 和 output projection。组合回退每轮只新增一个模块，才能观察模块间的交互效应。

最终配置还要通过编译和设备约束。一个 FP16 WV 若在 Converted BC 中落到 CPU，就不是可接受的 BPU 混合精度方案。此时应比较后端支持的替代方案，例如 W8、Dynamic A8、U8 或 Value 中心化。

## 4. 误差分析与精度方案选择

量化评估需要回答三个不同问题：张量是否接近，模型决策是否改变，最终任务是否退化。三者有因果关系，但不能互相替代。一个张量 cosine 下降不一定改变任务输出；一个坐标 token 翻转也可能让任务指标显著变化，即使全量 logits cosine 仍很高。

### 4.1 张量误差、决策误差与任务误差

#### 4.1.1 张量误差

张量误差用于定位数值变化发生在哪个边界。设参考张量 Y 和量化张量 Y_hat，常用指标包括：

~~~text
cosine
  = <Y, Y_hat> / (norm(Y,2) * norm(Y_hat,2))

relative_L2
  = norm(Y_hat - Y, 2) / max(norm(Y,2), eps)

MAE
  = mean(abs(Y_hat - Y))

RMSE
  = sqrt(mean((Y_hat - Y)^2))

max_abs_error
  = max(abs(Y_hat - Y))
~~~

这些指标观察的侧面不同。Cosine 关注整体方向，对统一幅值缩放不敏感；relative L2 同时反映方向和能量；MAE 描述平均误差；RMSE 对较大误差更敏感；max absolute error 用于发现局部尖峰。

量化分布还需要记录 Scale、截断率和零值变化。真正的截断率应在量化前按可表示浮点范围计算：

~~~text
xmin_q = s * (qmin - z)
xmax_q = s * (qmax - z)

clip_rate
  = mean((x < xmin_q) or (x > xmax_q))
~~~

q 等于 qmin 或 qmax 只能说明码值落在边界，既可能来自截断，也可能是合法值舍入到边界，不能直接称为 clip rate。

#### 4.1.2 决策误差

决策误差关注模型如何使用张量。Language 模型的直接决策对象是 logits 排序和生成 token；检测或 Grounding 模型还包含标签、坐标和结构控制 token。

常用决策指标包括 top-1 agreement、top-k overlap、参考 token rank、top-1/top-2 margin、结构 token 有效率和停止原因。Margin 可以区分两类变化：

- Float 候选接近并列时，小误差导致的排序交换；
- Float margin 较大时，量化仍改变 top-1 的高置信偏移。

两者的风险不同。坐标、结束符和状态切换 token 还会改变后续生成路径，通常比普通描述文本的近并列交换更敏感。

#### 4.1.3 任务误差

任务误差回答模型是否仍完成目标任务。它需要独立 Ground Truth，而不是把 Float 输出当作标签。Float 与量化模型都应在同一数据、预处理和评价协议下分别计算任务指标，再比较两组结果。

| 任务 | 任务指标示例 |
|---|---|
| 分类或问答 | Accuracy、Exact Match、任务专用得分 |
| Detection / Grounding | Precision、Recall、F1、matched IoU |
| OCR / Text Grounding | 文本准确率、区域匹配、文本与区域联合指标 |
| Pointing | 像素距离、归一化距离、PCK |
| 结构化生成 | 格式有效率、完整序列率、停止原因 |

张量指标用于定位，决策指标用于解释路径变化，任务指标用于决定方案是否可用。只保留其中一层，会让量化结论缺少因果链或业务含义。

### 4.2 固定历史的自回归误差比较

自回归模型会把上一步 token 作为下一步输入。若 Float 在第 t 步生成 token a，量化模型生成 token b，从第 t+1 步开始，两条路径的输入历史已经不同。之后的 logits 差异不再等于单步量化误差。

定位时应使用固定历史比较：

~~~text
same prompt
same accepted token history
same position IDs and mask
same Float or controlled input KV
  -> Float current-step logits
  -> Quantized current-step logits
~~~

KV 也必须受到控制。若两侧各自自由生成并累计自己的 cache，即使 token ID 暂时相同，历史 K/V 也可能已经包含不同量化误差。可采用两种实验：

1. 使用同一 Float 历史重新计算两侧当前步输入，隔离当前算子误差；
2. 各自沿真实 Float/量化 cache 回放同一已接受 token 前缀，观察 cache 误差累积。

固定历史实验解释局部误差，自由生成实验评估最终影响。前者不能替代真实生成，后者也不适合定位第一处偏差。

### 4.3 单变量定位与高精度回退

量化方案包含多个相互作用的变量：Weight 位宽、Activation 位宽、Scale 粒度、static/dynamic、signedness、正交旋转、中心化和局部高精度。一次同时修改多个变量，即使结果改善，也无法确定收益来自哪里。

受控实验固定 checkpoint、输入 tensor、预处理、历史 token、KV、采样参数和图接口，每轮只改变一个变量。误差定位按计算顺序寻找第一个明显漂移边界，而不是从最终输出猜测某一层。

高精度回退是一种局部因果探针：

~~~text
all modules use candidate quantization
  -> restore one module to Float/W8/A16
  -> compare downstream error
~~~

若单模块回退显著恢复下游结果，该模块是当前配置的敏感点。结论仍带有上下文条件：上游输入已经量化，模块与其他模块也可能存在组合效应。组合回退应按累计顺序逐个增加模块，并保留每一步结果。

Weight 与 Activation 还应分开：

- 只对 Weight 执行 Q/DQ，判断 W4/W8 本身的误差；
- 固定 Weight，切换 Static/Dynamic A8，判断激活范围问题；
- 固定两侧，改变 QK、WV、KV 或 Requantize 边界，判断算子合同；
- 最后再测试正交旋转、中心化或混合精度。

这一顺序从简单局部变量逐步进入图级变换，避免把一个 Scale 问题误判为整层需要 FP16。

### 4.4 Ground Truth 集合匹配与任务指标

Detection 和 Grounding 输出是集合，预测顺序通常没有语义。第 N 个预测框不能直接与第 N 个 Ground Truth 框比较。正确评价需要类别或文本约束下的一对一匹配。

基本过程为：

~~~text
predicted objects + Ground Truth objects
  -> filter compatible labels / phrases
  -> build pairwise IoU or distance matrix
  -> one-to-one assignment
  -> apply IoU or distance threshold
  -> TP / FP / FN and matched quality
~~~

匹配算法可以使用 Hungarian assignment，或在固定排序和阈值下使用贪心匹配。两者可能产生不同结果，因此算法、IoU 阈值、类别匹配规则和重复框处理必须固定。

Float 与量化模型分别对同一 Ground Truth 匹配：

~~~text
Float       vs Ground Truth -> Precision_F, Recall_F, F1_F, IoU_F
Quantized   vs Ground Truth -> Precision_Q, Recall_Q, F1_Q, IoU_Q
~~~

两组任务指标之差回答量化是否损害模型能力。Float 与量化输出之间的 pairwise 比较则回答两条路径有多相似。前者是任务评价，后者是量化一致性评价，不能互相替代。

对于 OCR，需要同时处理文本匹配和区域匹配；Pointing 使用点到目标或标注点的归一化距离，并按固定阈值计算 PCK。自由生成中标签数量、框数量和输出顺序都可能变化，评价器应先解析成结构化集合再计算指标。

### 4.5 数值收益、编译支持与设备成本

PyTorch Q/DQ 量化仿真改善，只说明候选量化数学值得继续验证。最终方案还要满足编译器能表达、目标设备能执行和端到端性能可接受。

| 观测 | 首个控制变量 | 候选方法 | 主要成本 |
|---|---|---|---|
| Weight Q/DQ 已有明显误差 | Weight bits | W4 -> W8 | 权重容量与读取带宽 |
| Static A8 clip rate 高 | Scale strategy | Dynamic A8 或细粒度 Static | 在线量化或更多 Scale |
| Clip rate 低但 new zero rate 高 | 粒度与 signedness | Per-row、U8、中心化 | kernel 约束与附加算子 |
| Hidden channel 极值集中 | Hidden domain | 正交旋转、per-group | 权重重写与域一致性 |
| V 存在 token 维均值偏移 | WV 的 V 侧 | Value 中心化 | reduction 与补偿项 |
| QK 稳定但 WV 下降 | A/V 分侧实验 | U8×S8、centered S8、A16 | kernel 支持与资源 |
| Prefill 稳定、Decode 漂移 | Stage 与 cache | 独立 Scale、KV 高精度 | 多 profile 与 Scale 元数据 |
| lm_head 小 margin 翻转频繁 | lm_head bits | W8 或局部高精度 | 词表投影权重成本 |
| 只有少数模块回退有效 | Module precision | 局部 W8/A16/FP16 | 图复杂度与 fallback 风险 |

候选方法应依次回答：

1. **数值收益**：张量、决策和任务误差是否改善；
2. **图支持**：Exported BC 和 Converted BC 是否保留预期算子与 Scale 语义；
3. **设备闭环**：是否在目标计算单元执行，是否出现 Host fallback；
4. **资源成本**：模型大小、内存带宽、临时内存、Scale 计算和调用次数；
5. **端到端收益**：真实任务精度与延迟是否达到目标。

设备利用率不是唯一性能指标。动态量化、缓存搬运、图提交和输出物化可能形成串行瓶颈。局部算子更快，也不保证端到端时延按同一比例下降。

### 4.6 量化配置的收敛顺序与可复现描述

量化策略宜按以下顺序收敛：

~~~text
Float 计算语义和接口顺序
  -> 整数码域与 Weight Scale
  -> Activation Scale 粒度
  -> Static / Dynamic A8
  -> QK / Softmax / WV / KV / lm_head
  -> 正交旋转或 Value 中心化
  -> 少量敏感模块混合精度
  -> BC 与设备成本
  -> Ground Truth 任务结果
~~~

顺序的依据是可解释性。先保证 Float 语义，再处理局部量化；先解决 Scale 和位宽，再引入全隐藏域变换；最后用混合精度收敛剩余敏感路径。否则多个机制同时变化，误差来源和收益来源都难以复现。

一个完整量化配置至少应记录：

| 配置对象 | 必要信息 |
|---|---|
| Weight | 位宽、整数域、Scale 粒度、group/block shape、舍入 |
| Activation | 位宽、S8/U8、static/dynamic、统计位置、范围算法 |
| MatMul | 输入 Scale 语义、累加精度、分组反标定、输出 Requantize |
| Attention | QK、mask、Softmax、WV 两侧和补偿路径 |
| KV cache | dtype、Scale 粒度、元数据、layout、更新规则 |
| 等价变换 | RMSNorm 折叠、隐藏域矩阵、权重变换顺序 |
| 编译图 | 算子类型、fallback、输入输出合同 |
| 评价 | 固定历史协议、Ground Truth 匹配、任务指标、设备时延 |

这份描述的用途不是保留一次调试过程，而是定义可重建的数值模型。模型、分辨率、上下文长度、校准数据或编译器版本改变后，应重新判断分布与设备支持，不能直接沿用旧 Scale 和旧结论。
