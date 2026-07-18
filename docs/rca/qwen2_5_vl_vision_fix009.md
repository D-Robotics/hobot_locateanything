# Qwen2.5-VL-3B Vision Fix #009 官方域对齐说明

更新日期：2026-07-17

## 1. 结论摘要

Fix #009 已经跑通 Qwen2.5-VL-3B 的自编译 Vision HBM，并能替换官方 Vision HBM，与以下官方组件配套运行：

```text
Fix #009 自编译 Vision
+ D-Robotics 官方 language HBM
+ D-Robotics 官方 embed.bin
+ D-Robotics 官方 libxlm 运行时
```

板端验证结果：

```text
Fix #009 Vision vs 官方 Vision cosine = 0.987963
image0.jpg：正确识别骑手、白马、红白跳杆
image1.jpg：正确识别木架上的小熊猫
```

本次跑通的是 **Vision 编译链和官方 Vision-Language 隐藏接口**。自编译 Language HBM 的乱码问题尚未解决，不能将本结果表述为“Qwen2.5-VL-3B 全模型已经完全自编译跑通”。

## 2. 官方产物与 Fix #009 产物不是同一个文件

S600 上的两个 HBM：

| 项目 | 官方 Vision HBM | Fix #009 自编译 Vision HBM |
|---|---|---|
| 路径 | `w4/official_vision.hbm` | `fix009_official_domain/Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm` |
| 文件大小 | 762,028,080 bytes | 762,029,104 bytes |
| SHA256 | `bfe60982d4bd19608668f00795a3ec82caf71095c5cdf2ced32e186745504409` | `d4511b8f910c25d8111056ce4cddf7652c91b59e05c3d95fdabc4dead0e94df8` |
| 来源 | D-Robotics 预编译下载 | 4090 上由 Fix #009 脚本重新校准、导出、转换和编译 |

两者 `cmp` 结果为不同文件。Fix #009 不是重命名、复制或重新链接官方 HBM。

它们公开的 graph ABI 相同：

```text
march: nash-p
graph: visual
input:  [1,1024,588] float16
output: [1,256,2048] float16
```

公开 ABI 相同只说明运行时接口兼容，不表示内部权重和编译图相同。

## 3. 官方运行链的实际接口

官方运行链可以简化为：

```text
JPEG/PNG
  -> libxlm ImagePreprocessor
  -> [1,1024,588] fp16 Vision 输入
  -> visual HBM
  -> [1,256,2048] fp16 Vision feature
  -> 注入 Language inputs_embeds
  -> official language HBM
```

这里有两个不能混淆的接口：

1. Vision HBM 的输入布局和预处理域；
2. Vision HBM 输出与 Language HBM 输入之间的 2048 维隐藏空间域。

Fix #009 同时处理了这两个问题：验证时使用官方运行时真实输入，并将 Vision 输出变换到官方 Language 期望的隐藏空间。

## 4. 为什么之前的官方 cosine 结论不可靠

早期纯 Vision 实验使用手工构造输入：

```text
/tmp/qwen_image0_1x1024x588_fp16.bin
```

后来通过 `LD_PRELOAD` hook 拦截 `hbDNNInferV2/V3`，抓取官方 `libxlm` 实际送入 `visual` graph 的输入：

```text
/tmp/official_visual_input_000.bin
shape=[1,1024,588]
dtype=float16
stride=[1204224,1176,2]
SHA256=a406e17ea359cac16937b64d744aaf4a59b538b6f14887e397bd3c4c9fe72a29
```

真实输入与旧手工输入只有：

```text
cosine = 0.22047679
max_diff = 3.8945312
mean_abs_diff = 0.8645985
```

因此，早期用手工输入得到的以下结论不能用来评价官方 ABI：

```text
official Vision vs HF cosine 约 0.008
official Vision vs self Vision cosine 约 0.007
```

这些数值比较的是错误输入条件下的输出，不是官方 `vlm` 正常运行时的对照。

## 5. 使用官方真实输入后的关键现象

使用同一份官方真实输入分别运行官方 Vision、旧 self Vision 和 Fix #007：

| Vision | 输出 std | 输出 L2 |
|---|---:|---:|
| 官方 Vision | 2.40466 | 1741.18 |
| 旧 self，temporal sum | 2.41118 | 1746.07 |
| Fix #007，temporal slice 0 | 1.54590 | 1119.42 |

cosine：

```text
official vs old self  = 0.0133040
official vs Fix #007 = 0.0091986
old self vs Fix #007 = 0.7151876
```

官方与旧 self 的输出范数和标准差几乎一致，但逐元素 cosine 接近 0。这不像随机权重或严重量化崩溃，更像同一语义向量使用了不同的隐藏空间坐标基。

## 6. 官方隐藏空间旋转的证据

### 6.1 Embed 证据

对比：

```text
official_embed.bin
self/HF embed_tokens.bin
shape=[151936,2048] float16
```

随机抽取 1024 个 token 后得到：

```text
同 token 逐行 cosine 中位数 = -0.00732
token 范数比中位数 = 0.999887
token 间 cosine Gram 矩阵相关系数 = 0.9999911
Gram 矩阵 mean absolute difference = 0.0002679
```

含义是：

- 官方 embed 与 HF embed 在逐维坐标上几乎不相关；
- 每个 token 的长度几乎不变；
- token 两两之间的夹角几乎完全保持。

这是统一正交变换的典型特征。

### 6.2 Orthogonal Procrustes 恢复

使用 8192 个 token 求解最接近的正交矩阵 `R`：

```text
official_embed ~= self_embed @ R
```

使用未参与拟合的 4096 个 token 验证：

```text
self embed raw vs official     cosine = -0.006927
self embed @ R vs official     cosine = 0.999817
rmse = 0.0004316
```

同一个 `R` 应用于旧 self Vision 输出：

```text
old self raw vs official       cosine = 0.013304
old self @ R vs official       cosine = 0.994467
```

这说明官方 embed、官方 Vision 输出和官方 Language 输入使用了同一个旋转后的 2048 维隐藏域。

### 6.3 Walsh-Hadamard 结构

恢复矩阵绝大多数元素的绝对值接近：

```text
1 / sqrt(2048) = 0.0220971
```

将符号矩阵与标准 2048 阶 Walsh-Hadamard 矩阵匹配后：

```text
2048 行均能一一对应唯一的 Hadamard 行
绝大多数行匹配分数为 2048/2048
```

由标准 Hadamard、行置换和行符号构造精确矩阵 `Q`：

```text
max(abs(Q.T @ Q - I)) = 5.96e-08
self embed @ Q vs official       cosine = 0.999801
old self Vision @ Q vs official  cosine = 0.996371
```

从数值结构看，这与 QuaRot/随机 Hadamard 旋转一致。由于官方没有提供 Qwen2.5-VL 从零编译源码，文档中应使用准确措辞：

> 已证实官方产物使用了可由 Walsh-Hadamard、行置换和行符号精确描述的隐藏空间正交变换；“QuaRot”是根据数值结构和 SDK 代码线索作出的机制判断，不是来自官方公开源码的函数名确认。

## 7. Fix #009 相比旧自编译具体修改了什么

### 7.1 恢复 temporal sum

HF checkpoint 的 patch embedding 权重为：

```text
[1280,3,2,14,14]
```

当前 HBM 接口使用 Conv2D 和单帧输入，因此将 temporal=2 折叠为：

```python
w4d = w5d.sum(dim=2)
```

静态图片在 HF 预处理后包含两个相同 temporal slice：

```text
x*w0 + x*w1 = x*(w0+w1)
```

纯 PyTorch patch embedding 验证：

```text
w0 only vs HF Conv3D cosine = 0.7030
w0+w1 vs HF Conv3D cosine = 1.0000
```

因此：

- Fix #007 的 `w5d[:,:,0,:,:]` 会丢失第二个 temporal kernel；
- Fix #009 使用 `sum(dim=2)`；
- Fix #007 经官方旋转后 cosine 只有 `0.71394`，进一步证明 slice 0 不是官方路径。

### 7.2 保留官方 window 输出顺序

HF Transformers Vision 在 merger 后执行：

```python
reverse_indices = torch.argsort(window_index)
hidden_states = hidden_states[reverse_indices]
```

加入该步骤可以使自定义 PyTorch Vision 对 HF 达到约 `0.99987`，但官方 runtime 的 Language 注入接口使用的是当前 window 顺序。

因此 Fix #009：

- 不添加 HF 最后的 `reverse_indices`；
- 保留 SDK 当前 window token 顺序；
- 以官方 Language 能正确消费为最终标准。

这体现了两个不同目标：

```text
对齐 HF 裸 Vision 输出：需要 reverse_indices
对齐官方 S600 Vision-Language ABI：不添加 reverse_indices
```

### 7.3 将官方隐藏旋转折叠进 merger

Vision merger 最后一层原始计算：

```text
y = x @ W.T + b
```

官方 Language 需要：

```text
y_official = y @ Q
```

等价地修改最后一层参数：

```python
W_new = Q.T @ W
b_new = b @ Q
```

折叠后的计算：

```text
x @ W_new.T + b_new
= (x @ W.T + b) @ Q
```

这样做的优点：

- 不增加新的 `2048x2048` runtime matmul；
- graph 输入输出 shape 不变；
- 不改变 `libxlm`；
- 不增加额外 BPU 推理阶段；
- 旋转随 merger 最后一层一起量化和编译。

纯 PyTorch 验证：

```text
post-rotated output vs folded-weight output cosine = 1.000000
max_diff = 2.19e-05
```

### 7.4 使用官方等价的编译参数

Fix #009 参数：

```text
march=nash-p
opt=2
jobs=16
core_num=4
input_no_padding=True
output_no_padding=True
enable_hpc=True
max_l2m_size=25165824
```

`jobs=16` 是主机并行编译任务数，`core_num=4` 是最终 BPU 模型使用的核心数，两者不是同一个概念。

### 7.5 重新执行完整 Vision 编译流程

Fix #009 不是在旧 HBM 上做二进制补丁，而是重新执行：

```text
加载 HF checkpoint
-> temporal sum 权重映射
-> merger 官方域旋转折叠
-> 120 条 mmstar 校准
-> export BC
-> convert MLIR
-> compile HBO
-> link HBM
```

编译结果：

```text
compile_hbo = 9408.8229 秒
link_models = 29.2811 秒
HBM size = 762,029,104 bytes
```

## 8. Fix #009 与官方实现的相同点和不同点

| 项目 | 官方预编译 Vision | Fix #009 |
|---|---|---|
| 输入 ABI | `[1,1024,588] fp16` | 相同 |
| 输出 ABI | `[1,256,2048] fp16` | 相同 |
| BPU march | `nash-p` | 相同 |
| BPU core | 4 core | 4 core |
| Runtime | 官方 `libxlm` | 使用同一官方 `libxlm` |
| 隐藏域 | 官方旋转域 | 通过恢复的精确 Hadamard 变换对齐 |
| temporal 处理 | 官方源码未公开 | 明确使用 `sum(dim=2)` |
| window token 顺序 | 官方 Language 可直接消费 | 保持同一 SDK 顺序 |
| 校准流程 | 未公开 | SDK mmstar 120 条 |
| 权重转换源码 | 未公开 | 项目 `leap_llm` + Fix #009 脚本 |
| HBM 内容 | 官方二进制 | 独立重新编译，SHA256 不同 |
| 数值结果 | 基准 | 与官方 cosine `0.987963` |

不能声称 Fix #009 与官方内部编译流程逐项完全相同，因为官方没有公开：

- 原始导出代码；
- Hadamard 置换/符号的生成方式或 seed；
- 精确校准集和校准参数；
- 编译前完整权重变换步骤；
- 官方 HBM 内部量化常量。

可以声称的是：Fix #009 在公开 ABI、隐藏空间、输出数值和端到端行为上已经与官方兼容。

## 9. 板端验证结果

固定官方真实输入：

```text
Fix #009: min=-21.4844 max=21.1094 mean=-0.01256 std=2.42887 L2=1758.72
official: min=-21.3594 max=21.2812 mean=-0.01183 std=2.40466 L2=1741.18
cosine = 0.9879630
max_diff = 7.08887
rmse = 0.375698
```

端到端组合：

```text
Vision: Fix #009
Language: official_lang.hbm
Embed: official_embed.bin
```

结果：

- `image0.jpg`：官方和 Fix #009 都识别骑手、白马、跳杆和户外马术场景；
- `image1.jpg`：官方和 Fix #009 都识别木架上的小熊猫及其红褐色、白色面部特征。

这两张图证明 Fix #009 不是只在单一输入上数值拟合，而是能被官方 Language 正确消费。

## 10. 当前成功边界

已经完成：

```text
自编译 Qwen2.5-VL-3B Vision
+ 官方 embed
+ 官方 Language
= 正常多模态推理
```

尚未完成：

```text
自编译 embed
+ 自编译 Language
+ 自编译 Vision
= 完整自编译 Qwen2.5-VL-3B
```

自编译 Language 当前仍输出乱码。Fix #009 解决的是 Vision 输出到官方 Language 输入之间的隐藏域 ABI，不会自动修复 Language HBM 内部的 QuaRot、量化或权重变换问题。

## 11. 对 LocateAnything 部署的意义

LocateAnything 使用 Qwen2.5 语言 decoder，但 Vision 是 MoonViT。Fix #009 提供了两个可迁移经验：

1. 不能只保证 tensor shape 一致，还必须保证 Vision 输出和 Language 输入处于同一隐藏空间域；
2. 官方 QuaRot/Hadamard 变换可以折叠进 Vision projector/merger 最后一层，避免新增运行时算子。

但不能直接把 Qwen2.5-VL 的 `Q` 无条件用于 LocateAnything：

- 需要先确认 LA 的 hidden size 是否同为 2048；
- 需要确认目标 Language HBM/embed 使用哪个隐藏域；
- 如果最终采用完全自编译的原生 HF 域 Language，则 MoonViT projector 可能不需要官方 `Q`；
- 如果 LA Vision 要接官方旋转域 Language，则需在 LA projector 输出端应用与该 Language/embed 配套的同一个 `Q`。

因此 Fix #009 的核心不是“固定使用某个矩阵”，而是建立并验证跨 Vision-Language 的隐藏域契约。

## 12. 关键产物

4090：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix009_vision_official_domain/
```

包含：

```text
vision.visual.bc
vision.visual_convert.bc
vision.visual.hbo
Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm
official_hidden_rotation_exact.pt
official_hadamard_row_permutation.npy
official_hadamard_row_sign.npy
compile.jobs16.log
```

S600：

```text
/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/fix009_official_domain/
/home/sunrise/oe_locateanything/oellm_runtime/examples/vlm_demo/test_fix009_official_lang.json
```

可复现代码保存在当前工作区：

```text
tools/compile_fix009_official_domain.py
tools/dump_visual_input_hook.cpp
tools/dump_visual_input_hook.map
```

## 13. 最终判断

Fix #009 的成功不是因为 HBM 文件接近官方文件，也不是通过复制官方 HBM 实现，而是因为找到了并补齐三个实际契约：

```text
正确的 temporal patch 语义
+ 官方 libxlm 的真实输入 ABI
+ 官方 Vision/Embed/Language 共用的 Hadamard 隐藏空间
```

因此可以做出以下工程结论：

> Qwen2.5-VL-3B Vision 已经具备从 HF checkpoint 经自定义 Leap/HBDK 编译后替换官方 Vision HBM 的能力；在官方 Language 和官方 embed 条件下，板端数值与语义均验证通过。
