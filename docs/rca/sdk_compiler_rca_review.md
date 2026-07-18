# sdk compiler rca review

## 目标

将 LocateAnything-3B 部署到 D-Robotics S600 BPU。LocateAnything 由 MoonViT vision encoder、MLP projector、Qwen2.5-style language decoder 和 PBD 组成。当前先以 Qwen2.5-VL-3B 为基线，优先打通 Vision 侧，再迁移到 MoonViT/LA。

## 机器与环境

- 4090 编译机：`10.112.20.45`，用户 `kangjie.xu`，可免密 SSH。
- S600 部署机：`10.112.133.20`，用户 `sunrise`，可免密 SSH。
- 4090 编译环境：conda `oellm_clean`，Python 3.10，hbdk4 约 4.10.2a2，march `nash-p`。
- S600 runtime：`~/oe_locateanything/oellm_runtime`，HBRT4 约 4.9.6。

## 模型文件

S600 当前目录：`/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/`

官方套件在 `w4/`：

- `official_lang.hbm`
- `official_vision.hbm`
- `official_embed.bin`

自编译修复版在 `self_compiled/`：

- `Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm`
- `Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm`
- `Qwen2.5-VL-3B-Instruct_embed_tokens.bin`

不要使用 `w4/` 下旧的 0 字节 embed 和 285M 旧 Vision HBM。

## 已知端到端结果

| Language | Vision | Embed | 结果 |
|---|---|---|---|
| 官方 | 官方 | 官方 | 正常，可识别骑白马图片 |
| 官方 | 自编译 | 官方 | 可运行但视觉语义错误 |
| 自编译 | 官方 | 官方 | 乱码，说明自编译 Language 独立有问题 |
| 自编译 | 自编译 | 自编译 | 乱码 |

因此本轮固定官方 Language + 官方 Embed，只研究 Vision。

## 本轮实验：HBM descriptor

在 4090 使用 `Hbm`/`hbm_extract_desc` 读取官方和自编译 Vision HBM：

- graph name：`visual`
- input：`_input_0`
- input shape：`(1, 1024, 588)`
- input dtype：`float16`
- output：`_output_0`
- output shape：`(1, 256, 2048)`
- output dtype：`float16`
- march：`NashP`

两份 HBM 接口完全一致。

## 本轮实验：同输入直接在 S600 跑 Vision HBM

使用同一份 `image0.jpg`，生成 `[1,1024,588]` 输入，在 S600 分别调用：

```bash
hbrt4-run-model-nash -f official_vision.hbm -n visual ...
hbrt4-run-model-nash -f self_compiled_vision.hbm -n visual ...
```

两份 HBM 都成功运行，说明加载、权限、shape、dtype、runtime 基本正常。

对输出做 float32 比较：

- normalized 输入：全局 cosine `0.0073028`
- official 输出统计：约 `[-10.20, 12.34]`，mean `0.0102`，std `1.3646`
- self 输出统计：约 `[-26.16, 28.83]`，mean `-0.0087`，std `1.3611`
- mean absolute error：约 `1.3864`

这不是普通量化误差，而是两套模型输出几乎不相关。

使用运行时实际 `image_preprocess.cpp` 的 `/255` 输入重新测试：

- official 与 self cosine `0.0121`

因此 mean/std 归一化差异不是根因。

## 本地 PyTorch 对照

`leap_llm` 的 `ModuleMeta` 会把 `forward()` 替换成编译用 `build()`。通过 `compile_mode(False)` 恢复纯 Torch 路径后：

- 自编译 Vision HBM vs 当前 leap_llm PyTorch Vision：cosine `0.9612`
- 官方 Vision HBM vs 当前 leap_llm PyTorch Vision：cosine `0.0080`

这说明自编译 HBM 基本忠实实现了当前 Python 编译模型；官方 HBM 并不是当前这套裸 `temporal sum + Conv2d` 语义的量化近似。

使用 Hugging Face 原始 Qwen2.5-VL Vision encoder，输入 temporal pair 并设置 `grid_thw=[1,32,32]`：

- HF 原始 Vision vs 官方 HBM：cosine `0.0045`
- HF 原始 Vision vs 自编译 HBM：cosine `0.3964`
- HF 原始 Vision vs 当前 leap_llm PyTorch Vision：cosine `0.3995`

HF 与 leap_llm 当前实现本身也不完全一致，但官方 HBM 显然不是简单的 HF 输出量化误差。

## 权重加载检查

当前 `Qwen2_5_VL.build()` 日志：

```text
[FIX #005] proj_2d weight folded: [1280, 3, 2, 14, 14] -> [1280, 3, 14, 14]
miss_key: ['model.language_model.cache_cos', 'model.language_model.cache_sin']
unexpected_key: []
```

Vision 侧没有 missing/unexpected key。缺少的只是语言侧预计算 RoPE cache，与 Vision 无关。

## Fix #005

文件：`/home/kangjie.xu/oe_locateanything/toolchain/leap_llm/models/qwen2_5_vl/model.py`

原问题：HF checkpoint 的 `model.visual.patch_embed.proj.weight` 是 `[1280,3,2,14,14]`，而 `use_conv2d=True` 的 `proj_2d` 需要 `[1280,3,14,14]`。原来的 `strict=False` 会静默跳过 shape mismatch，导致随机 patch embedding。

当前修复是：

```python
w4d = w5d.sum(dim=2)
new_state_dict['model.visual.patch_embed.proj_2d.weight'] = w4d
```

该修复必要，但本轮结果表明它不足以复现官方 Vision HBM。

## 运行时源码发现

S600 运行时文件：

`/home/sunrise/oe_locateanything/main/runtime/src/image_preprocess.cpp`

当前 `BuildVisionPatchTensor()`：

- 输入固定 448x448 RGB
- 输出 `[1,1024,588]`
- patch 顺序：`py * 32 + px`
- 每个 patch 内顺序：`cy -> cx -> channel`
- uint8 输入只执行 `/255.0f`
- 没有执行 mean/std normalization
- `temporal_patch_size` 配置为 `1`

因此已排除“官方/自编译仅因 mean/std 不同导致输出差异”。

## 对当前根因的判断

目前最可能的根因按优先级：

1. 官方 Vision HBM 使用了与当前自编译代码不同的 patch embedding temporal 处理/权重预处理；
2. 官方 Vision 编译版本有未公开的 Qwen2.5-VL 专用转换逻辑，不能通过简单的 Conv3d temporal sum 复现；
3. 官方 HBM 与当前模型 checkpoint/SDK 版本可能不是同一导出链版本；
4. 其次才是 Vision block 内部实现、window index、rotary position 或 merger 的细节差异。

当前没有证据支持继续盲调校准数据；输出 cosine 只有约 0.007，首先应解决语义/结构对齐。

## 下一步建议

1. 从官方运行时/SDK 中定位官方 Qwen2.5-VL Vision 的输入预处理与 patch embedding 约定，尤其确认官方 HBM 是否真的接收 `[1,1024,588]` 的普通 patch 顺序。
2. 用 `hbrt4-run-model-nash` 对官方 HBM 生成一组可重复输入，尝试对输入做有限排列变换：HWC/CHW、patch 内 channel-first/channel-last、row-major/column-major；以官方 HBM 输出与 HF Vision 的相关性为判据。
3. 检查官方预编译包是否带有原始 config、模型版本、校准 manifest 或转换日志。
4. 只有找到能使官方 HBM 与参考模型相关的输入/权重语义后，再进行最小 Vision 重编；暂不重编 Language。
5. 记录每次 Vision HBM 的输入排列、权重折叠方式、校准集和输出 cosine，避免把 Language 乱码问题混入。

## 论坛参考的事实边界

论坛帖子 `https://forum.d-robotics.cc/t/topic/35332` 是开发者个人成功案例，不是官方从零编译教程。它只能作为经验参考，不能当作 D-Robotics 官方 Qwen2.5-VL 编译规范。

## 第二轮实验：输入布局扫描（2026-07-16）

为排除官方 HBM 的输入排列约定，固定同一张 `image0.jpg`，生成 12 种 `[1,1024,588] float16` 输入并在 S600 上逐一运行官方 `official_vision.hbm`：

- raw / normalized
- RGB / BGR
- channel-last / channel-first / H-W 转置 patch 顺序

与 HF Qwen2.5-VL Vision 输出比较，结果如下：

| 输入变体 | cosine |
|---|---:|
| normalized RGB channel-first | 0.0100 |
| normalized BGR channel-first | 0.0064 |
| normalized BGR H-W | 0.0063 |
| raw BGR channel-first | 0.0051 |
| raw RGB channel-first | 0.0054 |
| normalized RGB H-W | 0.0054 |
| normalized RGB channel-last | 0.0045 |
| 其他变体 | 0.0031 以下 |

没有任何布局使 cosine 显著升高；因此已基本排除：

- RGB/BGR 误差；
- patch row-major/column-major 误差；
- patch 内 channel-first/channel-last 误差；
- 运行时 `/255` 与 HF mean/std 归一化差异。

## 第二轮结论

官方 HBM 不是当前可见输入语义下的同一 Vision 模型量化版本。其输出与 HF、自编译 PyTorch、自编译 HBM 都接近不相关；而自编译 HBM 与自己的 PyTorch 模型 cosine 约 0.961，说明自编译编译链并非随机损坏。

本地 S600 SDK 和文档只提供 Qwen2.5-VL 的预编译 HBM 配置，没有找到从 HF checkpoint 重新导出 Vision 的官方脚本或 manifest。论坛 Gemma4 帖子也只能作为民间经验，不能解释官方 Qwen HBM 的隐藏转换。

## 下一步决策

不再继续盲猜官方 HBM 的输入排列，也暂不修改 Language。下一步采用可控路线：

1. 以 HF 原始 Vision 输出作为参考，而不是以官方 HBM 作为编译目标；
2. 对自编译 Vision 的 patch embedding 做两个明确变体：
   - temporal weight `sum(dim=2)`；
   - temporal weight `mean(dim=2)`；
3. 保持其余 Vision block、输入 shape、校准集完全不变，只替换一个变体并生成 HBM；
4. 在 4090 先做纯 Torch 输出对比，再上传 S600 与官方 HBM 做端到端语义测试；
5. 如果两个变体均不能使 HF 对齐，再检查 Qwen Vision block 的 rotary/window/merger，而不是继续调校准。

注意：`mean` 变体只差一个固定的 temporal 缩放因子，不能预期它单独解释 cosine≈0.007 的巨大差异；它的价值主要是验证官方导出是否采用平均折叠。若需要真正复现 HF Conv3d，需要重新设计输入接口或显式构造 temporal=2 的模型路径，而不是仅修改 4D Conv 权重。

## 论坛 Gemma4 成功案例的可迁移经验

已重新读取开发者 Shockley 的帖子及其 GitHub 仓库 `shockley6668/gemma4-e2b-rdk-s100p`。该帖子是个人成功案例，不是官方教程，但对本项目有直接参考价值。

### 经验一：不要把官方 HBM 当作编译目标

案例作者自己重写 `leap_llm_gemma4` 模型定义，明确实现 Vision PatchEmbedding、位置编码、ViT block、Pooler、Projector，并以 HF/PyTorch 语义作为参考；官方预编译 HBM 只用于部署参考，不用于猜测内部实现。

### 经验二：校准路径必须和板端输入路径一致

Gemma4 案例流程是：真实图像预处理 → patchify → `compile_mode(False)` 运行校准 → 切回 compile mode → 导出 BC → 转换/编译 HBM。板端 runtime 使用同样的 resize、归一化和 patch 顺序。

### 经验三：必须做 Golden/统计闭环

案例不只看最终文本，而是检查：Vision feature 的统计范围、Vision→Text 注入后的 `inputs_embeds`、mask、KV cache，并用 cosine/max_diff 做对齐。其经验中，Vision feature std 异常是语义错误的重要信号。

### 对 Qwen2.5-VL 的直接映射

当前 Qwen 编译 API 已经执行了真实图像校准流程：

```text
compile_mode(False)
_calibrate_forward()
compile_mode(True)
export visual BC
convert/compile HBM
```

但校准路径中的 Qwen `remove_repeat()` 为：

```python
pixel_values = pixel_values.reshape([-1, 3, 2, 14, 14])
pixel_values = pixel_values[:, :, 0]
```

它保留 temporal slice 0，最终喂给 `[1,1024,588]` 的 Vision graph。

旧 Fix #005 却在权重导出时执行：

```python
w4d = w5d.sum(dim=2)
```

这造成了编译时权重语义和校准/板端输入语义不一致。已新增 Fix #006：

```python
w4d = w5d[:, :, 0, :, :]
```

并同步修改：

- `/home/kangjie.xu/oe_locateanything/toolchain/leap_llm/models/qwen2_5_vl/model.py`
- `/home/kangjie.xu/miniforge3/envs/oellm_clean/lib/python3.10/site-packages/leap_llm/models/qwen2_5_vl/model.py`

模型加载已经确认：

```text
[FIX #006] proj_2d weight uses temporal slice 0
miss_key: language_model.cache_cos/cache_sin only
unexpected_key: []
```

### 当前执行策略

先只重新导出/编译 Vision HBM，不重编 Language。新产物使用独立目录和文件名，保留旧的 sum 版本用于 A/B 对比。验证顺序：

1. Vision BC/PyTorch 输出统计；
2. 4090 编译 HBM；
3. S600 同一 `[1,1024,588]` 输入对比；
4. 固定官方 Language + 官方 Embed 做端到端图片测试；
5. 记录 feature cosine、mean/std、端到端文本结果。

这比继续逆向官方 Vision HBM 更可控，也符合论坛成功案例的核心方法。

## Fix #006 / Fix #007 RCA 记录

本节记录每一次修复的目标、实际生效情况、证据和遗留问题。后续每次修改都应继续追加到本节，不能只保留最终版本。

### Fix #006：将 patch embedding 的 temporal 权重改为 slice 0

#### 发现的问题

Qwen2.5-VL 的 checkpoint 中：

```text
model.visual.patch_embed.proj.weight: [1280, 3, 2, 14, 14]
```

当前运行时/校准路径通过 `remove_repeat()` 将输入 reshape 为 `[3, 2, 14, 14]` 后保留 temporal slice 0，最终 Vision 图输入为 `[1, 1024, 588]`。旧导出逻辑却执行：

```python
w4d = w5d.sum(dim=2)
```

这使得 4D Conv2D 权重与输入的 temporal 语义不一致。

#### 修改

在 `Qwen2_5_VL._load()` 中改为：

```python
w4d = w5d[:, :, 0, :, :]
```

涉及文件：

- `/home/kangjie.xu/oe_locateanything/toolchain/leap_llm/models/qwen2_5_vl/model.py`
- `/home/kangjie.xu/miniforge3/envs/oellm_clean/lib/python3.10/site-packages/leap_llm/models/qwen2_5_vl/model.py`

#### 未生效原因

Fix #006 只修改了加载权重时的 `proj_2d.weight`，但 `vision_embedding.py` 的 `forward()` 在校准阶段仍执行：

```python
weight_2d = self.proj.weight.data.sum(2)
self.proj_2d.weight.data = weight_2d
```

因此校准和导出时又把 slice-0 权重覆盖回 temporal sum。Fix #006 生成的 HBM 实际仍等价于旧的 temporal-sum 版本。

#### 验证证据

- Fix #006 HBM SHA256：`0e048936c8a3a14ca0e2204b9cde39ecfaabdc38ac0895ce863d464841ec941c`。
- Fix #006 HBM 已上传至 S600：`/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/fix006_slice0/`。
- Fix #006 与旧自编译 Vision HBM 在同一输入上的输出 cosine 为 `1.0`，max diff 为 `0`。

结论：Fix #006 的设计方向正确，但由于 forward 覆盖逻辑未删除，最终部署产物没有实现预期修复。

### Fix #007：删除 forward 阶段的 temporal sum 覆盖

#### 根因修复

删除 `Qwen2_5_VisionPatchEmbed.forward()` 中的：

```python
weight_2d = self.proj.weight.data.sum(2)
self.proj_2d.weight.data = weight_2d
```

这样 `proj_2d.weight` 保持模型加载阶段生成的 temporal slice 0，不会在真实图像校准时被重新覆盖为 temporal sum。

涉及文件：

- `/home/kangjie.xu/oe_locateanything/toolchain/leap_llm/nn/modules/vision_embedding.py`
- `/home/kangjie.xu/miniforge3/envs/oellm_clean/lib/python3.10/site-packages/leap_llm/nn/modules/vision_embedding.py`

#### 纯 PyTorch 证据

Fix #007 的 Vision 输出与旧 temporal-sum 输出 cosine 约为 `0.6178`，说明 forward 不再复写为旧的 temporal-sum 权重。

#### BC 静态证据

Fix #006 和 Fix #007 的 BC 接口都为：

```text
input:  float16 [1, 1024, 588]
output: float16 [1, 256, 2048]
```

转换后的 BC 对比：

| 项目 | Fix #006 | Fix #007 |
|---|---:|---:|
| `vision.visual.bc` 字节数 | 670,254,124 | 670,250,094 |
| `vision.visual_convert.bc` 字节数 | 671,024,768 | 671,024,818 |
| 转换图操作数 | 5,239 | 5,240 |
| `hbdk.constant` 数量 | 946 | 947 |

逐操作对齐显示，Fix #007 在 `model.visual.blocks.13.norm1` 附近新增一个 RMSNorm 相关常量，并且多个 RMSNorm 的转换参数发生变化。例如：

```text
Fix #006 RMSNorm 参数：9.999999e-7
Fix #007 RMSNorm 参数：8.310547e-7
```

类似差异还出现在 `blocks.13` 至 `blocks.31` 以及 `model.visual.merger.ln_q`。这表明 Fix #007 改变了校准阶段的激活分布，量化参数随之重新计算；它不是 Fix #006 BC 的简单复制。

#### 当前状态

- Fix #007 已生成 `vision.visual.bc` 和 `vision.visual_convert.bc`。
- Fix #007 的 HBO/HBM 仍在编译，尚未上传 S600。
- 当前编译任务：`PID 3416376`，使用 `jobs=4`。
- 当前 BC 静态分析没有发现输入输出接口变化。

#### 当前结论

Fix #007 相比 Fix #006 的主要改进不是改变 runtime 接口，而是修复了 Fix #006 遗漏的第二处权重覆盖点：让 temporal slice 0 的权重真正贯穿模型加载、校准、导出和转换流程。BC 证据已经证明该修改进入了转换图；最终是否改善 S600 Vision 输出，必须等待 HBM 后用同一输入计算 cosine、mean/std 和端到端语义结果。

### 后续记录规则

每个新 Fix 必须记录：

1. 修改的源码文件和具体逻辑；
2. 试图修复的根因；
3. 是否真正进入 BC/HBO/HBM；
4. BC 接口、算子数量和关键常量变化；
5. 4090 纯 Torch 对比；
6. S600 同输入 cosine、max diff、mean/std；
7. 官方 Language + 官方 Embed 固定条件下的端到端结果；
8. 未解决的问题和下一步实验。

## Temporal 输入路径澄清：`remove_repeat()` 的含义与证据等级

当前代码：

```python
pixel_values = pixel_values.reshape([-1, 3, 2, 14, 14])
pixel_values = pixel_values[:, :, 0]
```

### 代码事实

假设 `pixel_values` 原始第一维表示展开后的 patch/token 数量，则第一步把每个 patch 重新解释为：

```text
[3, 2, 14, 14]
= [RGB 通道, temporal 位置, patch 高度, patch 宽度]
```

第二步 `[:, :, 0]` 保留 temporal 维度的第 0 个位置，结果形状为：

```text
[patch 数量, 3, 14, 14]
```

最后再 reshape 回 `[tokens, 588]`，其中 `588 = 3 × 14 × 14`。因此当前自定义 Conv2D Vision 图实际没有接收 temporal=2 的两个输入位置。

### 方法来源

这不是从 D-Robotics 官方 Qwen2.5-VL 编译教程获得的，也不是官方对 HBM 内部处理方式的说明。它直接来自当前 `leap_llm/apis/model/qwen2_5_vl.py` 中已有的 `remove_repeat()` 实现；其存在说明当前适配代码作者曾为 Conv2D/输入接口做过 temporal 展平处理。

Fix #006/#007 的启发来自两点：

1. 发现输入侧已有 `[:, :, 0]`，而权重侧仍有 `w5d.sum(dim=2)`，两条路径在 temporal 语义上不一致；
2. 论坛 Gemma4 个人案例强调要逐层对齐 HF/PyTorch、校准、BC、HBM 和 runtime，而不能把官方预编译 HBM 当作编译规则。

### 正确性等级

目前只能确认：

- `remove_repeat()` 是当前代码真实执行的输入变换；
- `w5d[:, :, 0, :, :]` 与该输入变换在形式上匹配；
- Fix #007 的 BC 已经显示该修改进入转换图。

目前不能确认：

- `remove_repeat()` 是否完整复现 Qwen2.5-VL 原始 HF 静态图片语义；
- slice 0 是否比 `w0+w1` 更接近 HF；
- D-Robotics 官方 HBM 是否采用相同 temporal 处理。

因此 `slice 0` 是一个可验证的工程假设，不是已被官方或端到端实验完全证明的结论。后续应比较：原始 Conv3D、重复帧加 `w0+w1`、单帧加 `w0`、以及必要时的 `mean(w0,w1)`。

## 官方编译与当前自编译的差异、以及耗时原因

### 已知差异

官方提供的是已经生成好的 HBM，内部的权重预处理、校准样本、图改写、量化参数和编译缓存均不可见。当前自编译流程则是：

```text
HF checkpoint
→ 自定义 leap_llm 模型适配
→ 自定义 calibration forward
→ export BC
→ convert MLIR/BC
→ HBDK compile HBO
→ link HBM
```

因此当前编译不是“用同一个官方编译命令重跑”，而是自行重建官方未公开的导出和量化流程。尤其是当前 Vision 图把原始 Conv3D 适配成 Conv2D，并运行了 120 个 mmstar 图像校准样本。

### 为什么当前 Vision 编译慢

最耗时的是 `compile HBO`，不是模型加载、校准或 BC 转换。HBDK 在这一阶段会用 `qemu-system-riscv64` 和 `hbcm-module-entry` 模拟 Nash-P 的 BPU 调度、内存规划和算子编排。`jobs=4` 是 4 组 BPU 仿真编译任务，不等同于普通的 4 个轻量 CPU 线程。

当前 Fix #007 之前还与两个旧验证任务争抢 CPU；这两个旧任务已经清理，未影响 Fix #007。此前 `jobs=16` 曾出现长时间同步等待，因此不能简单认为增加 jobs 就一定线性加速。

### 为什么之前总耗时约 5 小时

之前的“约 5 小时”不能直接作为当前 Vision HBO 阶段的基准，可能混合了不同 SDK 版本、不同图复杂度、不同 jobs、缓存状态和不同阶段的耗时。当前 Fix #007 的转换图有 5,240 个操作、65 个 `b30vpu.call`、259 个量化节点，并且量化参数因 temporal 修复重新校准；最终 HBO 编译可能比前面的 BC 导出/convert 慢很多。

准确结论必须以每个阶段的开始/结束时间为准，不能仅凭日志停在 `[6] compile HBO` 判断百分比。

## 重要事实更正：`remove_repeat()` 来源与历史编译耗时

### `remove_repeat()` 不是官方原始实现的已确认结论

此前将 `/home/kangjie.xu/oe_locateanything/toolchain/leap_llm/apis/model/qwen2_5_vl.py:161` 中的 `remove_repeat()` 当作 Qwen2.5-VL 官方原始处理逻辑，这是不严谨的。该文件位于项目自己的 `toolchain/leap_llm/` 中，当前证据只能证明它是本项目适配代码的一部分，不能证明它来自 Qwen 官方实现或 D-Robotics 官方 SDK。

在官方 SDK 原始目录：

```text
/home/kangjie.xu/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK/
```

未检索到对应的 `remove_repeat()`、Qwen2.5-VL Python 模型定义或从 HF checkpoint 开始的 Vision 导出代码。因此：

- `remove_repeat()` 是当前自定义适配链中的代码事实；
- `w5d[:, :, 0, :, :]` 是为了匹配这段自定义输入路径提出的工程假设；
- 不能把它描述为官方 Qwen2.5-VL 的标准 temporal 处理方式；
- 也不能据此断言官方 HBM 使用了 slice 0。

### “官方编译”名称更正

此前把官方预编译 HBM 和之前的自编译过程混称为“官方编译”，应更正为：

- `official_lang.hbm` / `official_vision.hbm`：D-Robotics 提供的预编译部署产物；
- 之前 LocateAnything Vision/Language HBM：本项目自己调用 HBDK 编译器完成的自编译产物；
- 当前 Qwen2.5-VL Fix #006/#007：本项目在 `oellm_clean` 环境中重新构建的另一套自编译流程。

### 历史日志中的真实耗时

历史 LocateAnything Vision 日志：

```text
/home/kangjie.xu/oe_locateanything/main/logs/locateanything_vit_compile.log
```

记录：

```text
export_module: 0.8466 s
convert_mlir: 2.7739 s
compile_hbo: 3900.7447 s（约 65.0 分钟）
link_models: 18.2907 s
```

历史 LocateAnything Language 日志：

```text
/home/kangjie.xu/oe_locateanything/main/logs/locateanything_language_compile.log
```

记录：

```text
prefill compile_hbo: 4975.9805 s（约 82.9 分钟）
decode compile_hbo: 3665.0050 s（约 61.1 分钟）
```

因此之前自编译 Vision + Language 的 HBO 阶段合计约：

```text
3900.7447 + 4975.9805 + 3665.0050 = 12541.7302 s ≈ 3.48 小时
```

加上导出、转换和链接后，整体约 3.5～4 小时是合理的，与你记忆中的“约 5 小时”基本一致。

### 当前 Fix #007 为什么不能直接按历史时间估计

当前 Fix #007 的日志：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix007_vision_slice0.compile.jobs4.log
```

在 `2026-07-17 10:43:30` 进入：

```text
[6] compile HBO
```

其转换图为约 5240 个操作，并由 `q.model.compile_hbo(...)` 启动。历史 LocateAnything Vision 也是约 3900 秒，但当前 Fix #007 截至此前已经持续超过该基准，说明两者至少存在一个或多个实质差异：

1. 自定义 Qwen2.5-VL 图与 LocateAnything MoonViT 图不是同一图；
2. 当前 `oellm_clean` 与历史 `oellm` 环境不同；
3. 当前 Fix #007 使用的 HBDK/compiler 版本或编译参数不同；
4. 当前调用中使用了 `jobs=4`，而历史日志需要继续从启动脚本/环境确认并行参数；
5. 当前 Qwen 图的量化节点、VPU 节点、内存规划或常量布局更难编译；
6. 历史任务可能使用了不同的编译缓存/临时资源状态。

目前不能再简单说“之前也要十几个小时”或“当前正常只是更复杂”。已确认的事实是：之前自编译 Vision 的 HBO 阶段约 65 分钟，而当前 Fix #007 已明显超过这个量级，应该进一步调查环境、参数和编译器行为。

## 再次更正：当前应只与 Qwen2.5-VL 历史自编译比较

前一版记录错误地引用了 LocateAnything/MoonViT 的编译日志作为当前 Qwen2.5-VL 编译的时间参照。该比较无效，已在本节纠正。当前目标一直是先打通 Qwen2.5-VL-3B，再迁移到 LocateAnything；后续不得用 LA 编译耗时替代 Qwen 基线耗时。

### Qwen2.5-VL 历史自编译证据

历史日志：

```text
/home/kangjie.xu/oellm_clean/compile.log
```

这份日志与当前任务使用同一个 `oellm_clean` 环境，代码来自 SDK OELLM 开发人员提供并安装到该环境中的 OELLM/leap_llm 代码，不是 LA 的模型编译日志。

历史 Qwen2.5-VL Vision：

```text
Function 'build' done in 7.8357 seconds
Function 'convert_mlir' done in 7.6506 seconds
compile_hbo: 9273.0636 seconds ≈ 154.6 分钟
```

历史 Qwen2.5-VL Language：

```text
prefill compile_hbo: 约 4077.7655 秒 ≈ 68.0 分钟
                         （以该历史日志对应的阶段记录为准）
decode compile_hbo: 3796.5611 秒 ≈ 63.3 分钟
```

历史日志中记录的并行参数为：

```text
Vision:  jobs=16
Decode:  jobs=32
```

历史 Qwen 输出产物时间也对应这一流程：

```text
Vision HBM: 2026-07-16 00:06
Language HBM: 2026-07-16 00:05
```

### 当前 Fix #007 与历史 Qwen 编译的关键差异

当前 Fix #007 使用：

```text
环境：oellm_clean
Vision：Qwen2.5-VL-3B
jobs=4
```

而历史 Qwen 基线使用：

```text
环境：oellm_clean
Vision：Qwen2.5-VL-3B
jobs=16
```

因此当前 Fix #007 和历史 Qwen 自编译不是同一编译参数。当前 jobs=4 的选择是因为此前 jobs=16 曾出现长时间同步等待，但历史日志证明 jobs=16 在完整 Qwen 基线中确实成功完成过，并且 Vision HBO 耗时约 2.6 小时。

当前 Fix #007 已经在 `[6] compile HBO` 阶段运行超过此前历史 Vision 的 154.6 分钟基准，说明需要重点检查：

1. `jobs=4` 相比历史 `jobs=16` 带来的并行度差异；
2. Fix #007 转换图与历史 Qwen Vision 转换图的操作/量化/内存规划差异；
3. 当前任务是否存在 HBDK 子进程异常等待；
4. 是否可以在保留 BC 的前提下，用历史成功参数 `jobs=16` 重新进行一次受控 A/B 编译。

目前不能说当前变慢是因为 Qwen 图天然比 LA 图复杂；正确表述是：当前 Qwen Fix #007 使用了与历史 Qwen 基线不同的 `jobs`，且当前修改改变了转换图和量化参数。后续性能比较只使用 `/home/kangjie.xu/oellm_clean/compile.log` 及同一 Qwen 编译链的日志。

## Qwen2.5-VL Fix #007 编译耗时 RCA（仅与 Qwen 历史自编译比较）

### 对比对象

本次只比较同一 Qwen2.5-VL-3B 编译链：

- 历史成功基线：`/home/kangjie.xu/oellm_clean/output/Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.visual_convert.bc`
- Fix #006：`/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix006_vision_slice0/vision.visual_convert.bc`
- Fix #007：`/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix007_vision_slice0/vision.visual_convert.bc`

不再使用 LocateAnything/MoonViT 的编译日志作为 Qwen 基线。

### BC 对比结果

历史基线和 Fix #006 的转换 BC 完全相同：

```text
size   = 671,024,768 bytes
SHA256 = 14d526cd9a79fff5714108627d4ecd30aa1e2a90a775c5d643962e62fbde16c8
ops    = 5239
b30vpu.call = 65
b30.quantize = 259
b30.conv2d = 227
hbdk.constant = 946
```

这证明 Fix #006 最终没有改变转换图，也与之前 S600 输出和旧自编译版本完全一致的结论相符。

Fix #007：

```text
size   = 671,024,818 bytes
SHA256 = acbbf46d9e280689e6e5d3105564517031d7201b0b662dd1d9dfef344b6e01a
ops    = 5240
b30vpu.call = 65
b30.quantize = 259
b30.conv2d = 227
hbdk.constant = 947
```

因此 Fix #007 只增加 1 个转换操作和 1 个常量，核心算子数量没有增加：

```text
b30vpu.call、quantize、conv2d 数量均未增加
```

Fix #007 的变化主要是 temporal 修复后重新校准导致的常量/量化参数变化，而不是图规模成倍增加。仅从 BC 图规模看，没有证据说明 Fix #007 应该比历史基线慢数量级。

### 历史 Qwen Vision 编译参数和耗时

历史日志：

```text
/home/kangjie.xu/oellm_clean/compile.log
```

历史 Qwen Vision 使用：

```text
jobs = 16
core_num = 4
march = nash-p
opt = 2
input_no_padding = True
output_no_padding = True
enable_hpc = True
max_l2m_size = 25165824
```

耗时：

```text
build = 7.8357 s
convert_mlir = 7.6506 s
compile_hbo = 9273.0636 s ≈ 154.6 min
```

### Fix #007 当前编译状态

当前任务：

```text
PID 3416376
script /tmp/compile_fix007_vision_jobs4.py
jobs = 4
```

当前日志：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix007_vision_slice0.compile.jobs4.log
```

已进入：

```text
[6] compile HBO
```

当前 Python 父进程仍有约 400% CPU，占用约 4 个逻辑核；没有旧 Qwen verifier 或 LA sanity 任务继续抢占资源。父进程处于 `futex_wait_queue`，但其累计 CPU 时间持续增加，不能仅凭该状态判定死锁。当前没有生成 `.hbo` 或 `.hbm`。

### 当前结论

1. 历史 Qwen 基线 BC 与 Fix #006 完全相同，因此不能把 Fix #006 的慢归因于图变化；
2. Fix #007 相比历史基线只多 1 个操作和 1 个常量，不能解释长数量级耗时；
3. 历史 Qwen Vision 用 `jobs=16` 成功在约 154.6 分钟完成；
4. 当前 Fix #007 使用 `jobs=4`，且已超过历史基线耗时；
5. 当前最强嫌疑是并行度差异 `jobs=4 vs jobs=16`，其次才是 HBDK 对 Fix #007 新常量/量化参数的调度敏感性；
6. 当前进程仍在计算，尚无证据证明它已完全卡死。

### 受控重编建议

必须保留当前 BC，不需要重新执行模型加载、校准或导出。建议流程：

1. 继续观察当前 `jobs=4`，直到出现 `.hbo`/`.hbm` 或确认进程异常；
2. 如果决定重编，先停止当前 Fix #007 的完整进程树，避免两个 HBDK 编译互相抢资源；
3. 直接加载现有 `vision.visual_convert.bc`；
4. 执行与历史成功基线一致的 `jobs=16`、`core_num=4`、`max_l2m_size=25165824` 编译；
5. 输出到独立目录 `qwen2_5_vl_fix007_vision_slice0_jobs16/`，不覆盖 jobs=4 产物；
6. 用日志明确记录 HBO 开始时间、完成时间、退出码和 HBM SHA256。

由于历史 `jobs=16` 已在同一 `oellm_clean` 环境的 Qwen Vision 上成功，受控 `jobs=16` 重编是有依据的 A/B 实验；但不能保证 Fix #007 新量化常量一定不会触发此前观察到的同步等待。该实验应在只保留当前 BC、无其他 HBDK 任务的条件下进行。

## HBDK 编译参数说明与证据边界

当前 `hbdk4.compiler.compile()` 文档确认：

- `opt=2`：编译优化等级，默认值为 2。它控制编译器进行的图优化/调度优化强度；不是 CPU 线程数，也不改变模型语义本身。提高优化等级通常可能增加编译时间，具体等级含义需以当前 HBDK 版本实现为准。
- `jobs`：编译优化阶段启动的线程/并行任务数。它与 BPU `core_num` 不同。
- `input_no_padding=True`：声明模型输入使用 native shape，不在输入侧采用 padding 形状。需要与 runtime 实际传入的 tensor 布局和尺寸一致。
- `output_no_padding=True`：声明模型输出使用 native shape，不在输出侧采用 padding 形状。需要与后续 runtime 读取输出的 shape 一致。
- `max_l2m_size=25165824`：允许编译器最多使用约 24 MiB 的 L2 memory，以减少 DDR 访问；不是模型输入输出 padding 参数。

`enable_hpc=True` 不出现在当前公开的 `hbdk4.compiler.compile()` Python 签名和 docstring 中，当前只能确认它曾作为 OELLM/SDK 封装传给底层编译器，不能在没有底层实现说明的情况下断言其精确含义。后续应把它标记为 SDK 扩展参数，并通过同一 BC 的 A/B 编译实测其影响。

重要更正：历史 Qwen Vision 日志的 Vision 编译行明确包含 `opt=2`、`input_no_padding=True`、`output_no_padding=True`、`jobs=16`、`march=nash-p`，但该行没有显示 `enable_hpc=True`；不能把 Language 阶段日志中的 `enable_hpc` 自动推断为 Vision 阶段也使用了它。当前 Fix #007 脚本显式传入 `march`、`core_num`、`max_l2m_size`、`jobs`，其余参数是否由 `q.model.compile_hbo()` 封装补齐，需要从实际封装源码或运行时打印的 kwargs 确认。

## Fix #007 `jobs=16` 受控编译结果

### 实验目的

在保持 Fix #007 `vision.visual_convert.bc`、`march=nash-p`、`core_num=4` 和 L2M 配置不变的情况下，将 HBO 编译并行度从 `jobs=4` 改为历史 Qwen Vision 成功使用过的 `jobs=16`，验证当前长耗时是否主要由并行度造成。

### 实验配置

```text
输入 BC：/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix007_vision_slice0/vision.visual_convert.bc
march=nash-p
opt=2
jobs=16
core_num=4
input_no_padding=True
output_no_padding=True
max_l2m_size=25165824
```

输出目录：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix007_vision_slice0_jobs16/
```

### 结果

日志：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix007_vision_slice0_jobs16/compile.jobs16.log
```

日志显示：

```text
[==================================================]100%
[3] link HBM
[DONE] ...jobs16.hbm 734526160
```

HBM：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix007_vision_slice0_jobs16/Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.jobs16.hbm
```

编译器记录的 HBO 编译到链接耗时：

```text
04h:38m:41s
```

HBM：

```text
size   = 734,526,160 bytes
SHA256 = f356f534c5148c4560a0aa2d2dd0858b5279d57f7adeae5363b452522fa8f1e6
march  = NashP
model  = visual
```

进程 `PID 3886496` 已正常退出，没有残留 HBDK/QEMU 编译进程。

### 与 `jobs=4` 的结论

- `jobs=4` 的 Fix #007 在 HBO 阶段长时间没有产出 HBM，随后已被停止；
- 相同 Fix #007 BC 使用 `jobs=16` 成功生成 HBM，耗时约 4 小时 39 分钟；
- 因此 `jobs=16` 是当前 Qwen2.5-VL Fix #007 更可靠的编译配置；
- 这次实验不能证明 Fix #007 的模型语义正确，只证明该 BC 可以在 NashP 上完成 HBO/HBM 编译；
- HBM 体积为约 734.5 MB，与旧历史/旧 Fix #006 HBM 体积不同，必须在 S600 做 descriptor、同输入输出统计和 cosine 验证。

### 下一步

1. 将 jobs16 HBM 上传到 S600 独立目录；
2. 先做板端 SHA256 校验；
3. 用 `/tmp/qwen_image0_1x1024x588_fp16.bin` 执行 Vision 输出；
4. 与官方 Vision、旧自编译、Fix #006 jobs4 HBM 做同输入比较；
5. 记录 mean、std、min、max、cosine、max diff；
6. 固定 `official_lang.hbm + official_embed.bin` 做端到端图片测试。

## Fix #007 jobs16 HBM 上传与 S600 初测

### 上传校验

Fix #007 jobs16 HBM 已通过 Windows 中转上传至 S600 独立目录：

```text
/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/fix007_jobs16/Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.jobs16.hbm
```

三端文件大小和 SHA256 一致：

```text
size   = 734,526,160 bytes
SHA256 = f356f534c5148c4560a0aa2d2dd0858b5279d57f7adeae5363b452522fa8f1e6
```

三端分别为 4090、Windows 中转文件和 S600。

### 测试配置

固定：

```text
Language = /home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/w4/official_lang.hbm
Embed    = official_embed.bin
Image    = /home/sunrise/oe_locateanything/oellm_runtime/examples/vlm_demo/image0.jpg
```

只替换 Vision HBM，使用单轮输入并显式发送 `exit`，避免管道 EOF 导致 demo 重复生成。

### 单轮端到端结果

官方全套：

```text
official Vision + official Language + official Embed
```

结果：正确识别为白马、骑手、障碍杆和户外马术场景。

Fix #006：

```text
Fix #006 Vision + official Language + official Embed
```

结果：可以加载和运行，但视觉语义错误，输出将图片描述为包含多个物体/拼贴等内容。

Fix #007 jobs16：

```text
Fix #007 jobs16 Vision + official Language + official Embed
```

结果：可以加载和运行，但只输出：

```text
The image showssegmentsthat
```

属于异常/碎片化输出，没有恢复到官方 Vision 的正确识别效果。

### 当前解释边界

这次结果确认：

1. Fix #007 jobs16 HBM 文件完整，S600 能加载；
2. NashP runtime、HBM descriptor 和文件传输没有问题；
3. Fix #007 与 Fix #006 的端到端文本行为不同，说明 Fix #007 的修改确实影响了 Vision→Language 路径；
4. Fix #007 尚未解决视觉语义对齐问题。

但端到端文本不能单独告诉我们误差具体发生在 Vision patch embedding、Vision block、merger/projector、输入预处理还是 Language 注入。因此下一步仍需使用同一 `[1,1024,588]` 输入导出四个 Vision HBM 的中间输出统计和 cosine，优先比较：

```text
official Vision
旧自编译 Vision
Fix #006 Vision
Fix #007 jobs16 Vision
```

### 后续实验

- 不修改官方 Language 和 Embed；
- 先完成 Vision HBM 的直接输出对比；
- 重点记录 mean/std/min/max、cosine、max diff；
- 若 Fix #007 与旧自编译明显不同但仍与官方不接近，继续检查 Conv3D temporal 处理和输入预处理，而不是继续调整 HBO jobs。

## Fix #007 jobs16 纯 Vision 板端输出对比

### 测试条件

四个 Vision HBM 使用同一输入：

```text
/tmp/qwen_image0_1x1024x588_fp16.bin
shape = [1, 1024, 588]
dtype = float16
```

使用 S600 板端 `hbrt4-run-model-nash` 直接执行 `visual` 图，不经过 Language、Embed 或 VLM demo。

### 输出统计

| 版本 | mean | std | min | max | L2 norm |
|---|---:|---:|---:|---:|---:|
| official Vision | 0.0102447 | 1.3646019 | -10.2031 | 12.3438 | 988.1082 |
| old self-compiled | -0.0086953 | 1.3611103 | -26.1563 | 28.8281 | 985.5712 |
| Fix #006 jobs4 | -0.0086953 | 1.3611103 | -26.1563 | 28.8281 | 985.5712 |
| Fix #007 jobs16 | -0.0095474 | 1.2501702 | -27.0000 | 25.0156 | 905.2469 |

### Cosine / max diff

相同输入下：

```text
official vs old_self       cosine = 0.00730277  max_diff = 31.328125
official vs Fix006         cosine = 0.00730277  max_diff = 31.328125
official vs Fix007_jobs16  cosine = 0.01036743  max_diff = 28.1904297
old_self vs Fix006         cosine = 1.0000001   max_diff = 0
old_self vs Fix007_jobs16  cosine = 0.62120956  max_diff = 22.5273438
```

### 结论

1. Fix #007 jobs16 HBM 的纯 Vision 输出与 Fix #006/旧自编译输出明显不同，证明 Fix #007 的 temporal/forward 修改确实影响了最终 HBM；
2. Fix #007 与官方 Vision 的 cosine 从旧版本约 `0.00730` 提升到约 `0.01037`，有改善但仍非常低，不能认为已经对齐；
3. Fix #007 输出 std 和 L2 norm 下降，输出范围仍与官方明显不同；
4. jobs16 只影响 HBO 编译并行度，不是这次输出差异的来源；jobs4/jobs16 使用同一 Fix #007 BC，理论上应产生相同语义图；
5. 当前主要问题仍在 Vision 模型/输入语义与官方 HBM 不一致，而不是 HBM 文件损坏、S600 runtime 或传输问题。

### 下一步定位方向

当前不宜继续围绕 `jobs` 调参。应优先验证：

- `/tmp/qwen_image0_1x1024x588_fp16.bin` 是否真的是官方 HBM 所需的输入语义；
- 当前自定义 `remove_repeat()` 是否错误丢弃 temporal slice 1；
- 对静态图片，是否应使用 `w0+w1` 或重复 temporal 输入，而不是只用 `w0`；
- Vision 前处理的 RGB/BGR、归一化、patch 顺序是否与官方 HBM 内部约定一致；
- 若输入一致，继续比较 Vision block / rotary / merger 的模型定义。

下一轮实验应固定同一 BC/编译参数，只生成明确的 temporal 变体，并继续用纯 Vision HBM 做板端 cosine 对比，避免 Language 端乱码干扰。

## 输入语义变体验证（官方 HBM / Fix #007）

### 执行参数修正

首次扫描出现两类核心数配置错误，不能作为模型结论：

- 官方 HBM pipeline 要求 4 核，必须使用 `-c 0,1,2,3`；
- Fix #007 jobs16 HBM pipeline 实际要求 1 核，必须使用 `-c 0`；
- 旧自编译扫描误用了 `w4/` 下的异常旧 HBM，加载时报 `Invalid ELF section header`，该轮不纳入结果。

修正核心数后，使用同一批输入变体完成官方/Fix #007 对比。

### 输入变体结果

| 输入变体 | 官方 vs Fix #007 cosine |
|---|---:|
| 基准 normalized RGB channel-last | 0.0103674 |
| raw RGB channel-last | 0.0119208 |
| normalized RGB channel-first | 0.0139555 |
| normalized RGB H-W | 0.0121477 |
| normalized BGR channel-first | 0.0067227 |
| normalized BGR channel-last | 0.0099513 |
| normalized BGR H-W | 0.0097945 |
| raw BGR channel-first | 0.0097305 |
| raw BGR channel-last | 0.0065438 |
| raw BGR H-W | 0.0072080 |
| raw RGB channel-first | 0.0136997 |
| raw RGB H-W | 0.0093048 |

官方 HBM 自身以基准输入为参考时，各变体 cosine 最高约 `0.806`（normalized RGB channel-last 与自身为 1.0），说明输入排列会影响官方输出，但没有发现一个简单变体能让 Fix #007 达到高相似度。

### 结论

1. 简单的 RGB/BGR、raw/normalized、channel-first/channel-last、H-W 转置不能解释当前官方 vs Fix #007 的 cosine≈0.01；
2. 当前问题不能优先归因于单纯输入布局；
3. 下一步应重点验证 temporal 处理：当前自定义 `remove_repeat()` 是否把 temporal=2 错误压成单 slice，以及 Conv2D 权重应使用 `w0`、`w0+w1`、`mean(w0,w1)` 哪一种；
4. 还需继续检查 Qwen Vision 的模型定义和官方 HBM 是否有不可见的权重/图预处理。

## Temporal 根因验证：HF 静态图片实际需要 `w0+w1`

### 实验方法

在 4090 上直接加载 HF 原始 Qwen2.5-VL-3B 权重和官方 Transformers Vision 模型，对同一张图片做 Patch Embedding 级别对照。处理器输出：

```text
pixel_values shape = [12696, 1176]
reshape            = [12696, 3, 2, 14, 14]
```

检查两个 temporal 位置：

```text
max(abs(slice0 - slice1)) = 0.0
mean(abs(slice0 - slice1)) = 0.0
```

也就是说，对静态图片，HF 预处理确实复制了两个完全相同的 temporal slice。

### 原始 Conv3D 与折叠权重对比

HF 权重：

```text
[1280, 3, 2, 14, 14]
```

完整 Conv3D 输出作为参考，比较只使用 temporal slice 0 的 Conv2D 等价计算：

| 权重处理 | 与 HF Conv3D Patch Embedding cosine | 输出 std |
|---|---:|---:|
| `w0 = w[:, :, 0, :, :]` | `0.7030` | `0.1483` |
| `w0+w1 = w.sum(dim=2)` | `1.0000` | `0.2019` |
| `mean(w0,w1)` | `1.0000`（方向相同，但幅值缩小） | `0.1010` |

`w0+w1` 与完整 Conv3D 的 max diff 只有 float16 数值误差；`mean` 与完整输出方向相同但幅度约减半；`w0` 明显不是原始 HF 静态图片语义。

### 对 Fix #006/#007 的影响

这次实验修正了之前对 `remove_repeat()` 的判断：

- `remove_repeat()` 是项目自定义的输入适配逻辑；
- 但 HF 原始静态图片输入本身包含两个相同 temporal slice；
- 只保留 slice 0 会丢掉原始 Conv3D 的第二项贡献；
- Fix #006/#007 的 `w0` 方向虽然与当前自定义 Conv2D 输入 shape 一致，但不等价于 HF 原始 Vision；
- 当前 Fix #007 与官方 Vision cosine 仍约 `0.0104`，与这一 temporal 语义错误相符。

### 下一版建议

下一版不应继续使用：

```python
w5d[:, :, 0, :, :]
```

而应优先验证：

```python
w4d = w5d.sum(dim=2)
```

同时保留当前输入 `[1,1024,588]`，因为对重复静态 temporal 输入，Conv3D 的数学结果可由单帧 Conv2D 加 `w0+w1` 精确表达。需要重新校准、生成 BC/HBM，并在 S600 用同一输入比较官方 cosine。

`mean(w0,w1)` 只作为幅度敏感性实验，不应作为第一候选，因为它会把 HF Patch Embedding 输出整体缩小约 2 倍。

### 重要边界

该实验只证明 temporal patch embedding 层的数学关系，不证明后续自定义 Vision block、rotary、window attention、merger 和量化流程已经与 HF 对齐。若改回 `w0+w1` 后端到端 cosine 仍低，需继续排查后续模型定义和量化校准。

## Fix #008：恢复 HF 静态图片的 temporal sum

### 动机

HF 原始 Qwen2.5-VL 对静态图片会生成两个完全相同的 temporal slice。完整 Conv3D 的 Patch Embedding：

```text
x*w0 + x*w1 = x*(w0+w1)
```

纯 PyTorch 实验证明：

```text
w0 与 HF Conv3D cosine = 0.7030
w0+w1 与 HF Conv3D cosine = 1.0000
mean(w0,w1) 与 HF Conv3D cosine = 1.0000，但幅度减半
```

因此 Fix #007 的 `slice 0` 不是 HF 静态图片语义的正确折叠方式。

### 修改

Fix #008 将模型加载阶段的：

```python
w4d = w5d[:, :, 0, :, :]
```

恢复为：

```python
w4d = w5d.sum(dim=2)
```

同时保留 Fix #007 删除 `forward()` 中 temporal sum 覆盖的修复，避免校准时重复覆盖权重。

同步修改：

```text
/home/kangjie.xu/oe_locateanything/toolchain/leap_llm/models/qwen2_5_vl/model.py
/home/kangjie.xu/miniforge3/envs/oellm_clean/lib/python3.10/site-packages/leap_llm/models/qwen2_5_vl/model.py
```

### 当前编译状态

后台脚本：

```text
/tmp/compile_fix008_vision_jobs16.py
```

输出目录：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix008_vision_temporal_sum/
```

参数：

```text
march=nash-p
jobs=16
core_num=4
max_l2m_size=25165824
```

已完成：

```text
模型加载
120 张 mmstar 校准
export BC
convert MLIR
```

Fix #008 转换图：

```text
b30vpu.call = 65
b30.quantize = 259
b30.conv2d = 227
hbdk.constant = 946
vision.visual_convert.bc = 671,024,768 bytes
```

当前已进入：

```text
[6] compile HBO
```

尚未生成 `.hbo` / `.hbm`。

## 官方 Vision HBM 与 HF Vision 的严格复测（2026-07-17）

为确认 Fix #008 的 PyTorch 输出与 HF cosine `0.5736` 是否真的代表模型语义偏差，使用完全相同的输入重新比较：

```text
S600/HF 输入：/tmp/qwen_image0_1x1024x588_fp16.bin
官方输出：/tmp/vision_cmp_official/_output_0.npy
输入 shape：[1,1024,588] float16
输出 shape：[256,2048] float32（由板端 float16 输出加载后转 float32）
HF grid_thw：[1,32,32]
HF temporal 输入：将同一静态 patch 复制为两个相同 temporal slice
```

首先确认 Fix #008 的 `0.5736` 主要来自输出 token 顺序：当前自定义 Vision 在 merger 后没有执行 HF 的：

```python
reverse_indices = torch.argsort(window_index)
hidden_states = hidden_states[reverse_indices, :]
```

结果：

```text
Fix #008 PyTorch raw vs HF              cosine = 0.573591973
Fix #008 PyTorch reverse_indices vs HF  cosine = 0.999872031
max_diff = 1.2655425
rmse = 0.0224403
```

因此 Fix #008 的 temporal sum 和主体 Vision 计算已经基本复现 HF；`0.5736` 不能再解释为主体模型只对齐了一半，主要是 merger 后缺少恢复原 token 顺序。

随后对 S600 官方 Vision HBM 输出做同样的原始/重排比较：

```text
official raw vs HF                      cosine = 0.008025857
official[window_index] vs HF            cosine = 0.008020009
official[argsort(window_index)] vs HF   cosine = 0.008020009
official raw vs HF[window_index]        cosine = 0.008020009
official raw vs HF[argsort]             cosine = 0.008020009
```

统计：

```text
HF       min=-28.2764 max=29.3054 mean=-0.00935 std=1.40175
official min=-10.2031 max=12.3438 mean= 0.01024 std=1.36460
```

结论：

1. 官方 Vision HBM 与 HF 原始 Vision 的低 cosine 已严格复现，且不是 `window_index` 输出顺序导致；
2. 官方全套能够正常识图，说明官方 Vision 与官方 Language 之间存在可工作的配套特征接口；
3. 目前不能把官方 Vision HBM 的裸输出当作 HF Vision feature 的直接量化近似。可能原因包括官方使用了不同 checkpoint、未公开的输出变换/量化域或与官方 Language 配套的内部 ABI；
4. 当时暂定以 HF 语义为参考修复 merger 后的 reverse-index；该判断随后被官方真实输入和 QuaRot/Hadamard 证据推翻，最终方向见下一节“方向修正：最终必须对齐官方隐藏域”。

## 方向修正：最终必须对齐官方隐藏域（2026-07-17）

上一节第 4 点只适用于排查自定义 Python 模型内部语义，不适合作为部署验收目标。Qwen2.5-VL 阶段的实际验收目标应为：

1. 自编译 Vision 能接入官方 Language + 官方 embed；
2. 同一图片、同一 prompt 下，语义输出接近官方全套；
3. 自编译 Vision 的裸输出应进入官方 Language 期望的 2048 维隐藏域；
4. HF cosine 只用于定位模型层级错误，不能替代官方运行时 ABI 验收。

### 抓取官方运行时真实 Vision 输入

此前纯 Vision HBM 比较使用的是手工生成文件：

```text
/tmp/qwen_image0_1x1024x588_fp16.bin
```

但该文件没有被证明与官方 `libxlm.so` 实际送入 `visual` graph 的输入一致。为消除猜测，新增只读 `LD_PRELOAD` hook，拦截：

```text
hbDNNInferV2@CONFIG
hbDNNInferV3@CONFIG
```

hook 只匹配：

```text
shape=[1,1024,588]
dtype=float16
stride=[1204224,1176,2]
```

官方全套运行仍正常输出：

```text
In the image, I see a person riding a white horse ...
```

捕获文件：

```text
/tmp/official_visual_input_000.bin
/tmp/official_visual_input_001.bin
```

两份输入 SHA256 完全一致：

```text
a406e17ea359cac16937b64d744aaf4a59b538b6f14887e397bd3c4c9fe72a29
```

真实输入与旧手工输入比较：

```text
official runtime input vs old manual input cosine = 0.22047679
max_diff = 3.8945312
mean_abs_diff = 0.8645985
```

因此此前用旧手工输入得到的 `official vs HF≈0.008` 和 `official vs self≈0.007` 不能作为官方 ABI 的最终结论。旧输入不是 `libxlm` 的真实输入。

### 使用真实输入重建官方基线

固定输入：

```text
/tmp/official_visual_input_000.bin
```

板端输出统计：

| Vision HBM | std | L2 |
|---|---:|---:|
| official | 2.40466 | 1741.18 |
| old self / temporal sum | 2.41118 | 1746.07 |
| Fix #007 / temporal slice 0 | 1.54590 | 1119.42 |

输出比较：

```text
official vs old self  cosine = 0.0133040
official vs Fix #007 cosine = 0.0091986
old self vs Fix #007 cosine = 0.7151876
```

官方和 old self 的幅度、范数非常接近，但逐元素 cosine 接近 0；这提示两者可能处于不同的隐藏空间基，而不是模型主体完全无关。

## 根因：官方 QuaRot / Hadamard 隐藏空间旋转

### Embed 的直接证据

比较：

```text
official_embed.bin
self Qwen2.5-VL-3B-Instruct_embed_tokens.bin
```

二者 shape 均为：

```text
[151936,2048] float16
```

随机抽取 1024 个 token：

```text
same-token cosine median = -0.00732
official/self token norm ratio median = 0.999887
1024-token cosine Gram matrix correlation = 0.9999911
Gram matrix mean absolute difference = 0.0002679
```

这是一组非常强的正交变换证据：官方 embed 与 HF/self embed 的坐标不同，但 token 间内积和范数几乎完全保持。

使用 8192 个 token 做 Orthogonal Procrustes，恢复 `2048x2048` 旋转矩阵 `R`：

```text
未参与拟合的 4096 个 token：
self embed raw vs official      cosine = -0.006927
self embed @ R vs official      cosine = 0.999817
rmse = 0.0004316
```

### Vision 的交叉验证

把同一个 `R` 应用于 old self Vision 输出：

```text
old self raw vs official        cosine = 0.013304
old self @ R vs official        cosine = 0.994467
rmse = 0.253469
```

这证明官方 Vision、官方 embed 和官方 Language 使用同一个旋转后的 2048 维隐藏域。此前所有“官方/自编译混搭失败”都与该隐藏域不匹配一致。

Fix #007 的 temporal slice 0 经相同旋转后只有：

```text
Fix #007 @ R vs official cosine = 0.712678
```

因此 temporal sum 是向官方对齐的正确分支，slice 0 不是。

### 精确 Hadamard 结构

恢复矩阵几乎所有元素绝对值都接近：

```text
1 / sqrt(2048) = 0.0220971
```

将其符号矩阵与标准 2048 阶 Walsh-Hadamard 矩阵匹配后发现：

```text
2048 行均能一一映射到唯一 Hadamard 行
绝大多数行相关得分 = 2048 / 2048
只有极少数元素受 SVD/float16 误差影响
```

由行置换和行符号构造精确矩阵 `Q`，满足：

```text
max(abs(Q.T @ Q - I)) = 5.96e-08
```

验证：

```text
self embed @ Q vs official      cosine = 0.999801
old self Vision @ Q vs official cosine = 0.996371
Fix #007 Vision @ Q vs official cosine = 0.713940
```

该变换与 QuaRot/随机 Hadamard 旋转特征一致。项目代码中 `eagle3/model.py` 也明确出现了 `quarot-transformed` embed 的说明，但当前 Qwen2.5-VL 自编译 API 没有应用这一步。

## Fix #009：对齐官方隐藏域

### 修改设计

Fix #009 使用：

```text
temporal patch weight = w5d.sum(dim=2)
保留 window 顺序，不添加 HF 最后的 reverse_indices
将精确官方 Hadamard 旋转 Q 折叠进 Vision merger 最后一层
```

merger 最后一层原计算：

```text
y = x @ W.T + b
```

目标：

```text
y_official_domain = y @ Q
```

等价折叠：

```python
W_new = Q.T @ W
b_new = b @ Q
```

因此不新增运行时 `2048x2048` 算子，不改变 graph 输入/输出 shape，也不增加额外 BPU 延迟。

纯 PyTorch 验证：

```text
raw PyTorch vs official HBM       cosine = 0.012966
post-rotated vs official HBM      cosine = 0.984662
folded-weight vs official HBM     cosine = 0.984662
folded vs post-rotated            cosine = 1.000000
max_diff = 2.19e-05
```

### Fix #008 处置

Fix #008 没有官方隐藏旋转，继续编译不能满足官方 Language ABI。其 HBO 编译在运行约 1.5 小时后停止；以下产物保留作为历史对照：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix008_vision_temporal_sum/vision.visual.bc
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix008_vision_temporal_sum/vision.visual_convert.bc
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix008_vision_temporal_sum/compile.jobs16.log
```

### Fix #009 编译记录

编译期间后台 PID：

```text
912420
```

脚本：

```text
/tmp/compile_fix009_official_domain.py
```

输出目录：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix009_vision_official_domain/
```

日志：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix009_vision_official_domain/compile.jobs16.log
```

参数：

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

HBM 编译前已完成：

```text
模型加载
精确 Hadamard 旋转折叠
120 张 mmstar 校准
export BC
convert MLIR
```

随后进入：

```text
[6] compile HBO
```

该编译已于 2026-07-17 完成，最终结果见后文“Fix #009 最终板端验证”。

转换图算子统计与旧 temporal-sum 图相同：

```text
b30vpu.call = 65
b30.quantize = 259
b30.conv2d = 227
hbdk.constant = 946
```

持久化 RCA 产物：

```text
4090:
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix009_vision_official_domain/official_hidden_rotation_exact.pt
SHA256: 274a231024ee3ca507fcb52cc3c220b74f0bdfc94a2f868636127307125d0a13

/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix009_vision_official_domain/official_hadamard_row_permutation.npy
SHA256: 5f6937cc46ea1e02db4737e09f7f3ac8d35d8d801759c8ecd278b914907ed3c7

/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix009_vision_official_domain/official_hadamard_row_sign.npy
SHA256: 0e6ff4f0f953ed6da5406ecd2cc675bd878fbf45a3c86cfe52d10c343d0c76b4

S600:
/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/rca_inputs/image0_official_visual_input_fp16.bin
SHA256: a406e17ea359cac16937b64d744aaf4a59b538b6f14887e397bd3c4c9fe72a29
```

### Fix #009 验收标准

HBM 生成后必须按以下顺序验证：

1. 使用 hook 捕获的 `/tmp/official_visual_input_000.bin` 纯 Vision 跑板；
2. Fix #009 vs official Vision cosine 目标优先达到 `>0.98`；
3. 使用官方 Language + 官方 embed + Fix #009 Vision 运行 `image0.jpg`；
4. 输出应正确识别骑手、白马和障碍杆；
5. 再用至少一张不同图片验证，排除对单图拟合；
6. 若纯 Vision 高 cosine 但端到端失败，再检查 Vision token 注入顺序；不再回退到 HF reverse-index 路线。

## Fix #009 最终板端验证（2026-07-17）

### 编译结果

```text
compile_hbo: 9408.8229 秒（约 2 小时 36 分 49 秒）
link_models: 29.2811 秒
HBM size: 762,029,104 bytes
SHA256: d4511b8f910c25d8111056ce4cddf7652c91b59e05c3d95fdabc4dead0e94df8
```

HBM：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix009_vision_official_domain/Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm
```

S600：

```text
/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/fix009_official_domain/Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm
```

4090、本机中转、S600 三端 SHA256 一致。

Graph descriptor：

```text
march: nash-p
toolkit: 4.10.2a2.dev202603180400+4c23b55.develop
graph: visual
input:  [1,1024,588] float16
output: [1,256,2048] float16
```

### 纯 Vision 板端比较

固定使用官方 `libxlm` hook 捕获的真实输入：

```text
/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/rca_inputs/image0_official_visual_input_fp16.bin
```

统计：

```text
Fix #009: min=-21.4844 max=21.1094 mean=-0.01256 std=2.42887 L2=1758.72
official: min=-21.3594 max=21.2812 mean=-0.01183 std=2.40466 L2=1741.18
```

比较：

```text
Fix #009 vs official Vision cosine = 0.9879630
max_diff = 7.08887
rmse = 0.375698
```

达到预设 `>0.98` 的官方隐藏域对齐目标。

### 官方 Language + 官方 Embed 端到端

配置：

```text
/home/sunrise/oe_locateanything/oellm_runtime/examples/vlm_demo/test_fix009_official_lang.json
```

组合：

```text
Vision: Fix #009
Language: official_lang.hbm
Embed: official_embed.bin
```

`image0.jpg`：

```text
The image depicts a rider on a white horse jumping over a set of red and white
striped crossbars ...
```

与官方全套一样正确识别骑手、白马、跳杆和户外马术场景。

`image1.jpg`：

官方：

```text
In the image, there is a small red panda ... sitting on a wooden structure ...
```

Fix #009：

```text
In the image, there is a small red panda ... sitting on a wooden structure ...
```

两者均正确识别小熊猫、木架、红褐色毛发和白色面部标记。

### 最终结论

Qwen2.5-VL-3B Vision 自编译路线已经跑通：

```text
HF/static-image temporal sum
+ 官方 QuaRot/Walsh-Hadamard 隐藏空间旋转
+ 官方 libxlm 输入 ABI
+ 官方 4-core HBDK 编译参数
= 可替换官方 Vision HBM 的自编译 HBM
```

Fix #009 不仅纯 Vision 数值接近官方，而且已经通过两张不同图片的官方 Language 端到端语义验证。这是当前第一份同时满足“自编译产物”和“官方运行时兼容”的 Qwen2.5-VL-3B Vision HBM。

## Fix #010：Language 官方隐藏域对齐（2026-07-17）

### 目标与问题边界

Fix #009 已证明官方 Qwen2.5-VL Vision、embed 和 Language 使用同一个 2048 维 Walsh-Hadamard 正交隐藏域。旧自编译 Language 即使搭配官方 embed 仍输出乱码，因此 Language 不能只替换 embed；每层残差流、Attention 输出、MLP 输出和最终 lm_head 必须在同一个隐藏域内数学闭合。

本轮保持以下接口不变：

```text
Q/K/V 内部坐标不变
RoPE 坐标不变
KV cache 坐标和运行时 ABI 不变
prefill/decode 输入输出 shape 不变
chunk_size=256
cache_len=1024
march=nash-p
```

### Language 变换设计

设官方隐藏域旋转为正交矩阵 `Q`，原残差向量为 `x`，官方域残差为 `xQ`。

Embedding：

```text
E_new = E @ Q
```

每层 Attention 的 RMSNorm 权重折叠进 Q/K/V 输入投影：

```text
Wq_new = (Wq * gamma_attn) @ Q
Wk_new = (Wk * gamma_attn) @ Q
Wv_new = (Wv * gamma_attn) @ Q
gamma_attn_new = 1
```

Attention 内部 Q/K/V、RoPE 和 KV cache 因而继续保持原坐标；Attention 输出投影负责回到官方残差域：

```text
Wo_new = Q.T @ Wo
```

MLP 同理：

```text
Wgate_new = (Wgate * gamma_mlp) @ Q
Wup_new   = (Wup   * gamma_mlp) @ Q
gamma_mlp_new = 1
Wdown_new = Q.T @ Wdown
```

最终 RMSNorm 权重折叠进 lm_head：

```text
Wlm_new = (Wlm * gamma_final) @ Q
gamma_final_new = 1
```

该设计不是在运行时额外插入 2048x2048 MatMul，而是将旋转离线折叠进现有权重，因此不改变 Language 图的接口，也不要求运行时旋转 KV cache。

### PyTorch 数学等价性验证

验证脚本：

```text
tools/qwen25_language_official_rotation.py
tools/validate_fix010_language_rotation.py
```

在同一 token 输入上比较旋转前后 Language 输出和每层 KV：

```text
orthogonal_max_error = 5.960464477539063e-08
input_cosine = 0.9999999999996428
input_max_diff = 1.378e-07
logits_cosine = 0.999999999888713
logits_max_diff = 0.000348568
logits_rmse = 6.4989e-05
kv_key_cosine_min = 0.9999999998191405
kv_value_cosine_min = 0.9999999994474578
argmax_equal = True
```

这证明 Fix #010 的离线权重变换在 PyTorch 层保持原模型 logits 和 KV 语义，且输入/残差流进入官方隐藏域。

### 编译输入、参数与产物

编译脚本：

```text
tools/compile_fix010_language_official_domain.py
```

输出目录：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix010_language_official_domain/
```

参数：

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

校准仍使用 SDK `mmstar/conversation.json` 的 120 条样本。为保持 Vision token 与旋转后的 Language 输入域一致，校准路径中的 Vision merger 同样临时折叠 `Q`；这只影响校准和 Language BC 导出，不生成新的 Vision HBM。

已生成的中间产物：

```text
language.prefill.bc
language.prefill_convert.bc
language.decode.bc
language.decode_convert.bc
Qwen2.5-VL-3B-Instruct_embed_tokens.bin
```

Fix #010 embed 与官方 embed 的抽样比较：

```text
cosine = 0.999718904
max_diff = 0.006896973
rmse = 0.000446225
```

这与此前裸 HF/self embed 的完全不同隐藏域形成明确对照，也验证了恢复的 `Q` 与官方 embed 预处理一致。

### 编译结果与后续验收顺序

2026-07-17 通过 `nohup` 启动后台编译，PID 为 `3319216`，最终正常退出。错误扫描未发现 `Traceback`、LLVM 崩溃或 `FAILED`。`Illegal b30 fusion operator detected` 是编译器对部分 GQA 融合的优化回退提示，本次仍成功生成 HBO 和 HBM，不能将该提示单独视为失败。

精确耗时：

```text
prefill compile_hbo = 3574.3709 秒（约 59.6 分钟）
decode compile_hbo  = 4056.2590 秒（约 67.6 分钟）
```

最终产物：

```text
language.prefill.hbo = 1,812,376,432 bytes
language.decode.hbo  = 1,788,275,968 bytes
Language HBM         = 1,825,571,064 bytes
embed                =   622,329,856 bytes
```

4090 路径：

```text
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix010_language_official_domain/Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm
/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix010_language_official_domain/Qwen2.5-VL-3B-Instruct_embed_tokens.bin
```

SHA256：

```text
Language HBM = 05961201af02c22894f48a4c5d90f859878c473892f8c9e3c0012bbe7f7aabd0
embed        = f9efbe1d4905a581993e255de0e815f304c24a0bf00f5bee8f89f5e47e464caf
```

HBM descriptor：

```text
march: nash-p
toolkit: 4.10.2a2.dev202603180400+4c23b55.develop
graphs: prefill, decode

prefill input_0:  [1,256,2048] float16
prefill output_0: [1,256,151936] float16
decode input_0:   [1,1,2048] float16
decode output_0:  [1,1,151936] float16
KV cache input/output dtype: int8
```

Windows 中转的 `.fresh` HBM 和 embed 已分别通过 SHA256，与 4090 一致。

第一次向 S600 上传时，普通 `scp` 达到本地工具 30 分钟超时，随后使用 `sftp reput` 续传。续传启动瞬间旧 ssh 连接仍有少量缓冲写入，导致原目标 HBM 比源文件多 `3,837,952` 字节：

```text
无效 HBM size   = 1,829,409,016 bytes
无效 HBM SHA256 = 8b3b428dbca4d218c09c875840fca945d6de821bb0a6c0be9a45b6c8e4593b34
```

该文件明确判定无效，不用于测试。为避免续传状态污染，随后将 Windows 上已校验的 HBM gzip 上传到全新文件名，在 S600 解压为 `.hbm.fresh`。压缩包和解压后 HBM 均通过 SHA256：

```text
S600 valid HBM:
/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/fix010_language_official_domain/Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm.fresh
size   = 1,825,571,064 bytes
SHA256 = 05961201af02c22894f48a4c5d90f859878c473892f8c9e3c0012bbe7f7aabd0

S600 valid embed:
/home/sunrise/oe_locateanything/oellm_runtime/model/Qwen2.5-VL-3B-Instruct/fix010_language_official_domain/Qwen2.5-VL-3B-Instruct_embed_tokens.bin
size   = 622,329,856 bytes
SHA256 = f9efbe1d4905a581993e255de0e815f304c24a0bf00f5bee8f89f5e47e464caf
```

测试配置：

```text
/home/sunrise/oe_locateanything/oellm_runtime/examples/vlm_demo/test_fix010_full_self.json
```

该配置固定 `Vision=Fix #009`，并明确指向有效的 `.hbm.fresh` 和 Fix #010 embed。JSON、三个路径和文件大小均已检查。板端语义验证仍待人工运行，因此当前只能确认“Fix #010 编译、传输和配置成功”，不能提前宣称全自编译 Qwen2.5-VL 已跑通。

后续严格按以下顺序验收：

1. 完成 S600 上传并校验 HBM/embed SHA256；
2. 固定 `Vision=Fix #009`，只替换 `Language=Fix #010`、`Embed=Fix #010`；
3. 先做纯文本短回答测试，再测试 `image0.jpg` 和 `image1.jpg`；
4. 若仍乱码，优先抓取官方与 Fix #010 的 prefill 输入、输出和 logits，保持 Vision 不变继续单变量 RCA。

### Fix #010 S600 板端验证（2026-07-18）

实际加载组合：

```text
Vision: Fix #009 自编译 HBM
Language: Fix #010 自编译 HBM（.hbm.fresh）
Embed: Fix #010 自生成 embed
Runtime: 官方 SDK vlm/libxlm + HBRT 4.9.6
```

启动日志确认 `decode`、`prefill` 和 `visual` 三张图均从上述自编译产物成功加载，`VLM Demo XLM Engine is ready`。

纯文本输入：

```text
[User] hi？
[Assistant] Hello! How can I assist you today?

prefill token num: 256
prefill cost: 46.254 ms
prefill speed: 5534.656462 tokens/s
decode token num: 9
decode cost per token: 14.256667 ms
decode speed: 70.142623 tokens/s
```

该结果证明 Fix #010 Language HBM 与 Fix #010 embed 的纯文本链路已恢复正常，不再出现旧自编译 Language 的乱码输出。

`image1.jpg` 图文输入：

```text
[User] 描述一下图片
[Assistant] The image shows a small red panda sitting on a wooden platform.
The red panda has a distinctive reddish-brown fur coat with white markings
around its eyes, ears, and chest area. Its face is expressive and
friendly-looking. The small red panda appears to be in a relaxed posture,
possibly on the edge of the wooden platform.

vit cost: 42.733 ms
prefill token num: 512
prefill cost: 86.972 ms
prefill speed: 5886.952123 tokens/s
decode token num: 67
decode cost per token: 14.208612 ms
decode speed: 70.379852 tokens/s
```

输出正确识别小熊猫、木质平台、红褐色毛发和白色面部标记。由此确认：

1. Fix #009 Vision 的官方隐藏域输出能够被 Fix #010 Language 正确消费；
2. 自编译 Vision + 自编译 Language + 自生成 embed 的完整图文主链路已经跑通；
3. 运行时仍使用官方 SDK，但本次没有加载官方 Vision HBM、官方 Language HBM 或官方 embed；
4. `image0.jpg` 仍作为第二张不同图像的最终回归测试，用于排除单图偶然性。
