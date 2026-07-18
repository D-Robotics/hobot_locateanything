<div align="center">

<img src="assets/LocateAnything.jpg" alt="LocateAnything on S600" width="820">

# LocateAnything-3B 部署到 D-Robotics S600

在 S600 BPU 上部署 LocateAnything-3B，保留原生 MoonViT、152,681 词表和
6-token Parallel Block Decoding（PBD）。

[English](README.md) · **中文**

</div>

## 当前状态

- Qwen2.5-VL-3B 自编译基线已经在 S600 上通过文本和图片语义测试。
- LocateAnything Fix #011 已通过 checkpoint 完整加载、隐藏域数学等价性和
  Language/Vision BC 导出。
- Fresh Fix #011 HBM 正在编译或等待板端验证；历史 LA HBM 仅作为 RCA 对照，
  不能当作发布模型。
- 自研 Host runtime 已具备 HBM session、embedding lookup、attention mask 和
  position IDs 等基础模块；完整 PBD/Hybrid 采样与 grounding 闭环仍在推进。

## 模型约束

| 组件 | 配置 |
|---|---|
| Vision | MoonViT，27 层，hidden 1152，patch 14 |
| Projector | 2x2 merge，4608 -> 2048 -> 2048 |
| Language | Qwen2.5/Qwen2，36 层，hidden 2048，2 KV heads |
| Vocabulary | 152,681，输入/输出 embedding tied |
| PBD | q_len=6，text-mask token 151676 |
| 固定图像 profile | 448x448，1024 patches -> 256 visual tokens |
| Language profile | prefill 1024，cache 2048，PBD decode 6，AR decode 1 |

## 快速开始

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
git clone https://github.com/NVlabs/Eagle.git eagle

cd toolchain
pip install -e . --no-deps
cd ..

# 先导出 BC，确认不会进入数小时 HBO 编译后才暴露图错误。
EXPORT_ONLY=1 ./main/scripts/compile_locateanything_language.sh

# 后台编译，使用 setsid + nohup，SSH 断开后继续运行。
./main/scripts/compile_locateanything_language.sh
./main/scripts/compile_locateanything_vit.sh
```

查看日志：

```bash
tail -f main/logs/locateanything_language_compile.log
tail -f main/logs/locateanything_vit_compile.log
```

## 关键设计

1. 不使用 Qwen2.5-VL 的 ViT 替代 MoonViT。
2. Language、embedding table 和 MoonViT projector 使用同一 2048 维 signed
   Walsh-Hadamard 隐藏域，变换离线折叠进权重，不增加运行时 MatMul。
3. `DynamicQuantLinear` lm_head 保留 Leap `build()` 能力；普通 `nn.Linear`
   无法消费 HBDK `OpResult`。
4. 固定 448x448 Vision HBM 输出 256 个 visual tokens，Host prompt 必须插入
   完全相同数量的 image token，不能混用上游动态分辨率的 925-token 示例。

## 文档

- [原始 LocateAnything 源码审计](docs/SOURCE_REVIEW.md)
- [LocateAnything 编译教程](docs/tutorials/LOCATEANYTHING_COMPILATION.md)
- [Qwen2.5-VL 基线说明](docs/tutorials/QWEN2_5_VL_BASELINE.md)
- [S600 运行时与同步](docs/tutorials/S600_RUNTIME.md)
- [已知问题](docs/KNOWN_ISSUES.md)
- [完整 SDK Compiler RCA](docs/rca/sdk_compiler_rca_review.md)

## 目录

```text
oe_locateanything/
├── baselines/qwen2_5_vl/       Qwen 编译基线、配置和实验快照
├── docs/                       源码审计、教程、RCA、已知问题
├── main/                       LA 编译脚本、runtime、输出和日志目录
├── toolchain/leap_llm/         OELLM 源码与 LA 独立适配
├── eagle/                      NVIDIA Eagle/LocateAnything（不入 Git）
└── oellm/                      S600 SDK（不入 Git）
```

## 说明

论坛案例 <https://forum.d-robotics.cc/t/topic/35332> 来自独立开发者，不是官方
从零编译教程。本项目借鉴其工程方法，但所有 Qwen/LA 结论均以本仓库的代码、
产物 checksum 和板端实验为准。

本项目使用 [CC BY-NC 4.0](LICENSE)。上游模型、SDK 和 vendored 代码继续遵循各自
许可证。
