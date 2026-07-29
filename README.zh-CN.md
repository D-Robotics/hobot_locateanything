# OE LocateAnything

本仓库只包含两个相互独立的 S600 部署产品：

| 产品 | 内容 |
|---|---|
| [`LocateAnything/`](LocateAnything/) | LocateAnything-3B 量化、HBM 编译、四核 BPU Runtime、PBD 生成与六任务定位验证 |
| [`Qwen-2.5-VL-3B/`](Qwen-2.5-VL-3B/) | 用于验证 OELLM/HBDK/HBRT 编译运行链路的 Qwen2.5-VL-3B 基准实现 |

两个产品分别维护脚本、文档、配置和工作区，不依赖仓库目录名，也不再依赖旧 `main/`、`baselines/` 路径。

数据集、SDK 压缩包、中转文件和历史调查材料不进入项目。模型、校准张量与运行结果统一写入各产品中被 Git 忽略的 `workspace/`。
