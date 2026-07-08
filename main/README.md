# main

LocateAnything-3B 在 D-Robotics S600 上的部署工作目录。

## 子目录

| 目录 | 说明 |
|---|---|
| `examples/` | 基线与集成示例（PyTorch baseline、HBM 对齐脚本等） |
| `vision/` | MoonViT + MLP Vision Module，编译为 Vision HBM |
| `language/` | Qwen Prefill / PBD Decode / AR Decode Language Module |
| `runtime/` | Host 侧 runtime：tokenizer、visual embedding 插入、KV cache、PBD / Hybrid 采样 |
| `configs/` | 编译与运行时配置 |
| `scripts/` | 构建、验证、benchmark 脚本 |
| `golden/` | 校准数据与 golden 输出 |
| `benchmarks/` | benchmark 输入与结果 |
| `outputs/` | 编译产物（HBM、bin），不入 Git |
| `logs/` | 编译与验证日志，不入 Git |
