# Documentation

## Start Here

| Document | Scope |
|---|---|
| [Compiler porting guide](COMPILER_PORTING_GUIDE.zh-CN.md) | Complete path from environment setup and Qwen2.5-VL chain validation to LocateAnything BC/HBM compilation |
| [Source review](SOURCE_REVIEW.md) | Authoritative checkpoint contract, upstream source paths, and PBD generation semantics |
| [Runtime architecture](RUNTIME_ARCHITECTURE.md) | Host/BPU responsibilities, runtime modules, and graph invocation |
| [Calibration strategy](CALIBRATION.md) | Task-specific data contract, scale collection, data isolation, and acceptance gates |
| [Known issues](KNOWN_ISSUES.md) | Reproducible symptoms, evidence, fixes, and prevention |

## Focused Guides

| Document | Scope |
|---|---|
| [LocateAnything compilation quick reference](tutorials/LOCATEANYTHING_COMPILATION.md) | Commands and graph contracts for daily builds |
| [Qwen2.5-VL chain validation](tutorials/QWEN2_5_VL_BASELINE.md) | Reference-model compilation and S600 semantic verification |
| [S600 runtime and synchronization](tutorials/S600_RUNTIME.md) | Code sync, artifact transfer, runtime contract, and evidence levels |
| [Deployment workspace](../main/README.md) | Ownership of scripts, runtime, outputs, logs, and examples |

## Engineering Records

- `rca/`: detailed investigations and controlled comparison reports.
- `archive/`: superseded deployment notes retained for traceability.
- `BUILD_STATUS.md`: current long-running build metadata and monitoring commands.
