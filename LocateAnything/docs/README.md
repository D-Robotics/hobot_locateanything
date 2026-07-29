# LocateAnything Technical Documents

The active documentation is intentionally flat. Each file covers one durable
engineering boundary; historical task notes and investigation snapshots are
kept outside the product tree. The active compiler contract is 672x672, Vision
W8, Language/lm_head W8/W8, prefill 1024, cache 4096, base PBD/AR q=6/q=1,
and a 13-graph candidate family. The older three-graph board package is
historical evidence, not the current fused-release claim.

| Document | Scope |
|---|---|
| [Float model architecture](LOCATEANYTHING_FLOAT_MODEL.zh-CN.md) | MoonViT, Qwen2.5 decoder, spatial tokens, six grounding tasks, PBD and AR |
| [LLM/VLM quantization pipeline](LLM_VLM_QUANTIZATION_PIPELINE.zh-CN.md) | General Float, quantized simulation, BC, HBO, HBM, and device flow |
| [LLM/VLM quantization techniques](LLM_VLM_QUANTIZATION_TRICKS.zh-CN.md) | Code domain, scale selection, W8/A8, orthogonal rotation, centering, and diagnosis |
| [LocateAnything OELLM deployment](LOCATEANYTHING_OELLM_DEPLOYMENT.zh-CN.md) | Model-specific deployment, historical three-graph evidence, and the 13-graph release candidate |
| [Compiler porting guide](COMPILER_PORTING_GUIDE.zh-CN.md) | Environment, registration, calibration, BC/HBM compilation, and S600 transfer |
| [Calibration](CALIBRATION.md) | Six-domain data, tensor materialization, scale collection, and data isolation |
| [Runtime architecture](RUNTIME_ARCHITECTURE.md) | Host/BPU boundary, graph contracts, KV cache, and generation state machine |
| [S600 runtime](S600_RUNTIME.md) | Build, deployment, CLI installation, and board verification |
| [Benchmarking](BENCHMARKING.md) | Repeated q=1/q=6 runs and CPU, BPU, memory, bandwidth, and thermal evidence |
| [Known issues](KNOWN_ISSUES.md) | Technical failures, causes, evidence, and fixes |
| [Project layout](PROJECT_LAYOUT.md) | Source ownership and generated workspace contract |

Qwen2.5-VL is maintained as a separate product under
[`../../Qwen-2.5-VL-3B/`](../../Qwen-2.5-VL-3B/).
