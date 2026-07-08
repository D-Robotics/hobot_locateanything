# main

Unified S600 deployment workspace for LocateAnything on OELLM.

## Subdirectories

- `vision/` - MoonViT + MLP custom OELLM visual module.
- `language/` - LocateAnything Qwen2.5 prefill/decode/PBD modules.
- `runtime/` - host runtime, tokenizer, visual embedding insertion, KV-cache and PBD/hybrid sampling.
- `configs/` - compile/runtime configs.
- `scripts/` - setup, build, validation and benchmark scripts.
- `golden/` - calibration and golden outputs.
- `benchmarks/` - benchmark inputs and results.
- `outputs/` - generated HBM/bin artifacts.
- `logs/` - build and validation logs.
