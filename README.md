# OE LocateAnything

This repository contains two independent S600 deployment products:

| Product | Purpose |
|---|---|
| [`LocateAnything/`](LocateAnything/) | LocateAnything-3B quantization, HBM compilation, four-core BPU runtime, PBD generation, and six-task grounding validation |
| [`Qwen-2.5-VL-3B/`](Qwen-2.5-VL-3B/) | Qwen2.5-VL-3B compiler and runtime baseline used to validate the OELLM/HBDK/HBRT chain |

Each product owns its scripts, documentation, configuration, and generated workspace. Neither product depends on the repository name or on the former `main/` and `baselines/` layout.

Large datasets, SDK archives, relay files, and historical investigation material are intentionally kept outside Git. Generated model and validation artifacts live under each product's ignored `workspace/` directory.
