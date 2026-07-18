<div align="center">

<img src="assets/LocateAnything.jpg" alt="LocateAnything on D-Robotics S600" width="820">

# LocateAnything on D-Robotics S600

An architecture-preserving deployment stack for LocateAnything-3B on the
D-Robotics S600 BPU, covering compilation, quantization, HBM packaging,
runtime integration, and staged validation.

[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-D--Robotics%20S600-35a853)](https://developer.d-robotics.cc/)
[![SDK](https://img.shields.io/badge/OELLM-1.0.5-2563eb)](oellm/README.md)
[![Model](https://img.shields.io/badge/model-LocateAnything--3B-f59e0b)](https://huggingface.co/nvidia/LocateAnything-3B)
[![PBD](https://img.shields.io/badge/decoding-PBD%20q%3D6-d946ef)](docs/SOURCE_REVIEW.md)

**English** | [中文](README.zh-CN.md)

</div>

## Overview

LocateAnything-3B targets open-vocabulary grounding, region understanding, and
structured coordinate generation through a MoonViT vision encoder, a Qwen2.5
language decoder, and Parallel Block Decoding (PBD). This project builds an
S600 compiler and runtime stack around the model's native interfaces, with a
reproducible path from checkpoint weights and BC/HBM artifacts to board-side
execution.

The project first establishes Qwen2.5-VL-3B as an independently validated
reference for the OELLM/HBDK compiler and HBRT runtime chain. Its methods for
static-image patch embedding, hidden-domain alignment, Language compilation,
and numerical verification are then applied to LocateAnything's MoonViT,
Qwen decoder, and PBD graphs.

## Highlights

- **Architecture preservation**: MoonViT, 1D RoPE, the 152,681-token
  vocabulary, coordinate tokens, and six-token PBD semantics remain explicit.
- **Staged graph contracts**: Vision, prefill, PBD decode (`q=6`), and AR
  decode (`q=1`) are exported as independently verifiable BPU graphs.
- **Zero-overhead hidden-domain alignment**: a signed Walsh-Hadamard transform
  is folded into embeddings, Attention/MLP projections, lm_head, and the
  MoonViT projector without adding runtime matrix multiplications.
- **Reproducible builds**: numerical preflight, BC export, detached HBM
  compilation, versioned artifacts, and checksum management are documented.
- **Evidence-driven validation**: source contracts, tensor interfaces,
  numerical alignment, and S600 results define each acceptance gate.

## Architecture

```mermaid
flowchart LR
    IMAGE["Image"] --> PATCH["Patchify 448x448<br/>1024 x 588"]
    TEXT["Prompt"] --> TOKENIZER["LocateAnything tokenizer<br/>vocab 152681"]

    subgraph VISION["Vision HBM"]
        PATCH --> MOONVIT["MoonViT<br/>27 layers, hidden 1152"]
        MOONVIT --> PROJECTOR["2x2 merge + projector<br/>4608 -> 2048 -> 2048"]
    end

    subgraph LANGUAGE["Language HBM"]
        TOKENIZER --> EMBEDS["Text embeddings"]
        PROJECTOR --> MERGE["Visual token insertion"]
        EMBEDS --> MERGE
        MERGE --> PREFILL["Prefill<br/>chunk 1024"]
        PREFILL --> KV["KV cache<br/>length 2048"]
        KV --> PBD["PBD decode<br/>q=6"]
        KV --> AR["AR decode<br/>q=1"]
    end

    PBD --> HYBRID["Hybrid generation"]
    AR --> HYBRID
    HYBRID --> BOX["ref / box parser"]
```

## Quick Start

### 1. Clone the project and model source

```bash
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
git clone https://github.com/NVlabs/Eagle.git eagle

hf download nvidia/LocateAnything-3B \
  --local-dir eagle/Embodied/LocateAnything-3B
```

### 2. Install the compiler adapter

Install the D-Robotics S600 OELLM 1.0.5 SDK first, then install the vendored
toolchain in the SDK environment:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

cd toolchain
pip install -e . --no-deps
cd ..
```

### 3. Run the numerical preflight

```bash
PYTHONPATH=$PWD/toolchain \
python main/scripts/validate_locateanything_rotation.py \
  --model-path eagle/Embodied/LocateAnything-3B \
  --component all \
  --device cuda:0 \
  --dtype float32
```

### 4. Export BC graphs before the long build

```bash
export REPO_ROOT=$PWD
export CONDA_ENV=oellm_clean

EXPORT_ONLY=1 ./main/scripts/compile_locateanything_language.sh
tail -f main/logs/locateanything_language_compile.log

EXPORT_ONLY=1 ./main/scripts/compile_locateanything_vit.sh
tail -f main/logs/locateanything_vit_compile.log
```

### 5. Compile the HBM artifacts

Run Language and Vision sequentially to avoid compiler resource contention:

```bash
./main/scripts/compile_locateanything_language.sh
tail -f main/logs/locateanything_language_compile.log

# Start after the Language build has completed.
./main/scripts/compile_locateanything_vit.sh
tail -f main/logs/locateanything_vit_compile.log
```

The complete environment setup, source changes, mathematical derivation, build
commands, and validation gates are documented in the
[compiler porting guide](docs/COMPILER_PORTING_GUIDE.zh-CN.md).

## Documentation

| Document | Description |
|---|---|
| [Documentation index](docs/README.md) | Entry point for guides, architecture, RCA, and references |
| [Compiler porting guide](docs/COMPILER_PORTING_GUIDE.zh-CN.md) | From Qwen2.5-VL chain validation to LocateAnything HBM compilation |
| [Source review](docs/SOURCE_REVIEW.md) | Checkpoint contract, MoonViT, Qwen decoder, and PBD semantics |
| [Runtime architecture](docs/RUNTIME_ARCHITECTURE.md) | Host/BPU split and runtime module design |
| [Known issues](docs/KNOWN_ISSUES.md) | Reproducible failures, evidence, fixes, and prevention |
| [Qwen2.5-VL baseline](baselines/qwen2_5_vl/README.md) | Compiler-chain validation artifacts and scripts |

## Repository Layout

```text
oe_locateanything/
├── baselines/qwen2_5_vl/       Qwen2.5-VL compiler-chain validation
├── docs/                       guides, architecture, RCA, and issue records
├── main/                       build scripts, runtime, configs, and examples
├── toolchain/leap_llm/         OELLM source and LocateAnything adapters
├── eagle/                      NVIDIA Eagle source and model weights
└── oellm/                      D-Robotics SDK placement
```

Generated weights, BC/HBO/HBM files, logs, and runtime build directories are
kept outside Git history.

## Citation

```bibtex
@misc{locateanything2025,
  title  = {LocateAnything},
  author = {NVIDIA},
  year   = {2025},
  url    = {https://huggingface.co/nvidia/LocateAnything-3B}
}

@misc{oe_locateanything2026,
  title  = {oe_locateanything: LocateAnything-3B Deployment on D-Robotics S600},
  author = {Xu, Kangjie},
  year   = {2026},
  url    = {https://github.com/LiuAnclouds/oe_locateanything}
}
```

## Acknowledgements

- [NVIDIA Eagle](https://github.com/NVlabs/Eagle) and LocateAnything teams
- [Moonshot AI](https://github.com/MoonshotAI) for MoonViT
- [Qwen](https://github.com/QwenLM/Qwen2.5) for the language model family
- [D-Robotics](https://developer.d-robotics.cc/) for the S600 platform and OELLM toolchain
- The D-Robotics developer community for shared deployment experience

## License

This project is licensed under [CC BY-NC 4.0](LICENSE). Model weights, the
D-Robotics SDK, NVIDIA Eagle, and vendored upstream components retain their
respective licenses.
