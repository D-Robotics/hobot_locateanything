# Qwen2.5-VL-3B on S600

This product records the Qwen2.5-VL-3B compiler baseline used to validate the
OELLM, HBDK, HBM, and HBRT chain on D-Robotics S600. It owns its checkpoint
metadata, compiler scripts, runtime configurations, and technical notes; it is
not a LocateAnything implementation.

## Layout

```text
Qwen-2.5-VL-3B/
├── compiler/      Vision/Language compilation and rotation validation
├── deploy/        S600 libxlm runtime configurations
├── checkpoint/    checkpoint metadata; large weights remain Git-ignored
└── docs/          validated build and numerical-alignment notes
```

Generated BC, HBO, HBM, embeddings, and rotation matrices are written to the
ignored `workspace/` directory only when a build is started.

## Build

Install the S600 OELLM compiler in the active Python environment. If
`leap_llm` is not installed as a package, point the scripts to its source:

```bash
export OELLM_LEAP_ROOT=/absolute/path/to/leap_llm
```

The scripts discover all other paths from this product directory. Defaults can
be overridden with `MODEL_PATH`, `PROCESSOR_MODEL_PATH`, `CALIBRATION_PATH`,
`ROTATION_PATH`, and `OUTPUT_DIR`.

```bash
python compiler/compile_vision.py
python compiler/compile_language.py
```

Vision and Language builds should run sequentially because HBDK compilation is
CPU- and memory-intensive. Full commands and verification results are recorded
in [the baseline guide](docs/QWEN2_5_VL_BASELINE.md).

## Verified Boundary

The validated stack used self-compiled Vision and Language HBM files plus the
matching embedding table. Text and image semantics were verified on S600. This
result validates the generic compiler/runtime chain only; LocateAnything keeps
its own MoonViT, vocabulary, coordinate tokens, and PBD graphs.
