# Deployment Workspace

`main/` contains LocateAnything compilation entrypoints, runtime code, and
generated-artifact locations. Model weights, HBM/BC/HBO files, logs, and local
build products are intentionally excluded from Git.

| Directory | Ownership |
|---|---|
| `scripts/` | BC validation, detached HBM compilation, artifact checks |
| `vision/` | MoonViT + projector generated artifacts |
| `language/` | Qwen2.5/PBD generated artifacts and tokenizer staging |
| `runtime/` | Custom S600 host runtime and focused C++ probes |
| `configs/` | Versioned runtime/compiler configuration templates |
| `examples/` | PyTorch and HBM validation utilities |
| `golden/` | Generated PyTorch reference tensors; ignored |
| `outputs/` | Generated BC/HBO/HBM/embed outputs; ignored |
| `logs/` | Compiler and board-validation logs; ignored |
| `benchmarks/` | Generated benchmark results; ignored |

The Qwen2.5-VL compiler-chain validation is organized under
`baselines/qwen2_5_vl/`; LocateAnything artifacts remain under `main/` so both
sets can be compared without path or checksum ambiguity.

Recommended order:

1. Run `scripts/validate_locateanything_rotation.py`.
2. Export Language and Vision with `--export_only`.
3. Launch `compile_locateanything_language.sh` and
   `compile_locateanything_vit.sh`.
4. Record SHA256 and graph contracts.
5. Transfer to a versioned S600 model directory.
6. Validate numerical agreement before semantic/PBD tests.
