# Contributing

## Boundaries

- `compiler/` is the editable OELLM compiler package and must retain the
  LocateAnything MoonViT, Qwen decoder, 152681 vocabulary, 672 profile, and
  PBD `q=6` contracts.
- `compiler/scripts/` contains calibration, compilation, validation, and
  compiler-side evaluation entrypoints.
- `deploy/` is the S600 C++ host runtime, CLI, tokenizer, and deployment tool.
- `tests/` contains repository tests; generated tensors, model weights, logs,
  and HBM/BC/HBO files stay outside Git.
- `workspace/` contains generated data and is never treated as source code.

## Before a change

```bash
python -m pytest tests -q
git diff --check
```

Long compiler and board jobs must use a new output/run directory and record
the command, environment, exit code, artifact SHA256, and host.
