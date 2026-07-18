# Qwen2.5-VL-3B Baseline on S600

## Purpose

Qwen2.5-VL is the first validation stage for the OELLM/HBDK compiler and S600
runtime chain. It provides a controlled reference for checkpoint loading,
Vision/Language compilation, embedding export, hidden-domain analysis, and
board-side text/image verification. LocateAnything reuses this workflow with
its MoonViT, vocabulary, PBD, and runtime contracts.

## Root Cause and Fixes

1. Static-image patch embedding must fold the temporal Conv3d weights by
   summing both temporal slices into the compiler's Conv2d weight.
2. The public reference Vision, Language, and embedding artifacts use a common
   2048-dimensional signed Walsh-Hadamard hidden domain.
3. Fix #009 folds that transform into the Vision merger output projection.
4. Fix #010 folds it into embeddings, every Attention/MLP residual boundary,
   final norm/lm_head, and the calibration Vision output.

The SDK provides the compiler/runtime components and precompiled reference
artifacts. This project reconstructs and validates the checkpoint-to-HBM steps
through controlled artifact comparisons.

## Reproduction Material

```text
baselines/qwen2_5_vl/
  configs/test_fix010_full_self.json
  reference/compile_fix009_official_domain.py
  reference/compile_fix010_language_official_domain.py
  reference/qwen25_language_official_rotation.py
  reference/validate_fix010_language_rotation.py
docs/rca/sdk_compiler_rca_review.md
```

The reference Python files are exact experiment snapshots and contain the
original 4090 paths. Configure those paths for the target build host.

## Validated S600 Command

```bash
cd ~/oe_locateanything/oellm_runtime/examples/vlm_demo
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
./vlm -c test_fix010_full_self.json
```

The final test loaded the Fix #009 Vision HBM, Fix #010 Language HBM, and
Fix #010 embedding table. Text and image semantics were normal.
