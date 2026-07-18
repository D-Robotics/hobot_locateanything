# Qwen2.5-VL-3B S600 Compiler Baseline

This directory archives the compiler baseline used before adapting
LocateAnything-3B. It is evidence that the public OELLM 1.0.5 toolchain can
produce working Qwen2.5-VL-3B Vision and Language HBMs for S600 when the
hidden-domain contract is handled consistently.

## Verified Baseline

- Fix #009: self-compiled Vision HBM aligned to the 2048-dimensional S600
  reference hidden domain.
- Fix #010: self-compiled Language HBM and self-generated embedding table in
  the same hidden domain.
- S600 text test: `hi?` produced a normal assistant response.
- S600 image test: `image1.jpg` was correctly described as a red panda on a
  wooden platform.
- Runtime: the SDK runtime was used, but no precompiled model HBM or embedding
  table was loaded by the final test configuration.

The forum post at <https://forum.d-robotics.cc/t/topic/35332> is an independent
developer success case, not an official from-zero compilation guide. It was
used as engineering reference only.

## Contents

- `configs/test_fix010_full_self.json`: validated S600 runtime configuration.
- `reference/`: exact host-side experiment snapshots. These retain the paths
  from the original 4090 RCA and are not polished command-line entrypoints.
- `../../docs/rca/sdk_compiler_rca_review.md`: complete investigation log.
- `../../docs/rca/qwen2_5_vl_vision_fix009.md`: focused Vision alignment report.
- `../../docs/tutorials/QWEN2_5_VL_BASELINE.md`: concise reproduction guide.

Qwen2.5-VL is a compiler/runtime baseline only. LocateAnything has a different
MoonViT encoder, vocabulary, PBD decode contract, checkpoint layout, and host
runtime.
