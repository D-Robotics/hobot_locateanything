# Qwen2.5-VL-3B S600 Compiler Baseline

This directory contains the Qwen2.5-VL compiler-chain validation used by the
LocateAnything deployment. It demonstrates the complete path from checkpoint
weights to working Vision/Language HBMs, embedding export, S600 runtime loading,
and text/image semantic verification.

## Verified Baseline

- The self-compiled Vision HBM is aligned to the 2048-dimensional S600
  reference hidden domain.
- The self-compiled Language HBM and generated embedding table use the same
  hidden domain.
- S600 text test: `hi?` produced a normal assistant response.
- S600 image test: `image1.jpg` was correctly described as a red panda on a
  wooden platform.
- Runtime: the SDK runtime was used, but no precompiled model HBM or embedding
  table was loaded by the final test configuration.

The forum post at <https://forum.d-robotics.cc/t/topic/35332> is a deployment
case published by a community developer. Its practical workflow informed the
project's experiment design, while the Qwen2.5-VL results here are reproduced
from this repository's code and artifacts.

## Contents

- `configs/test_fix010_full_self.json`: validated S600 runtime configuration;
  the filename is retained to preserve experiment provenance.
- `reference/`: exact host-side experiment snapshots. These retain the paths
  from the original 4090 RCA and can be configured for another build host.
- `../../docs/rca/sdk_compiler_rca_review.md`: complete investigation log.
- `../../docs/rca/qwen2_5_vl_vision_fix009.md`: focused Vision alignment report.
- `../../docs/tutorials/QWEN2_5_VL_BASELINE.md`: concise reproduction guide.

The validated chain is carried forward into LocateAnything with its MoonViT
encoder, 152,681-token vocabulary, PBD decode graphs, checkpoint layout, and
Host runtime.
