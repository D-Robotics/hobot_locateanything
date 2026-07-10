# oellm_runtime reference

Read-only snapshot of the D-Robotics `oellm_runtime` S600 SDK, copied from
`oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK/oellm_runtime/`.

## Contents

| File | Source | Purpose |
|---|---|---|
| `xlm.h` | `include/xlm.h` | High-level C API for LLM/VLM/VLA inference |
| `vlm_demo.cc` | `examples/vlm_demo/vlm_demo.cc` | Qwen2.5-VL / Qwen3-VL demo driver |
| `vlm_demo_CMakeLists.txt` | `examples/vlm_demo/CMakeLists.txt` | Build config (renamed to avoid clash) |
| `run_vlm.sh` | `examples/vlm_demo/run_vlm.sh` | Run wrapper; sets `HB_DNN_USER_DEFINED_L2M_SIZES` |
| `build_vlm.sh` | `examples/vlm_demo/build_vlm.sh` | Cross-compile build wrapper |
| `qwen2.5vl_3b_config.json` | `examples/vlm_demo/qwen2.5vl_3b_config.json` | VLM config schema template |

## Usage

These files are **not compiled** into the LocateAnything runtime. They are
kept for cross-reference when adapting the upstream flow for LA.

Borrowed code that becomes part of LA lives under `main/runtime/src/` and
`main/runtime/example/`, renamed with `Vendored from
oellm_runtime/...` annotations, per the project's code-isolation rule
(`docs/` memory: `feedback_locateanything_full_original_code`).

## Notes

- `run_vlm.sh` is the source of the `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6`
  env var requirement (see `docs/KNOWN_ISSUES.md` #017).
- Refresh this snapshot when the oellm_runtime SDK version is bumped.
