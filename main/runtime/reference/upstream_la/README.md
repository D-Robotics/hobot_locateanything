# upstream LA reference (upstream snapshot)

Read-only snapshot of the LocateAnything-3B upstream source, copied from
the build-host checkpoint `eagle/Embodied/LocateAnything-3B/` for
cross-reference when porting the PBD generate loop to C++.

## Contents

| File | Purpose |
|---|---|
| `modeling_locateanything.py` | Top-level `generate()` + `_prepare_inputs_in_mtp` / `_prepare_input_in_ar` / `_sample_token_in_mtp` / `_sample_token_in_ar` (lines 304-537) |
| `generate_utils.py` | `sample_tokens` / `handle_pattern` / `is_valid_box_frame` / `decode_bbox_avg` / `decode_ref` / `get_token_ids_from_config` (lines 15-503) |
| `modeling_qwen2.py` | `prepare_inputs_for_generation` (lines 1551-1606) — trims input to uncached suffix |
| `modeling_vit.py` | MoonViT vision tower |
| `mask_sdpa_utils.py` | `update_causal_mask_for_one_gen_window_2d` (PBD mask) |
| `mask_magi_utils.py` | magi attention (compile-time only) |
| `processing_locateanything.py` | `LocateAnythingProcessor` (prompt assembly) |
| `image_processing_locateanything.py` | image preprocessing (resize/normalize/patchify) |
| `configuration_locateanything.py` / `configuration_qwen2.py` | config dataclasses |
| `batch_infer.py` | minimal inference CLI (query format: `cat</c>car`) |
| `config.json` | all special token ids + block_size=6 |

## Usage

Reference for `main/runtime/src/` + `main/runtime/docs/INFERENCE_FLOW.md`.
Any C++ port of the generate loop / sampling / handle_pattern must match
these files line-for-line; deviations are bugs.

Per `feedback_locateanything_read_upstream_first` memory: change LA leap
DSL / runtime code only after reading the corresponding upstream function.

## Do not edit

Upstream snapshot. Behavior changes go in `main/runtime/src/`.
