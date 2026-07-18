# LocateAnything Calibration Strategy

## 1. Scope

LocateAnything is a grounding model rather than a general image-question
answering model. Its activation distribution is shaped by fixed-resolution
MoonViT features, structured coordinate tokens, long object lists, PBD windows,
and AR fallback. Calibration data must represent those execution paths instead
of reusing a generic VLM question-answering corpus by default.

The current independent builders, `locateanything-lm-3b` and
`locateanything-vit-3b`, do not yet consume `--calib_json_path` or
`--calib_image_path`. A path passed by the shell script is therefore not
evidence that calibration occurred. A valid build log must contain an explicit
sample count, stage names, and a scale summary.

## 2. Why Calibration Is Required

The LocateAnything Leap graph contains data-dependent modules:

- `ConstFakeQuant.forward()` records activation `absmax`; its initial value is
  zero and `build()` embeds the recorded range.
- `RMSNorm.forward()` records hidden-state energy and updates `i_scale` and
  `i_scale_pow`; their initial values are one.
- Vision attention uses fake-quantized QK/WV matmuls.
- Language attention uses fake-quantized QK/WV matmuls and calibrated cache
  quantizers in addition to dynamically quantized Linear layers.

`DynamicQuantLinear` and `DynamicQuantMatmul` derive activation scales at
runtime, but that does not remove the calibration requirement for the explicit
fake-quant and norm modules around them.

## 3. Dataset Contract

Use the upstream LocateAnything JSONL convention so data provenance remains
compatible with the original project:

```jsonl
{"task":"phrase_grounding","conversations":[{"from":"human","value":"Locate a single instance that matches the following description: the red car on the left."},{"from":"gpt","value":"<ref>the red car on the left</ref><box><100><200><400><500></box>"}],"image":"images/000001.jpg"}
```

Requirements:

- image paths are resolved relative to a declared dataset root;
- coordinates use the 1001 tokens `<0>` through `<1000>`;
- assistant responses are retained during calibration so coordinate, ref, box,
  null, and termination-token activations are represented;
- every image is processed through the compiled 448x448 profile, producing
  exactly 1024 patches and 256 visual tokens;
- malformed records, missing images, unexpected token counts, and sequences
  longer than the compiled profile fail before calibration begins.

## 4. Recommended Composition

Start with 128 to 256 deterministic samples. Increase the count only after a
scale-convergence comparison shows that the smaller set is insufficient.

| Task family | Share | Required coverage |
|---|---:|---|
| Multi-class object detection | 25% | sparse and crowded scenes, small and large boxes |
| Single-instance phrase grounding | 15% | attributes, spatial relations, long referring phrases |
| Multi-instance phrase grounding | 10% | repeated classes and variable box counts |
| GUI grounding | 15% | box and point outputs, mobile, desktop, and web layouts |
| OCR and scene text | 10% | short labels, dense text, multilingual text |
| Document layout | 10% | title, paragraph, table, figure, and form regions |
| Long-tail and fine-grained localization | 10% | uncommon classes and small targets |
| Negative/no-object cases | 5% | `<box>none</box>`, null, and early termination |

The task proportions are a deployment starting point, not a model-quality
benchmark. Adjust them to the target product workload and record the final
recipe with the HBM artifacts.

## 5. Data Isolation

- Build calibration data from training splits or a dedicated calibration
  pool.
- Keep COCO `val2017`, grounding validation sets, and board smoke-test images
  outside the calibration set when they are used for reported evaluation.
- Deduplicate images by content hash across calibration and evaluation.
- Record dataset name, split, license, source URL, selection seed, and SHA256
  for every manifest.

The existing 256-image COCO `val2017` subset is retained as verification data;
it should not become the final calibration set if it remains part of numerical
or semantic evaluation.

## 6. Calibration Execution

Calibration must run before `compile_mode(True)` and BC export:

1. Load the checkpoint, tokenizer, processor, and fixed 448x448 profile.
2. Apply the shared hidden-domain transform to Language weights, embedding
   table, and MoonViT projector.
3. Run MoonViT eager forward on every selected image to collect Vision
   fake-quant ranges.
4. Build multimodal prefill embeddings with exactly 256 visual tokens and run
   Language eager forward with real masks, position IDs, and KV tensors.
5. Run representative PBD `q=6` windows containing coordinate and text-mask
   tokens.
6. Run representative AR `q=1` fallback windows.
7. Freeze and print all `ConstFakeQuant.absmax` and `RMSNorm` scale statistics.
8. Export BC only after the scale audit passes.

## 7. Acceptance Gates

A calibrated build must satisfy all of the following:

- calibration parameters are consumed by the selected model factory;
- the log records manifest SHA256, sample count, task counts, and all four graph
  paths: Vision, prefill, PBD decode, and AR decode;
- no quantized module that executed during calibration retains an unexplained
  zero `absmax`;
- repeated runs with the same manifest produce the same scale summary;
- a held-out PyTorch comparison passes for Vision output, Language logits, and
  KV tensors;
- calibrated and uncalibrated artifacts are stored in separate directories and
  compared with identical board inputs.

## 8. Current Artifact Classification

The Language build started on 2026-07-18 under
`la_fix011_hidden_domain_language` did not execute a calibration forward. It is
useful only as a compiler-structure control if retained. It must not be marked
as a release candidate regardless of whether HBM linking succeeds.
