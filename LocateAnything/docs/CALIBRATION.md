# LocateAnything Calibration Strategy

## 1. Scope

LocateAnything is a grounding model rather than a general image-question
answering model. Its activation distribution is shaped by fixed-resolution
MoonViT features, structured coordinate tokens, long object lists, PBD windows,
and AR fallback. Calibration data must represent those execution paths instead
of reusing a generic VLM question-answering corpus by default.

The active lifecycle has four commands: `prepare -> calibrate -> build -> verify`.
Their prerequisite is a frozen manifest of selected images, prompts, and
targets. Prepare runs the original Float processor and model to materialize
calibration tensors. Calibrate feeds those tensors through the OELLM models and
freezes activation scales. Build consumes the frozen scale contract, and Verify
checks data, numerical, artifact, and task-level evidence.

The release build consumes the frozen Scale manifest and graph-coverage record
produced by `compiler/quantize.py calibrate`. Merely passing a dataset path to a
compiler process is not evidence that calibration occurred. A valid build must
bind the selected manifest, its SHA256, the 1,200-sample count, and the graph
coverage record to the exported artifacts.

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

## 3. Paper-Derived Task Mix

LocateAnything-Data contains 138M natural-language queries across six task
domains. The calibration recipe must start from this distribution instead of
from a generic VLM corpus:

| Domain | Paper query share | Representative training sources named in the paper |
|---|---:|---|
| Detection | 66.9% | OpenImages, Objects365, V3Det, SKU110K, CrowdHuman, PACO, BDD100K, NuImages |
| GUI | 16.5% | ScaleCUA, GroundCUA, GTAGrounding, OSAtlas, MultiUI |
| Referring | 7.3% | RefCOCO/+/g, Flickr30k Entities, HumanRef, RoboAfford, HumanPart |
| OCR | 3.6% | BLIP3OCR, IDLOCR, TextOCR, ReCTS, LSVT, ArT, HierText |
| Layout | 3.5% | PubLayNet, DocLayNet, TableBank, M6Doc, CDLA, TabRecSet |
| Pointing | 2.2% | PixMo Points, OpenImages, Objects365, RoboAfford |

The raw percentages describe training frequency, not a deployment calibration
quota. The current release is intentionally detection-primary because object
detection is the main board workload. Minority-task records are retained to
exercise GUI, referring, OCR, layout, pointing, null-output, and AR fallback
paths; their exact counts are taken from the frozen manifest rather than
reconstructed from prose.

### Current detection-primary release profile

The release calibration set contains exactly 1,200 records:

| Source group | Records | Purpose |
|---|---:|---|
| COCO 2017 train Detection | 500 | Single-category, multi-category, and multi-instance detection |
| SKU110K train Detection | 120 | Dense retail scenes and long box sequences |
| GroundCUA train GUI | 180 | Interface-element grounding |
| RefCOCOg train Referring | 120 | Free-form referring expressions |
| HierText train OCR | 120 | Text-region localization |
| DocLayNet train Layout | 100 | Document layout categories |
| PixMo-Points train Pointing | 60 | Point prediction and short structured outputs |

The 512-record prefix is not another release dataset. It is a deterministic
checkpoint used to compare Scale values against the complete 1,200-record run.
All BC and HBM builds consume the Scale manifest produced after all 1,200 records.

The 500 COCO records use 200 single-category images, 220 two-category images, and 80
images containing three to five requested categories. Images are unique across
these strata. Each requested category has a matching `<ref>` block and one or
more boxes; the combined response is capped at 48 boxes.

COCO 2017 train is the initial source because it provides real category names,
same-image multi-category annotations and complete bounding boxes. The source
is read from the pinned `detection-datasets/coco` repository through the
Hugging Face dataset API; only the selected images are written to the local
bundle. The input remains the COCO training split, so validation images are not
introduced into calibration.

```text
COCO2017/
  hf_cache/
  detection_v7/
    coco_detection.jsonl
    images/
      000000000009.jpg
      ...
```

The following commands are maintainer-only dataset curation utilities, not the
release calibration entrypoint. Materialize the 500-record source manifest.
The default endpoint is
`https://hf-mirror.com`; set `--endpoint https://huggingface.co` when direct
Hub access is available.

```bash
python compiler/scripts/calibration/materialize_coco.py \
  --output-dir /data/COCO2017/detection_v7 \
  --cache-dir /data/COCO2017/hf_cache \
  --single-category 200 \
  --two-category 220 \
  --multi-category 80 \
  --seed 20260728
```

Combine those 500 COCO records with 120 SKU110K records from the frozen
six-domain source bundle. The remaining 580 records are carried forward from
the GUI, Referring, OCR, Layout, and Pointing training sources:

```bash
python compiler/scripts/calibration/compose_detection.py \
  --coco-jsonl /data/COCO2017/detection_v7/coco_detection.jsonl \
  --baseline-selected-jsonl /data/calibration/la_820_hq_v6/selected.jsonl \
  --output-dir workspace/calibration/sources_release \
  --seed 20260729
```

Freeze and materialize the complete release manifest with explicit quotas:

```bash
python compiler/scripts/calibration/prepare.py select \
  --input-jsonl workspace/calibration/sources_release/detection_coco.jsonl \
  --input-jsonl workspace/calibration/sources_release/detection_retail.jsonl \
  --input-jsonl workspace/calibration/sources_release/other_tasks.jsonl \
  --output-dir workspace/calibration/current \
  --num-samples 1200 \
  --quota detection=620 \
  --quota gui=180 \
  --quota referring=120 \
  --quota ocr=120 \
  --quota layout=100 \
  --quota pointing=60 \
  --seed 20260729
```

The generated `selection_summary.json` is the source of truth for counts and
input hashes. Do not infer the release mix from filenames alone.

The generated prompt uses the tokenizer's category separator:

```text
Locate all the instances that matches the following description: person</c>motorcycle.
```

The target response groups boxes by category:

```text
<ref>person</ref><box>...</box><ref>motorcycle</ref><box>...</box>
```

Do not replace `</c>` with a comma and do not rewrite Detection labels to the
generic word `object`. Both transformations remove the multi-category path
that the detection-primary release is intended to calibrate.

## 4. Dataset Contract

The source manifest uses one JSON object per image/query. It intentionally
keeps source metadata separate from generated tensors:

```jsonl
{"sample_id":"refcocog-train-000001","task":"referring","source":"RefCOCOg","split":"train","license":"CC BY 4.0","image":"images/000001.jpg","phrase":"the red car on the left","target_response":"<ref>the red car on the left</ref><box><100><200><400><500></box>"}
```

Requirements:

- every record declares `task`, `source`, `split`, `license`, and `image`;
- image paths are resolved relative to the source JSONL, a per-record root, or
  `--image-root`;
- `prompt` may be supplied explicitly; otherwise the selector renders the
  exact task template from Table 9 of the paper using `categories`, `phrase`,
  `multiple`, and `output_type`;
- coordinates use the 1001 tokens `<0>` through `<1000>`;
- `target_response` is optional but recommended for negative samples and rare
  formats; the native model prediction is always retained separately;
- source images remain at native resolution in the selected bundle;
- generation uses aspect-ratio-preserving letterbox to a fixed 672x672 canvas,
  producing exactly 2304 patches and 576 visual tokens;
- `<box>` point and box coordinates are transformed into the letterboxed
  coordinate domain; the source response and scale/padding metadata are retained
  so runtime outputs can be mapped back to the original image;
- malformed records, missing images, unexpected token counts, and sequences
  longer than the compiled profile fail before calibration begins.

The selector rejects `val`, `validation`, `test`, `dev`, and `eval` splits by
default, hashes every image, removes cross-dataset duplicates by content, and
copies only the selected images into the bundle.

## 5. PyTorch Materialization

The internal tools under `compiler/scripts/calibration/` collect and select
source data before the release workflow begins.

First collect pinned training subsets. The collector uses streaming datasets
and downloads only accepted images. Source capacities should remain larger than
1,200 so hashing, invalid rows, and selection have headroom. For a fixed SKU110K shard,
pass `--local-parquet detection=/path/to/train-00000-of-00019.parquet` to avoid
remote streaming.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean

python compiler/scripts/calibration/collect_sources.py \
  --output-dir workspace/calibration/la_sources \
  --hf-endpoint https://hf-mirror.com \
  --seed 20260718 \
  --shuffle-buffer 10000 \
  --resume
```

The pinned adapters are:

| Domain | Streaming source | Adapter output |
|---|---|---|
| Detection | `benjamintli/sku110k` | dense SKU110K training boxes, capped at 48 per image |
| GUI | `likaixin/GroundCUA-train` | natural-language instruction and normalized box/point |
| Referring | `sionic-ai/refcocog_object_detection` | embedded COCO-train image, phrase, and box |
| OCR | `Berzerker/ocr_hiertext` | recognized text and word boxes |
| Layout | `docling-project/DocLayNet-v1.2` | layout classes and block boxes |
| Pointing | `allenai/pixmo-points` | label and one or more points |

Every Hugging Face source revision is pinned in the collector. The large-image
SKU110K source uses a bounded streaming shuffle buffer so collection cannot
retain thousands of decoded shelf images in memory. Dataset mirrors do not
replace upstream license terms; the generated manifest records both the mirror
revision and the applicable upstream license pointer.

The selected COCO and retained records are composed into
`workspace/calibration/current/selected.jsonl`. Its record count, task counts,
source splits, and content hashes are checked before native LocateAnything
inference and tensor materialization begin. Run the static gate first:

```bash
python compiler/quantize.py prepare --preflight-only \
  --selected-jsonl workspace/calibration/current/selected.jsonl \
  --upstream-source /path/to/Eagle/Embodied \
  --model-path workspace/models/LocateAnything-3B
```

This gate checks the frozen manifest SHA256, all 1,200 image hashes, task and
source quotas, checkpoint shards, tokenizer IDs, processor geometry, 576 image
placeholders, and the 1,024-token Prefill limit. It imports the processor and
tokenizer metadata but never loads the 3B weights, initializes CUDA, or runs
model inference. A passing report is written beside the generated bundle as
`prepare_preflight.json`.

After the static gate passes, materialize the PyTorch calibration tensors:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate locateanything

python compiler/quantize.py prepare \
  --selected-jsonl workspace/calibration/current/selected.jsonl \
  --output-dir workspace/calibration/current/generated \
  --upstream-source /path/to/Eagle/Embodied \
  --model-path workspace/models/LocateAnything-3B \
  --device cuda:0 \
  --max-new-tokens 1024 \
  --slow-samples 128 \
  --resume
```

Maintainers may monitor a long preparation run from a second shell without
parsing the model log:

```bash
python compiler/scripts/common/monitor.py \
  --progress-jsonl workspace/calibration/current/generated/generation_progress.jsonl \
  --total 1200 \
  --pid-file workspace/logs/calibration_prepare.pid \
  --exit-file workspace/logs/calibration_prepare.exit.txt
```

The following focused tensor reload is an internal integrity diagnostic. It is
not a separate release stage:

```bash
python compiler/scripts/calibration/validate.py \
  --selected-jsonl workspace/calibration/current/selected.jsonl \
  --generated-jsonl workspace/calibration/current/generated/generated.jsonl \
  --output-json workspace/calibration/current/generated/prepared_bundle_validation.json
```

The generation phase imports the original `locateanything_worker.py`, calls
the upstream processor and PyTorch `generate()` path, and stores:

- Hybrid/PBD predictions for every sample;
- Slow/AR predictions for a deterministic cross-domain subset;
- fixed-profile MoonViT inputs `(1, 2304, 588)`;
- prompt IDs and masks produced by the native processor;
- native projected visual features `(1, 576, 2048)` before the compiler-side
  hidden-domain rotation;
- response token IDs, special-token IDs, artifact SHA256 values, task counts,
  and resume-safe progress records.

The tensor bundle is calibration input, not an HBM artifact. The Leap models
must still replay these tensors in eager mode so their own observers update.
A 448x448 tensor bundle is not valid calibration input for a 672x672 HBM:
the patch count, visual-token count, prompt placeholder count, and activation
distribution all differ.

## 6. Data Isolation

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

## 7. Calibration Execution

Calibration must run before `compile_mode(True)` and BC export:

1. Load the checkpoint, tokenizer, processor, and fixed 672x672 profile.
2. Apply the shared hidden-domain transform to Language weights, embedding
   table, and MoonViT projector.
3. Run MoonViT eager forward on every selected image to collect Vision
   fake-quant ranges.
4. Build multimodal prefill embeddings with exactly 576 visual tokens and run
   Language eager forward with real masks, position IDs, and KV tensors.
   With `chunk_size=1024`, the image leaves at most 448 positions for text and
   chat-template tokens.
5. Run representative PBD `q=6` windows containing coordinate and text-mask
   tokens.
6. Run representative AR `q=1` fallback windows.
7. Freeze and print all `ConstFakeQuant.absmax` and `RMSNorm` scale statistics.
8. Export BC only after the scale audit passes.

The unified calibration stage fixes the release sample count at 1,200 and
compares the deterministic 512-sample checkpoint with the complete Scale
snapshot. It refuses a zero-exit run whose durable artifacts disagree:

```bash
python compiler/quantize.py calibrate --component all \
  --generated-jsonl workspace/calibration/current/generated/generated.jsonl \
  --model-path workspace/models/LocateAnything-3B \
  --output-dir workspace/calibration/current/statistics \
  --max-samples 1200 --checkpoint-samples 512 --resume
```

Use `--dry-run` to print resolved paths and commands without collecting activation statistics.
A dry run is never recorded as a completed calibration.

## 8. Acceptance Gates

A calibrated build must satisfy all of the following:

- calibration parameters are consumed by the selected model factory;
- the log records manifest SHA256, 1,200 samples, task counts, and both the
  512-sample and 1,200-sample Scale summaries;
- graph coverage contains `visual` and the complete 13-graph Language family:
  `prefill`, `decode`, `decode_ar`, `decode_pbd_q7..q12`, and
  `decode_ar_q2..q5`;
- no quantized module that executed during calibration retains an unexplained
  zero `absmax`;
- repeated runs with the same manifest produce the same scale summary;
- a held-out PyTorch comparison passes for Vision output, Language logits, and
  KV tensors;
- calibrated and uncalibrated artifacts are stored in separate directories and
  compared with identical board inputs.

## 9. Current Artifact Classification

The Language build started on 2026-07-18 under
`la_fix011_hidden_domain_language` did not execute a calibration forward. It
linked successfully at 16:58 CST and is retained only as a compiler-structure
control. It must not be marked as a release candidate or used to trigger the
release Vision build.
