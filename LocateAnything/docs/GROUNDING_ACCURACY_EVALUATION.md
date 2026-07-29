# LocateAnything grounding accuracy evaluation

The task stage of `compiler/quantize.py verify` measures grounding response
quality independently from D3 tensor integrity and S600 runtime stability. Its
internal evaluator accepts both current D3 `generated.jsonl` records and flat
device prediction records.

## D3 evaluation

```bash
python compiler/quantize.py verify --component all --level task \
  --predictions-jsonl workspace/calibration/current/generated/generated.jsonl \
  --reference-jsonl workspace/calibration/current/generated/generated.jsonl
```

The output paths and IoU threshold come from `compiler/config.yaml`.

The evaluator discovers `prediction.hybrid.answer` and
`prediction.slow.answer` automatically. Use repeated `--mode` options to require
specific modes. A requested but missing mode is retained in the end-to-end
`overall` denominator as an invalid response, not silently omitted.

D3 normally generates `slow` for only a sampled subset. For that reason every
mode has three related outputs:

- `prediction_coverage`: how many rows actually contain that mode;
- `overall`: all reference rows, with missing outputs scored as zero;
- `available_predictions_only`: accuracy on rows where that mode was actually
  run; a present but malformed answer still scores zero.

Use `available_predictions_only` for slow-path model quality and report its
coverage beside it. Use `overall` when measuring end-to-end completion of an
inference run that was expected to produce every row.

## S600 prediction format

The recommended device-side JSONL format is one row per sample and mode:

```json
{"bundle_id":"0000-detection-...","mode":"s600_hybrid","answer":"<ref>object</ref><box><10><20><30><40></box>"}
```

Join it to the D3 manifest so evaluation uses the 672x672 profile-adjusted
reference coordinates:

```bash
python compiler/quantize.py verify --component all --level task \
  --predictions-jsonl workspace/evaluation/release_candidate/predictions.jsonl \
  --reference-jsonl workspace/evaluation/current/selected.jsonl
```

Nested `prediction`/`predictions`, flat `answer`, `hybrid_answer`, `slow_answer`,
`prediction_response`, `response`, and `output` fields are also accepted. Every
row needs `bundle_id` (or `sample_id`); the reference must supply `task` and
`profile_target_response` or `target_response`.

When `--reference-jsonl` is supplied, that manifest defines the evaluation
universe. A reference row with no corresponding device prediction remains in
the denominator as an invalid, zero-match output. `prediction_coverage` reports
this separately from syntax validity. Profile-adjusted targets take precedence
over unadjusted targets even when the profile target is carried by the device
row and the separate reference contains only the original target.

## Metrics

- **Format valid rate**: strict parse rate for one or more
  `<ref>label</ref>` groups, each followed by one or more two-coordinate points
  or four-coordinate positive-area boxes. Coordinates must be integers in
  `[0,1000]`. Invalid predictions remain in every denominator with zero matches.
  A single trailing `<|im_end|>` plus surrounding whitespace is accepted because
  it is LocateAnything's configured EOS and upstream decoding returns it.
  `</s>`, `<|endoftext|>` (the configured pad token), repeated EOS, unknown
  special tokens, and arbitrary trailing prose remain invalid.
- **Label/ref precision, recall, F1**: multiset match of Unicode-NFKC and
  whitespace-normalized ref labels. Case and punctuation are preserved so OCR
  spelling fidelity is not overstated.
- **Box precision, recall, F1**: one-to-one, exact-label box matching at the
  release threshold IoU `>= 0.90`. Matching maximizes match count; a
  geometrically correct box with the wrong label is not a true positive.
  Lower thresholds may be reported as diagnostics, but they do not replace the
  0.90 release result.
- **Mean IoU of true positives**: mean IoU only among threshold-passing matches.
  It must be read together with recall because it excludes misses.
- **Single-box mean IoU**: for records with exactly one target box, IoU is
  computed only when exactly one same-label box is predicted; missing, extra, or
  wrong-label output contributes zero.
- **Point PCK**: one-to-one, exact-label matching at Euclidean distance divided
  by the diagonal of the normalized `1000 x 1000` grid. PCK@0.05 and PCK@0.10
  are emitted by default with precision, recall, and F1.
- **Point target distance**: target-centric one-to-one nearest distance in grid
  units and normalized diagonal units. An unmatched target receives the full
  grid-diagonal penalty. Extra predictions are reflected by PCK precision.
- **Structured exact match**: normalized refs and all typed coordinates must
  match in order and value.

Every metric is reported overall and separately for Detection, GUI, Referring,
OCR, Layout, and Pointing. The optional details JSONL records parse failures and
per-sample matches for error analysis.

## Interpretation limits

This tool evaluates response agreement with the supplied annotations. It does
not establish open-world accuracy, calibration quality, robustness, or S600
stability by itself. In particular, evaluating the same training examples used
for calibration is a calibration-set fidelity check, not a held-out accuracy
benchmark. SKU110K targets are an annotated subset (up to the response limit),
so Detection scores agreement with that subset rather than exhaustive product
detection. OCR/layout spelling quality is represented by exact ref matching;
there is no edit-distance or semantic-label metric. The target-distance pairing
is deterministic nearest-first rather than a global minimum-cost assignment;
PCK threshold matching, by contrast, uses maximum-cardinality matching. Report held-out dataset
identity, split, sample count, model artifact hashes, runtime mode, and metric
thresholds with any published result.

## Fixed selection-held-out set

The project keeps a balanced 120-record evaluation set under
`workspace/evaluation/current`: 20 records per domain. Before a release result
is reported, this set must be checked against the current 1,200-record
calibration manifest by exact image SHA256, source sample ID, and cross-set
dHash distance `<= 4`. Leakage evidence produced for an older 820-record
manifest is historical and cannot satisfy this gate. The current evaluation
manifest and leakage-audit hashes must accompany S600 metrics.

This set measures unseen-sample behavior within the same six source datasets.
It is a valid selection-held-out test, but it is not evidence of cross-dataset
or open-world generalization. A cross-dataset claim needs separately sourced,
licensed data and its own frozen audit report.
