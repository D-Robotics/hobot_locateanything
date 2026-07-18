# Build Status

Last updated: 2026-07-18 (Asia/Shanghai)

## Fix #011 Language

| Field | Value |
|---|---|
| Source commit | `49e9702` |
| Compiler host | `kangjie.xu@10.112.20.45` |
| Process-group PID | `2094764` |
| Output | `/home/kangjie.xu/oellm_clean/output/la_fix011_hidden_domain_language` |
| Log | `/home/kangjie.xu/oellm_clean/output/la_fix011_hidden_domain_language/compile.jobs16.log` |
| Profile | chunk 1024, cache 2048, PBD q=6, AR q=1, W4, 4 cores, jobs 16 |
| State | HBM linked successfully at 2026-07-18 16:58 CST |
| Calibration | none; `calib_json_path` is not consumed by the independent Language API |
| Classification | compiler-structure control only; not a release candidate |

Artifacts:

| File | Bytes | SHA256 |
|---|---:|---|
| Language HBM | 1,825,443,280 | `6e16fffc943167fb9dab6d4c4c5e8921c4f1bd49dc6bb3c54109c427ae5716ce` |
| Embedding table | 625,381,376 | `8668944fcb527faf3bbcd1c03a88d9da69f400b0700028f51ac6abe700e04011` |

Preflight evidence:

- built-in rotation equals the Qwen Fix #010 reference matrix exactly;
- FP32 Language logits cosine `0.999999999986`;
- FP32 Language KV max difference `6.109476e-05`;
- `prefill`, `decode`, and `decode_ar` BC export passed;
- `decode_ar` input/output are `(1,1,2048)` and `(1,1,152681)`.

Calibration audit (2026-07-18):

- the active log contains no calibration stage or sample count;
- current fake-quant and RMSNorm observers were not populated by task data;
- HBM completion, if reached, remains useful only for the uncalibrated control
  in the next single-variable comparison.

Inspect the completed control artifact without modifying it:

```bash
cd /home/kangjie.xu/oellm_clean/output/la_fix011_hidden_domain_language
sha256sum LocateAnything-3B_language_*.hbm LocateAnything-3B_embed_tokens.bin
tail -n 20 compile.jobs16.log
```

## Fix #011 Vision

The Vision BC export and hidden-domain equivalence test passed. A release
Vision HBM build is paused until the task-specific calibration path in #029 is
implemented and scale-audited.
