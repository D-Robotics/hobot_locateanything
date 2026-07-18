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
| State | detached HBM compilation started; completion not yet claimed |
| Calibration | none; `calib_json_path` is not consumed by the independent Language API |
| Classification | compiler-structure control only; not a release candidate |

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

Monitor without attaching to the compiler process:

```bash
ps -o pid,ppid,sid,etime,pcpu,pmem,rss,args -p 2094764
tail -f /home/kangjie.xu/oellm_clean/output/la_fix011_hidden_domain_language/compile.jobs16.log
```

## Fix #011 Vision

The Vision BC export and hidden-domain equivalence test passed. Fresh Vision
HBM compilation is intentionally queued after Language to avoid resource
contention.
