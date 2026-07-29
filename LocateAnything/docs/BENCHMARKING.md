# S600 Runtime Benchmarking

`deploy/run_s600_benchmark.sh` repeatedly runs the real LocateAnything CLI and
writes timing, semantic, CPU, memory, BPU, temperature, and voltage evidence.
The default generation mode is `hybrid`.

## Fixed-case benchmark

Run from the LocateAnything product directory on S600:

```bash
deploy/run_s600_benchmark.sh \
  --image tests/fixtures/test-cat.jpg \
  --prompt "/detect cat" \
  --generation-mode hybrid \
  --max-new-tokens 2048 \
  --runs 10 \
  --warmup 1 \
  --output-dir workspace/benchmarks/cat-hybrid-q6 \
  --semantic-regex '<ref>cat</ref>.*<box><[0-9]+><[0-9]+><[0-9]+><[0-9]+></box>'
```

Use a new output directory when the image, prompt, HBM, embedding, generation
mode, or runtime changes. The runner refuses to overwrite existing evidence.

An additional documented numeric BPU node can be sampled with a repeatable
argument:

```bash
deploy/run_s600_benchmark.sh \
  --image tests/fixtures/test-cat.jpg \
  --bpu-metric core0=/documented/path/to/core0/load \
  --bpu-metric core1=/documented/path/to/core1/load
```

Do not label temperature or power nodes as BPU utilization. The collector also
uses `/usr/hobot/bin/hrut_somstatus` when the command exists and records missing
or unparsable values as unavailable rather than zero.

## Evidence

| File | Contents |
|---|---|
| `summary.json` | Protocol, platform, success rates, and aggregate timing/resource values |
| `runs.jsonl` | Per-run exit state, semantic result, timing, and resource peaks |
| `resource_samples.jsonl` | CPU, RSS, Linux sensors, BPU nodes, and vendor status time series |
| `logs/*.log` | Merged stdout/stderr for each warm-up and measured run |

A LocateAnything run is process-successful only when it exits with code zero
and prints both `[Assistant] >>>` and `STATUS: COMPLETE`. When
`--semantic-regex` is supplied, aggregate timing includes only runs that also
match the requested result pattern.

The CLI `[perf]` fields are direct measurements. Values calculated from token
counts and throughput are marked derived. BPS is the number of validated box
frames per runtime-reported end-to-end second; it is not memory bandwidth.
