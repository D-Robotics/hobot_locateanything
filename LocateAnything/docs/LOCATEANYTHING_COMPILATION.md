# LocateAnything-3B Compilation for S600

## Current Target

- `march=nash-p`
- fixed image profile: 672x672 letterbox
- Vision input/output: `(1,2304,588)` -> `(1,576,2048)`
- Language prefill: chunk 1024, cache 4096
- Language decode: PBD query length 6
- Language decoder/lm_head weights: W8; Vision weights: W8
- four BPU cores; `jobs=16`

The compiler uses a reproducible 2048-dimensional signed Hadamard transform.
It is folded offline into both the Qwen2.5 decoder and MoonViT projector.

## 1. Install the Compiler Adapter

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean
cd ~/oe_locateanything/LocateAnything/compiler
pip install -e . --no-deps
```

## 2. Prepare and Validate the Release Contract

```bash
cd ~/oe_locateanything/LocateAnything
python compiler/quantize.py prepare --preflight-only
python compiler/quantize.py prepare
python compiler/quantize.py calibrate
python compiler/quantize.py verify --component all --level contract
```

The first command checks the frozen 1,200-record manifest, image hashes,
checkpoint index and shard hashes, tokenizer, processor, runtime-tokenizer
parity, and sequence-length contracts without loading the 3B weights or using
CUDA. The regular `prepare` command runs the same preflight first, then creates
calibration tensors with `max_new_tokens=1024`. Contract verification checks the
selected/generated manifests, frozen scale files, graph coverage, and the fixed
672/1024/4096 build profile. It does not re-run the separate hidden-domain
numerical experiment. The internal rotation validator previously produced the
following 4090 reference values:

```text
language logits cosine = 0.999999999986
language KV max diff   = 6.109476e-05
vision output cosine   = 0.999999927
```

## 3. Export BC Before Long Compilation

```bash
python compiler/quantize.py build --component all --target bc
```

Expected BC contracts:

| Graph | Inputs | Primary output |
|---|---:|---|
| prefill | 75 | `(1,1,152681)` last-row logits + 72 q=1024 KV updates |
| decode | 75 | `(1,6,152681)` logits + 72 KV |
| decode_ar | 75 | `(1,1,152681)` logits + 72 KV |
| decode_pbd_q7...q12 | 75 | fused-PBD logits + 72 KV |
| decode_ar_q2...q5 | 75 | causal bridge logits + 72 KV |
| visual | 1 | `(1,576,2048)` visual embeddings |

The 13 Language graphs follow the upstream Hybrid state machine, not a generic
`q=1...12` sweep. `prefill`, `decode` (PBD q=6), and `decode_ar` (AR q=1) are
the three base graphs. In upstream generation, a PBD result is appended to the
generated sequence before its K/V rows exist in the cache. If a legal pattern
accepts `N` tokens, the next MTP call receives those `N` causal prefix tokens
followed by the duplicated anchor and five mask tokens; the fixed graph query
length is therefore `N+6`. Legal patterns can retain 1 through 6 tokens, which
requires `decode_pbd_q7...q12`. An `error_box` instead switches to AR after a
1-through-5-token coordinate prefix, so the causal bridge family is q=1...5;
q=1 is `decode_ar`, while `decode_ar_q2...q5` supply the remaining four shapes.
There is no AR q=6 branch: a complete six-token box is legal and remains in
PBD. Calibration consequently executes PBD q=6...12 and AR q=1...5, while the
compiled catalog represents the q=6 and q=1 cases with their base graph names.

## 4. Compile HBM

```bash
cd ~/oe_locateanything/LocateAnything
python compiler/quantize.py build --component all --target hbm --resume
```

The orchestrator builds Vision and Language sequentially and writes stage logs
under `workspace/logs/`. `--resume` reuses completed artifacts. Do not launch a
second HBDK build against the same output directory.

Reuse is identity-based rather than filename-based. The BC manifest binds the
checkpoint metadata, frozen Scale manifest, hidden-domain configuration,
compiler source digest, toolchain versions, and each BC SHA256. Converted BC,
HBO, and HBM files have SHA256 sidecars. When an input or compile contract
changes, the same command invalidates downstream reuse and rebuilds from the
nearest trusted stage. A reusable Language HBM must expose exactly the 13
release graphs with the declared shapes; a reusable Vision HBM must expose only
`visual` with the 672 profile.

## 5. Required Validation Inputs

An HBM build does not produce cross-stage comparison outputs or held-out board
predictions. Before using the aggregate verifier, prepare all of the following:

1. completed calibration manifests, Scale manifest, and graph coverage for
   contract validation;
2. one coherent pipeline directory containing `inputs.json`, a completed
   `float/stage.json`, and at least one completed candidate stage with its
   per-sample outputs; every stage must use the same phase and input set;
3. the configured held-out reference JSONL and the matching S600 predictions
   JSONL for task evaluation.

Once those artifacts exist, run:

```bash
python compiler/quantize.py verify --component all --level all
```

This command validates and summarizes existing evidence. It does not execute
the Float, Quantized-Eager, BC, HBM, or S600 collection stages. Pipeline
analysis rejects a missing Float stage, no completed candidate, mixed phases,
and mismatched input fingerprints. It can intentionally report a partial set
of candidate stages, so a release decision must also confirm that every
planned stage appears in the report. Run
`compiler/scripts/validate/hbm_sanity.py` separately to inspect the HBM graph
catalog, descriptors, and embedding file.

## 6. Required Validation Order

The required validation order is:

1. Confirm HBM graph names, shapes, dtype, file size, and SHA256.
2. Compare Vision HBM output against rotated PyTorch Vision on the same input.
3. Compare Language prefill/decode logits and KV against rotated PyTorch.
4. Transfer one artifact set to S600 with checksums.
5. Run fixed-resolution image-token insertion with exactly 576 visual tokens.
6. Validate AR first, then PBD q=6, Hybrid fallback, and box parsing.

Nonzero logits establish graph execution. Numerical comparison and grounding
output establish model correctness at the following validation levels.
