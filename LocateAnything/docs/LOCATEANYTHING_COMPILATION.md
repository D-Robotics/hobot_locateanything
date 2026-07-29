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
python compiler/quantize.py prepare
python compiler/quantize.py calibrate
python compiler/quantize.py verify --component all --level contract
```

Contract verification checks the selected/generated manifests, frozen scale
files, graph coverage, and the fixed 672/1024/4096 build profile. It does not
re-run the separate hidden-domain numerical experiment. The internal rotation
validator previously produced the following 4090 reference values:

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
| prefill | 75 | `(1,1024,152681)` logits + 72 KV |
| decode | 75 | `(1,6,152681)` logits + 72 KV |
| decode_ar | 75 | `(1,1,152681)` logits + 72 KV |
| decode_pbd_q7...q12 | 75 | fused-PBD logits + 72 KV |
| decode_ar_q2...q5 | 75 | causal bridge logits + 72 KV |
| visual | 1 | `(1,576,2048)` visual embeddings |

## 4. Compile HBM

```bash
cd ~/oe_locateanything/LocateAnything
python compiler/quantize.py build --component all --target hbm --resume
```

The orchestrator builds Vision and Language sequentially and writes stage logs
under `workspace/logs/`. `--resume` reuses completed artifacts. Do not launch a
second HBDK build against the same output directory.

## 5. Required Validation Order

Run the unified verifier after the build:

```bash
python compiler/quantize.py verify --component all --level all
```

The required validation order is:

1. Confirm HBM graph names, shapes, dtype, file size, and SHA256.
2. Compare Vision HBM output against rotated PyTorch Vision on the same input.
3. Compare Language prefill/decode logits and KV against rotated PyTorch.
4. Transfer one artifact set to S600 with checksums.
5. Run fixed-resolution image-token insertion with exactly 576 visual tokens.
6. Validate AR first, then PBD q=6, Hybrid fallback, and box parsing.

Nonzero logits establish graph execution. Numerical comparison and grounding
output establish model correctness at the following validation levels.
