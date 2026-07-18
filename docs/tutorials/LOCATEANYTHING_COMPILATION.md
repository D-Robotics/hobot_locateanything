# LocateAnything-3B Compilation for S600

## Current Target

- `march=nash-p`
- fixed image profile: 448x448
- Vision input/output: `(1,1024,588)` -> `(1,256,2048)`
- Language prefill: chunk 1024, cache 2048
- Language decode: PBD query length 6
- Language weights: W4; Vision weights: W8
- four BPU cores; `jobs=16`

The compiler uses a reproducible 2048-dimensional signed Hadamard transform.
It is folded offline into both the Qwen2.5 decoder and MoonViT projector.

## 1. Install the Vendored Toolchain

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oellm_clean
cd ~/oe_locateanything/toolchain
pip install -e . --no-deps
```

## 2. Validate the Hidden-Domain Rewrite

```bash
PYTHONPATH=~/oe_locateanything/toolchain \
python ~/oe_locateanything/main/scripts/validate_locateanything_rotation.py \
  --model-path ~/oe_locateanything/eagle/Embodied/LocateAnything-3B \
  --component all --device cuda:0 --dtype float32
```

Reference results from the 4090 host:

```text
language logits cosine = 0.999999999986
language KV max diff   = 6.109476e-05
vision output cosine   = 0.999999927
```

## 3. Export BC Before Long Compilation

Language:

```bash
PYTHONPATH=~/oe_locateanything/toolchain \
python -m leap_llm.apis.oellm_build \
  --model_name locateanything-lm-3b --march nash-p \
  --input_model_path ~/oe_locateanything/eagle/Embodied/LocateAnything-3B \
  --output_model_path ~/oellm_clean/output/la_export_language \
  --w_bits 4 --chunk_size 1024 --cache_len 2048 --decode_seq_len 6 \
  --device cuda:0 --prefill_core_num 4 --decode_core_num 4 \
  --jobs 16 --export_only
```

Vision uses `--model_name locateanything-vit-3b --w_bits 8 --vit_core_num 4
--image_width 448 --image_height 448 --export_only`.

Expected BC contracts:

| Graph | Inputs | Primary output |
|---|---:|---|
| prefill | 75 | `(1,1024,152681)` logits + 72 KV |
| decode | 75 | `(1,6,152681)` logits + 72 KV |
| decode_ar | 75 | `(1,1,152681)` logits + 72 KV |
| visual | 1 | `(1,256,2048)` visual embeddings |

## 4. Launch Detached HBM Compilation

```bash
cd ~/oe_locateanything
./main/scripts/compile_locateanything_language.sh
./main/scripts/compile_locateanything_vit.sh
```

Both scripts use `setsid + nohup + </dev/null`, print the process-group PID,
and write logs under `main/logs/`. Follow progress with:

```bash
tail -f ~/oe_locateanything/main/logs/locateanything_language_compile.log
tail -f ~/oe_locateanything/main/logs/locateanything_vit_compile.log
```

Before replacing a compile, record and stop the previous process group and its
HBDK children so the new run has an isolated resource profile.

## 5. Required Validation Order

1. Confirm HBM graph names, shapes, dtype, file size, and SHA256.
2. Compare Vision HBM output against rotated PyTorch Vision on the same input.
3. Compare Language prefill/decode logits and KV against rotated PyTorch.
4. Transfer one artifact set to S600 with checksums.
5. Run fixed-resolution image-token insertion with exactly 256 visual tokens.
6. Validate AR first, then PBD q=6, Hybrid fallback, and box parsing.

Nonzero logits establish graph execution. Numerical comparison and grounding
output establish model correctness at the following validation levels.
