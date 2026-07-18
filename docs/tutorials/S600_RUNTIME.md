# S600 Runtime and Synchronization

## Hosts

| Role | SSH |
|---|---|
| Compiler | `kangjie.xu@10.112.20.45` |
| S600 | `sunrise@10.112.133.20` |

The Git publication source is the compiler-side repository. The S600 checkout
intentionally omits large model/tokenizer files and contains generated runtime
build files; those deletions must not be committed.

## Synchronize Code

```bash
cd ~/oe_locateanything
git fetch origin main
git merge --ff-only origin/main
```

If the board has local generated files, preserve them before the fast-forward
and restore only board-local artifacts afterwards.

## Transfer Artifacts

Record checksums on the compiler host:

```bash
sha256sum LocateAnything-3B_*.hbm LocateAnything-3B_embed_tokens.bin
```

Transfer to a versioned directory, then verify the same checksums on S600.
Never overwrite the Qwen baseline or an older LA artifact set in place.

## Runtime Contract

- Vision 448x448 emits 256 embeddings; the host prompt must contain exactly
  256 image placeholders.
- Embedding lookup uses vocab 152681 and hidden size 2048.
- Prefill uses chunk 1024 and cache 2048.
- PBD decode uses six inputs with LA's diagonal-block mask and shifted
  position IDs.
- The SDK `vlm` binary runs the Qwen2.5-VL reference path. LocateAnything's
  tokenizer, PBD, and box-generation flow are implemented under `main/runtime`.

For the current chunk-1024 profile, provide the required L2M allocation in the
same shell:

```bash
export HB_DNN_USER_DEFINED_L2M_SIZES=8:8:8:8
```

## Evidence Levels

- HBM load success: ABI/runtime compatibility only.
- Nonzero logits/KV: graph execution only.
- PyTorch cosine/logit agreement: numerical validation.
- Correct `<ref>/<box>` response on S600: semantic validation.
- Dataset metrics: deployment completion.
