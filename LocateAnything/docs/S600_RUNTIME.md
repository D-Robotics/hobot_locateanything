# S600 Runtime and Synchronization

## Hosts

| Role | SSH |
|---|---|
| Compiler | `kangjie.xu@10.112.20.45` |
| S600 | `sunrise@10.112.133.20` |

GitHub `main` is the source-code authority. Compiler outputs and board runtime
state stay under the ignored `workspace/` tree and are never synchronized by a
source update.

## Release Status

The current source contract defines a 13-graph fused Language candidate.
`workspace/builds/` is a compiler-candidate area, not a deployment source. The
previously validated board package has only `prefill`, `decode`, and
`decode_ar`; it remains the historical three-graph baseline. A 13-graph HBM is
promoted to `workspace/artifacts/release/` only after graph-catalog, numerical,
and S600 semantic checks succeed.

## Synchronize Code

Update the compiler-host checkout from GitHub first:

```bash
# Run on the 4090 compiler host.
cd ~/oe_locateanything/LocateAnything
git fetch origin main
git merge --ff-only origin/main
```

The S600 deployment directory is not required to be a Git checkout. Source,
runtime configuration, and tokenizer files are transferred from this verified
4090 checkout by the deployment command below. Generated HBM files, build
directories, logs, and board evidence are kept outside the synchronized source
payload and are never removed by a code update.

## Transfer Artifacts

Use the checked-in deployment program from the 4090 compiler host only after a
candidate has been promoted to `workspace/artifacts/release/`. It packages
the runtime source and tokenizer, rewrites a copy of the runtime JSON to the
new versioned directory, and records the byte count and SHA256 of every
transferred payload:

```bash
cd ~/oe_locateanything/LocateAnything
./deploy/deploy_locateanything_s600.sh \
  --release la-s600-w8-4096-fused \
  --vision-hbm workspace/artifacts/release/la-s600-w8-4096-fused/LocateAnything-3B_vision.hbm \
  --language-hbm workspace/artifacts/release/la-s600-w8-4096-fused/LocateAnything-3B_language.hbm \
  --embed-bin workspace/artifacts/release/la-s600-w8-4096-fused/LocateAnything-3B_embed_tokens.bin \
  --runtime-config deploy/runtime_config.json \
  --deploy-dir deploy \
  --tokenizer-dir deploy/tokenizer \
  --ssh-target sunrise@10.112.133.20 \
  --dest-root /home/sunrise/locateanything_deployments \
  --dry-run
```

Review all six paths and hashes printed by the dry run, then repeat the same
command with `--execute` in place of `--dry-run`. The program uses an
`.incoming-*` directory until all board-side byte counts and SHA256 values
match. Only then is that directory renamed to the requested release name.

Both an existing final directory and an existing staging directory are hard
failures. The script never resumes a partial copy and never deletes a remote
directory. After an interruption, inspect the staging directory and its
manifests manually; choose a new release name or remove the rejected staging
directory only after establishing that it contains no evidence worth keeping.

The published layout is:

```text
la-s600-w8-4096-fused/
  artifacts/LocateAnything-3B_vision.hbm
  artifacts/LocateAnything-3B_language.hbm
  artifacts/LocateAnything-3B_embed_tokens.bin
  config/locateanything_3b_config.json
  deploy/                         # extracted checked-in runtime source
  tokenizer/                       # extracted tokenizer files
  bundles/runtime-source.tar
  bundles/tokenizer.tar
  DEPLOY_MANIFEST.sha256
  DEPLOY_MANIFEST.bytes
  RELEASE_INFO.txt
```

Keep the three manifest/evidence files unchanged beside all semantic and
benchmark evidence. Never overwrite the Qwen baseline or an older LA artifact
set in place. The deployment script proves transfer integrity only; it does
not prove HBM numerical or grounding correctness.

## Runtime Contract

- Vision 672x672 emits 576 embeddings; the host prompt must contain exactly
  576 image placeholders.
- Source images are letterboxed with the recorded scale/padding metadata;
  grounding coordinates are mapped back to the original image after decoding.
- Embedding lookup uses vocab 152681 and hidden size 2048.
- Vision uses W8 weights. Language decoder and `lm_head` use W8/W8.
- Prefill uses chunk 1024 and cache 4096; base PBD uses q=6 and base AR uses
  q=1.
- A promoted fused release must expose all 13 graphs: `prefill`, `decode`,
  `decode_ar`, `decode_pbd_q7..q12`, and `decode_ar_q2..q5`. The historical
  three-graph package remains valid only for legacy-baseline runs.
- PBD uses LA's diagonal-block mask and shifted position IDs. Fused profiles
  reduce repeated KV submission without changing the q=6 PBD policy.
- The SDK `vlm` binary runs the Qwen2.5-VL reference path. LocateAnything's
  tokenizer, PBD, and box-generation flow are implemented under `deploy/`.

For the current chunk-1024 profile, provide the required L2M allocation in the
same shell:

```bash
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
```

## Interactive CLI

Build the two resident HBM runners and install the user-level command:

```bash
cd /home/sunrise/oe_locateanything/LocateAnything
cmake --build deploy/build \
  --target vision_hbm_runner language_hbm_runner \
  -j4

sh deploy/install_locateanything_cli.sh
export PATH="$HOME/.local/bin:$PATH"
```

Start the persistent interactive runtime:

```bash
LocateAnything \
  -i workspace/samples/check_cat2.jpg \
  --max-new-tokens 2048
```

The Vision and Language HBM models load once and remain resident until `exit`.
Hybrid generation (q=6 PBD with q=1 fallback) is the default. Use
`--generation-mode slow` only when an explicit q=1 AR control run is needed.
Use `/image PATH`, `regen`, and `reset` inside the session. The runtime prints
measured Vision, Prefill, Decode, and end-to-end latency after each request.

## Evidence Levels

- HBM load success: ABI/runtime compatibility only.
- Nonzero logits/KV: graph execution only.
- PyTorch cosine/logit agreement: numerical validation.
- Correct `<ref>/<box>` response on S600: semantic validation.
- Dataset metrics at IoU 0.90, with the frozen evaluation manifest: deployment
  completion.

## Repeatable Benchmark Evidence

After semantic validation succeeds, use
[`deploy/run_s600_benchmark.sh`](../deploy/run_s600_benchmark.sh)
for warm-up plus repeated measured runs. It retains raw logs, run-level JSONL,
resource time series, artifact checksums, and a JSON aggregate. A semantic
regular expression must be supplied if semantic success rate is to be
reported; an exit code of zero alone is only process success.

The metric definitions and board invocation are documented in
[`BENCHMARKING.md`](BENCHMARKING.md).
In particular, runtime TPS is direct evidence, while phase durations computed
from token count/TPS and structured boxes per second are explicitly marked
`derived`. Missing BPU, temperature, or power sources remain `unavailable`.
On S600, `/usr/hobot/bin/hrut_somstatus` is sampled automatically when it is
present and executable; its temperature, voltage, and four BPU ratio fields are
parsed without converting command failures into zero readings.
