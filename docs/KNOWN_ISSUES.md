# Known Issues & Resolutions

Chronological log of deployment issues encountered while shipping LocateAnything-3B on the D-Robotics S600, together with root causes and fixes. Every non-trivial diagnostic session lands here so the next engineer can spot the same trap in seconds.

Format for each entry:

- **Symptom** — one-line surface of what broke.
- **Trigger** — what config / command / environment reproduces it.
- **Root cause** — the underlying mechanism, verified.
- **Evidence** — log lines, commands, versions.
- **Fix** — what we changed, where.
- **Alternatives considered** — what we did not do, and why.
- **Prevention** — how to catch this before it bites again.

---

## #001 README Chinese text corrupted via SSH heredoc (2026-07-08)

**Symptom**: `README.md` Chinese sections rendered as `鍚屾椂鍦?` etc. after editing over SSH.

**Trigger**: Editing UTF-8 content through `bash -c "cat > file <<'EOF' ... EOF"` from a session whose shell codepage did not agree with the terminal encoding.

**Root cause**: Double decoding. GBK bytes on the sending side were stored as UTF-8 on disk, then re-interpreted as UTF-8 on read. The transformation was not reversible because the intermediate stage lost information.

**Evidence**: `file README.md` reported `Unicode text, UTF-8 (with BOM)`, but Python `read_text(encoding="utf-8")` returned mojibake strings. Recovery attempts with `.encode("utf-8").decode("gbk")` and `.encode("latin1").decode("utf-8")` all failed.

**Fix**: Rewrote the affected Chinese paragraphs locally on Windows with the `Write` tool (guaranteed UTF-8 without BOM), then transferred with `scp` (binary transport, no shell decoding). Verified with a Python `sum(c in mojibake_chars for c in text) == 0` check on 4090.

**Alternatives considered**: In-place `sed` or `python3 -c` patches on 4090 — rejected because they still went through the same broken SSH pipeline for the replacement string.

**Prevention**: Never author non-ASCII content through heredoc over SSH. Author locally, `scp` to server, verify with a mojibake check before committing.

---

## #002 GitHub push over HTTPS 443 times out (2026-07-08)

**Symptom**: `git push origin main` on Windows repeatedly failed with `Failed to connect to github.com port 443`.

**Trigger**: Any push from either the 4090 or the Windows workstation that used the default HTTPS remote.

**Root cause**: Network policy blocking outbound TCP 443 to github.com from these hosts. SSH port 22 to github.com was not blocked.

**Evidence**: `curl -sI --max-time 8 https://github.com` returned `000`; `ssh -o BatchMode=yes git@github.com` returned a normal `Host key verification failed` (i.e. reached the server).

**Fix**: Switched the Windows clone's `origin` remote from `https://github.com/...` to `git@github.com:...`. First-attempt SSH push succeeded.

**Alternatives considered**: Retry HTTPS with a longer timeout — tried, still failed at 21s. HTTP proxy — no proxy available in the environment.

**Prevention**: Windows repo `.git/config` is now on the SSH remote permanently. Documented in the deployment runbook.

---

## #003 `hf download` fails with xet CAS 401 (2026-07-08)

**Symptom**: `hf download nvidia/LocateAnything-3B --local-dir ...` failed with `File reconstruction error: CAS Client Error: ... 401 Unauthorized`.

**Trigger**: `hf-hub >= 1.22.0` combined with `HF_ENDPOINT=https://hf-mirror.com`.

**Root cause**: `hf-hub 1.22` defaults to the new xet CAS protocol, which routes reconstruction requests to `cas-server.xethub.hf.co` directly. `hf-mirror.com` does not proxy xet CAS, so those requests hit the real HuggingFace CAS without auth and return 401.

**Evidence**: Full stack trace ended at `xet_get` calling `https://cas-server.xethub.hf.co/v2/reconstructions/...`. `pip show hf-xet` confirmed 1.5.1 was installed.

**Fix**: Downgraded `hf-hub` to `<0.30` (`pip install "huggingface_hub<0.30" hf_transfer`), which does not use xet and honours `HF_ENDPOINT` for the whole download flow.

**Alternatives considered**: `HF_HUB_DISABLE_XET=1` env var — worked, but still slow (~4 MB/s from hf-mirror because `hf_transfer` was deprecated in 1.22). Downgrade unlocks `hf_transfer` acceleration too.

**Prevention**: Pin `huggingface_hub<0.30` in the LocateAnything conda env; document in `docs/DEPLOYMENT.md` if we ever revise the download flow.

---

## #004 `pip install torch==2.8.0 --index-url cu121` cannot find distribution (2026-07-08)

**Symptom**: `pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu121` returned `Could not find a version that satisfies the requirement torch==2.8.0`.

**Trigger**: Trying to match torch 2.8.0 with a CUDA 12.1 wheel because the 4090 host runs driver 535 / CUDA 12.2.

**Root cause**: The `whl/cu121` index only ships torch up to 2.5.1. torch 2.6+ wheels only exist for cu124/cu126, which require driver ≥ 545.

**Evidence**: pip error listed available `cu121` versions: `2.1.0+cu121 ... 2.5.1+cu121`.

**Fix**: Installed `torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`. The `locateanything` and `oellm` conda envs are independent, so torch version divergence between them is fine.

**Alternatives considered**: Upgrade host driver to 545 — rejected because it requires sudo and could affect other users on the shared 4090.

**Prevention**: When picking torch/CUDA on a shared host, always check `nvidia-smi` driver → CUDA cap first. Pin torch in the conda env spec once decided.

---

## #005 Baseline OOM at 9.61 GB single attention block (2026-07-08)

**Symptom**: `demo_min.py` crashed with `CUDA out of memory. Tried to allocate 9.61 GiB` on a 24 GB RTX 4090 during MoonViT `sdpa_attention`.

**Trigger**: Feeding the original 1920×1280 `test-cat.jpg` directly through `LocateAnythingWorker.detect(...)` with no image resize.

**Root cause**: MoonViT is native-resolution. A 1920×1280 image at patch 14 produces ~12,500 tokens; a full-attention matrix `12500 × 12500` in fp16 is ~300 MB per head, ~4.8 GB per layer, and multi-layer intermediate storage blows past 9 GB.

**Evidence**: Traceback rooted in `modeling_vit.py:150 sdpa_attention` with `F.scaled_dot_product_attention`.

**Fix**: Added a downscale step in `main/examples/demo_min.py` that resizes any input whose long side exceeds 1024 px (`MAX_LONG_SIDE = 1024`), reducing token count to ~4000 and memory to a few hundred MB.

**Alternatives considered**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — masks the fragmentation but does not remove the O(N²) attention. Streaming attention — would require patching MoonViT. Both rejected as heavier than a bounded resize.

**Prevention**: Any script that feeds MoonViT native-res input must state its assumed max resolution up front. `MAX_LONG_SIDE` is a module-level constant to keep this discoverable.

---

## #006 `oellm_build` errors on unknown `vit_kwargs` keyword (2026-07-09)

**Symptom**: First launch of `oellm_build --model_name locateanything-lm-3b` died with `TypeError: LocateAnythingLanguageOnlyApi.compile() got an unexpected keyword argument 'vit_kwargs'`.

**Trigger**: Any custom Api that omits the `vit_kwargs` parameter in its `compile()` signature.

**Root cause**: `leap_llm/apis/oellm_build.py::main` invokes `model.compile(vit_kwargs=..., llm_kwargs=...)` uniformly for every model, whether or not the model produces a vision HBM.

**Evidence**: Traceback pointed at `oellm_build.py:665`.

**Fix**: Every custom Api accepts `vit_kwargs` in `compile()`, even when it never produces a vision HBM — just ignores it. Both `LocateAnythingLanguageApi` and `LocateAnythingApi` follow this convention.

**Alternatives considered**: Patch `oellm_build.py` to introspect the signature — rejected because it modifies wheel code needlessly.

**Prevention**: Api template docstring records the required `compile(self, vit_kwargs=None, llm_kwargs=None)` signature.

---

## #007 `leap_export` fails with `AttributeError: 'list' object has no attribute 'type'` (2026-07-09)

**Symptom**: Compile crashed at `hbdk4/compiler/leap.py:278` on `return_types = [v.type for v in results]`.

**Trigger**: A leap DSL `build()` method returning a tuple whose elements include intermediate Python `list`s.

**Root cause**: `leap.leap_export` expects the traced function to return a flat sequence of leaf tensors. Nested containers are not decomposed automatically.

**Evidence**: Our `LocateAnythingTextModel.build()` returned `(logits, new_keys, new_values)` where `new_keys` / `new_values` were Python lists of per-layer tensors. Upstream `Qwen2_5_VLTextModel.build()` uses `return token_logits, *new_keys, *new_values` — same shape flattened.

**Fix**: Changed `return logits, new_keys, new_values` → `return (logits, *new_keys, *new_values)`. PyTorch `forward()` (used only for calibration) still returns the tuple-of-lists form because torch does not care.

**Alternatives considered**: Wrapping every layer's KV in a `torch.stack` — rejected because leap.TensorType input signatures declare 2×num_layers separate tensors.

**Prevention**: Every leap DSL `build()` must return only `leap.Tensor` leaves. When in doubt, `[isinstance(v, leap.Tensor) for v in results]` should be all True.

---

## #008 LocateAnything vocab 152681 crashes decode compile_hbo (2026-07-09)

**Symptom**: `oellm_build --model_name locateanything-lm-3b` completed prefill (`prefill.hbo` 1.6 GB produced) but the python process died silently in the decode compile_hbo stage. No traceback in the log; last line was an `[info]` warning.

**Trigger**: `LocateAnythingLanguageApi.compile()` passing `input_no_padding=True, output_no_padding=True` in the `compile_hbo` kwargs, combined with `vocab_size = 152681`.

**Root cause**: BPU DMA reads outputs in 64-byte aligned chunks. hbdk4's `output_no_padding=True` promises "do not pad the last dim". LocateAnything's lm_head output last dim is `vocab_size × sizeof(fp16) = 152681 × 2 = 305362` bytes, `305362 % 64 = 50 ≠ 0`. hbdk4 emits `[info] output_no_padding=true will not be applied ... get 305362`, then hits an internal path bug on the decode stage that aborts python without a traceback. The prefill stage happened to complete before the abort because it exercises a different code path in hbdk4. The Qwen2.5-VL baseline uses vocab 151936, and `151936 × 2 = 303872` is divisible by 64, which is why the same `no_padding=True` kwargs work there.

**Evidence**: `log tail`:
```
[2026-07-09 15:26:24.649] [info] This configuration `output_no_padding=true` will not be applied.
When the product of the C dimension and the element size exceeds 16384, the product must be
divisible by 64, but get 305362.
```
`ps -p <pid>` = DEAD. No OOM in dmesg. No traceback in log. Produced `prefill.hbo` (1.6 GB) but not `decode.hbo` or `.hbm`. `Function 'compile_hbo' done` line present for prefill (4180 s), missing for decode.

**Fix**: Removed `input_no_padding` and `output_no_padding` from the compile_hbo kwargs in `toolchain/leap_llm/apis/model/locateanything_language.py`. hbdk4 now falls back to its default automatic padding (pads the last dim from 305362 to 305408 bytes internally). Host runtime later slices `logits[..., :152681]` to drop the 46-byte pad.

**Alternatives considered**:
- **Pad vocab_size to 152704** (23 dummy embeddings, zero weight): also works but requires touching config + model + Api + host sampling; heavier than a two-line kwargs change. Rejected on the principle "let the compiler do its job".
- **Per-stage kwargs (keep no_padding=True for prefill, drop for decode)**: adds branching in the Api that we would then have to maintain. Rejected on grounds of minimality.
- **`--w_bits 8` for the lm_head**: lm_head byte count would become 152681 × 1 = 152681, still `% 64 = 25 ≠ 0`. Does not fix the issue.

**Prevention**: When picking `no_padding` flags, always check `vocab_size × sizeof(dtype) % 64 == 0` for the lm_head output. Add this as a compile-time assertion in `LocateAnythingLanguageApi.__init__` (TBD).

---

## #009 Illegal b30 fusion / A 方案闭环记录 (2026-07-09)

**Symptom**: `oellm_build --model_name locateanything-lm-3b` 编译过程中打印 36 条 `[B30 Fusion MultiCore Legalize]: Illegal b30 fusion operator detected`，每层 self-attention 的 `wv_matmul` 命中一次。曾一度怀疑这是 decode 阶段静默 die 的根因。

**Trigger**: Qwen2.5-3B decoder 的 36 层 self-attention，每层 `wv_matmul` 输出 `[1, 2, 2048, 128]`（GQA: 16 Q-heads / 2 KV-heads / head_dim 128）进入 `b30fusion.scaled_dot_product_attention` / `b30fusion.group_query_attention` 融合初态。

**Root cause**: 良性——`B30FusionMultiCoreLegalizePass` 的诊断日志，非编译错误。

底层机制：
- B30 fusion 假设 Q/K/V 三分支 rank 4 对齐、num_heads 维度长度一致、heads axis=1、KV 广播显式化。
- GQA 破坏 "num_heads 一致"（Q 16, KV 2），wv 分支进入 PV-GEMM (`softmax(QK^T) @ V`) 融合入口时需要沿 head 轴复制 8 次到 16，融合初态不合约束。
- Pass 检测出 → 打印 "Illegal detected" → **同 pass 内部**调用 `LegalizeRankForB30VpuFusion` / `MergeAxesForB30VpuFusion` / `SplitB30VpuFusion` / `UnFoldFusion` 把 IR 改造成合法融合形态。
- 命名带 "Illegal" 是 MLIR `Legalize*Pass` 的惯例（"合法化前的 findings marker"），不是 bug。

只 `wv_matmul` 触发的原因：wv 是融合区域的输出端消费者，pass 挑选它作为 illegal report 的 layerName 锚点；wk/wq 各自进入的融合前半段（QK^T + softmax）rank/axis 天然对齐，不需要 legalize。

**Evidence**:
1. Per-layer 等价性：LA-LM 与 baseline qwen2_5-vl-3b 各 36 条 illegal，layerName 逐字符串等价（`layers.0~35.self_attn.wv_matmul`，同源 `matmul.py:27:19`），wk/wq 计数均为 0。
2. baseline 同样 36 条 illegal 后 `Function compile_hbo done in 4077.7s` 并产出 hbm，error/failed 计数=0。
3. `libHBDKPythonCAPI.so` strings 里 `B30FusionMultiCoreLegalizePass` 与 `LegalizeRankForB30VpuFusion`, `SplitB30VpuFusion`, `MergeAxesForB30VpuFusion`, `UnFoldFusion` 共存——detect + rewrite 两条路径在同一 pass。
4. `b30fusion.group_query_attention` op 名存在于二进制字符串表 → B30 硬件原生支持 GQA 融合。

**Fix**: 不改代码。将 illegal 日志归类为 diagnostic-only。

**Alternatives considered**:
- 把 Qwen2.5-3B 的 GQA (16/2) 改成 MHA (16/16)：破坏预训练权重的 KV 投影矩阵形状，加载失败；即使 pad KV 权重也会破坏精度。
- 在 leap DSL 里手动把 wv 输出 `.repeat(8, dim=1)` 扩到 16 heads：可行但工程冗余，且 hbdk4 pass 会把重复的 legalize 再做一遍。
- 关闭 hbdk4 pass 的 illegal 日志：需要改 wheel 源码，收益低于噪声成本。

**Prevention**: `docs/KNOWN_ISSUES.md` 记录 (此条)。下次遇到 `Illegal b30 fusion` 直接按本条判定良性，跳过重新排查；同时把注意力放在 log tail 的其他 warning（如 `output_no_padding=true will not be applied` #008）上。

---

## #010 A 方案修复 vocab 152681 编译闭环 (2026-07-09)

**Symptom**: 上一轮 M2 编译在 prefill.hbo 产出后 python 静默 die 于 decode.compile_hbo（详见 #008）。移除 `input_no_padding=True, output_no_padding=True` 两个 kwargs 后重编。

**Trigger**: 同 #008。

**Root cause**: 同 #008。

**Fix (verified)**: 移除后重编，A 方案生效：
- `prefill.compile_hbo` done in **4379.6 s** (1h13m)
- `decode.compile_hbo` done in **3952.2 s** (1h6m)  ← **上一轮 die 的阶段，本轮通过**
- `link_models` done in **57.1 s**
- 总 wall-clock ≈ **2h20m** (compile_hbo + link 部分)
- 最终产物：`LocateAnything-3B_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm` **1.6 GB** 落盘
- error/failed 计数 = **0**
- hbdk4 内部对 vocab 152681 × fp16 = 305362 bytes 的 last-dim 自动 pad 到 305408（≥ 64 对齐），host runtime 后续切 `logits[..., :152681]` 即可

**Alternatives considered**: 同 #008 的 B/C/D。A 方案两行改动 + 无精度影响 + 无侵入模型，成本最低。

**Prevention**: `LocateAnythingLanguageApi` 保持不传 no_padding kwargs；`LocateAnythingApi` (M4 unified) 亦沿用此约定；如未来 vocab 或 dtype 改变，重新校验 `vocab_size × sizeof(dtype) % 64 == 0`。

