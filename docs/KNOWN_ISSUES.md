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


---

## #011 4090 上 git push 走 HTTPS + PAT credential store (2026-07-09)

**Symptom**: 4090 上 `git push origin main` 长期挂 `Empty reply from server` (#002 反复) 或 `Permission denied (publickey)`。工作全靠 Windows 中转，4090 本地始终 ahead。ahead 一度累积到 14 commits。

**Trigger**: D-Robotics 内网 → github.com 走 `:443` 的 git-over-https 路径不稳定；SSH `:22` 出网被墙；4090 家目录里也没有配置过 GitHub SSH key。

**Root cause**: 两条通道各有一个短板：
- HTTPS 走 `git push` 时被中间盒截断 (`Empty reply`)，但走 `curl https://api.github.com/*` 与 `https://LiuAnclouds:<PAT>@github.com/.../.git` 的 push 是同一个 :443 端口却能通——说明中间盒对 "unauthenticated + Content-Length large" 的组合更容易 reset，带 PAT 后走匿名不同路径反而放行；
- SSH 22 出网被 D-Robotics 内网墙掉 + 4090 家目录里没有 `id_ed25519` 私钥文件，双重原因。

**Evidence**:
- `curl -H "Authorization: token $PAT" https://api.github.com/user` → HTTP 200 in 2.4s
- `git remote set-url origin https://LiuAnclouds:$PAT@github.com/.../.git; git push` → `8f207f0..917b5de main -> main` 一次过
- `ls ~/.ssh/id_*` → No such file or directory

**Fix**:
1. Fine-grained PAT (permission = `Contents: write`) 存到 `~/.git-credentials` (mode 600):
   ```
   git config --global credential.helper store
   echo "https://LiuAnclouds:<PAT>@github.com" > ~/.git-credentials
   chmod 600 ~/.git-credentials
   ```
2. `origin` 保持干净 URL `https://github.com/LiuAnclouds/oe_locateanything.git` (不把 token 存进 `.git/config`)。
3. 首次 push 手动加临时 `https://user:token@` URL 触发 credential helper 记录（我们本轮直接写文件，跳过 prompt）。

**Alternatives considered**:
- 4090 生成 ed25519 keypair 并加到 GitHub —— SSH :22 被内网墙，走不通。
- Windows 中转（bundle → scp → push）—— 沿用老链路，但每次多一步 scp，不推荐作为长期方案。
- 走 gh CLI —— 底层还是 HTTPS + PAT，不比直接 credential helper 简单。

**Prevention**:
- 不把 token 明文写进 `.git/config` 或 shell 命令行历史（曾经在一次 `git remote set-url` 里出现过，事后立刻清掉）。
- Token 泄露风险应对：GitHub → Settings → Developer settings → PAT 页面可以随时 revoke，rotate 时只改 `~/.git-credentials` 一行。
- 未来如果切到 fine-grained PAT，权限只勾 `Contents: write` + `Metadata: read`，不给整个 org / 其他 repo。

---

## #012 MoonViT LayerNorm 用 torch.nn 触发 leap trace 失败 (2026-07-09)

**Symptom**: M3-β vision 编译秒死。
```
TypeError: layer_norm(): argument 'input' (position 1) must be Tensor,
           not hbdk4.compiler._mlir_libs._mlir.ir.OpResult
```
Traceback 定位 `vision_block_leap.py:154 self.norm0(hidden_states)`。

**Trigger**: `oellm_build --model_name locateanything-vit-3b --march nash-p`，编译进入 `export_module` 的 leap trace 阶段。

**Root cause**: `vision_block_leap.py` / `vision_patch_merger_leap.py` / `vision_model_leap.py` 里 norm 层用了 `torch.nn.LayerNorm`。它的 `forward` 调 `F.layer_norm(input, ...)`，要求 `input` 是 `torch.Tensor`；但 leap trace 阶段传入的是 `hbdk4.compiler._mlir_libs._mlir.ir.OpResult`（leap IR 节点），类型不兼容，直接 raise `TypeError`。

不是所有 `torch.nn.*` 都不能进 leap DSL — `torch.nn.Linear` 等模块能走通因为 `leap_llm.nn.modules.DynamicQuantLinear` 是它的 leap-trace-aware wrapper；但 `nn.LayerNorm` 没有 monkey-patch 或替换，必须显式用 `leap_llm.nn.modules.LayerNorm`（该类的 `build(x)` 里走 `leap.layernorm(...)`）。

Language 侧没踩这个坑因为 Qwen2 用 RMSNorm，走的是 `leap_llm.nn.modules.RMSNorm`（已经是 leap-trace-aware）。

**Evidence**:
1. Traceback `.../torch/nn/functional.py:2905 in layer_norm` + `return torch.layer_norm(...)` 明确是 PyTorch 原生 kernel。
2. `grep "^class LayerNorm" toolchain/leap_llm/nn/modules/layer_norm.py` 找到 leap-trace-aware 版存在。
3. 首次编译 PID 4099466 秒死，log_age=10s 时 watchdog 已抓到 traceback。

**Fix**:
1. `vision_block_leap.py`: `from leap_llm.nn.modules import ..., LayerNorm` + `self.norm0 = LayerNorm(config.hidden_size)` × 2 处
2. `vision_patch_merger_leap.py`: 同上 × 2 处
3. `vision_model_leap.py`: 同上 × 1 处 (final_layernorm)
4. 清 `__pycache__` 避免旧字节码
5. 重编，进入下一层错误 #013

**Alternatives considered**:
- 保留 `nn.LayerNorm` 试着 monkey-patch 让它接受 OpResult — 复杂度高，破坏兼容
- 手写 `leap.reduce_mean + leap.sub + leap.reduce_mean + leap.rsqrt + leap.mul` 实现 LN — 无必要，`leap_llm.nn.modules.LayerNorm` 已封装好

**Prevention**:
- `_leap.py` 文件里禁止用 `torch.nn.LayerNorm / BatchNorm / GroupNorm / RMSNorm(内置)`；必须用 `leap_llm.nn.modules.*` 提供的 leap-trace-aware 版
- 加代码前 grep `_leap.py` 里的 `nn\.` 前缀，看是否有非 Linear 的 torch 原生模块
- Vision 塔 (MoonViT) 用 LN 不是 RMSNorm，因 SigLIP-SO400M 系谱；Qwen 家 language 塔用 RMSNorm

---

## #013 Vision attention DynamicQuantMatmul 外部再手动 transpose K 导致 shape 冲突 (2026-07-09)

**Symptom**:
```
loc(...blocks.0.qk_matmul): error: 'hbir.block_quantized_matmul' op cannot infer result type:
  kernel native::BlockQuantizedTransRhsMatmul config function call failure!
  Due to lhs[-1]:72 is not equal to rhs[-2]:1024
```
lhs[-1]=72 = MoonViT head_dim (1152/16)，rhs[-2]=1024 = image seq_len (448²/14²)，形状对不上。

**Trigger**: 修 #012 后重编，PID 4106407，log_age=42s 时抓到。`_attention_leap` 里 `qk_matmul(q, k_transposed_manually)`。

**Root cause**: `DynamicQuantMatmul` 内部展开为 `leap.block_quantized_matmul`，这个 op **假设 RHS 是 K 的 "trans-rhs" 形态**（即 K 的最后两维已经是 `(K, N)` 顺序，op 内部会做 `LHS @ RHS^T`）。

我们的 `vision_block_leap.py` 在 line 143 手动 `k = leap.transpose(k, [0, 1, 3, 2])` 把 K 变成 `(1, H, hd, seq)`，然后传给 `qk_matmul(q, k)`。**双重 transpose**：外部一次 + block_quantized_matmul 内部 assume 一次 = 语义变回没转，rhs[-2] 就变成了 seq_len 1024 而不是 head_dim 72。

Text 侧成功编译没踩这个坑，因它用 `FakeQuantMatmul(8, 8, None)` — 普通 matmul，需要外部手动 transpose，语义配套一致。

**Evidence**:
1. `toolchain/leap_llm/nn/modules/matmul.py::DynamicQuantMatmul.build`: `leap.block_quantized_matmul(x_q, y_q, x_s, y_s, mmaAlpha=1024.0)` — 是 "TransRhs" 变体。
2. Text 侧 `text_attention_leap.py:115`: `FakeQuantMatmul(8, 8, None)`, line 255 `key_states.transpose(2, 3)` 手动转 — 配对一致。
3. Upstream `modeling_vit.py::eager_attention`: `q @ k.transpose(-2, -1)` — 语义等价于 "外部 transpose + 普通 matmul"。

**Fix**: 对齐 text 侧成功模式，把 vision `qk_matmul` / `wv_matmul` 从 `DynamicQuantMatmul()` 改为 `FakeQuantMatmul(8, 8, None)`：
```python
# vision_block_leap.py
from leap_llm.nn.modules import DynamicQuantLinear, DynamicQuantMatmul, FakeQuantMatmul, LayerNorm
self.qk_matmul = FakeQuantMatmul(8, 8, None)
self.wv_matmul = FakeQuantMatmul(8, 8, None)
```

**Alternatives considered**:
- 保留 `DynamicQuantMatmul` 但**删掉**外部 `k.transpose(0,1,3,2)` — 也能通，但 `wv_matmul` 那一步也得核对同一约定；保留外部 transpose 让语义符合 upstream 更清晰
- 用 `leap.matmul(x, y, trans_a=False, trans_b=True)` 直接调 op — 绕过 quant wrapper，失去量化收益
- 把 vision attention 也改为 packed QKV multi-head 拆分算 — 工程冗余

**Prevention**:
- `_attention_leap` 里注释区分 `DynamicQuantMatmul` (trans-rhs) vs `FakeQuantMatmul` (plain) 的语义
- 加代码前先在 text_attention_leap.py 找 successful pattern 对齐；vision / text 用不同 matmul 类要注释说明

---

## #014 M3-β 圆满闭环 (2026-07-09)

**Symptom**: N/A (成功记录)

**Trigger**: 修完 #012 + #013 后 PID 4117845 编译，wall-clock 75 分钟。

**Root cause**: 前两条修复到位，attention shape 语义与 upstream MoonViT 完全对齐，其他 27 层 encoder + patch_embed + merger + final_layernorm 逐一走过。

**Evidence** (hbdk4 verify):
- march = `nash-p`
- toolkit = `4.10.2a2.dev202603180400+4c23b55.develop`
- graphs = `["visual"]`, single-graph
- input: `(1, 1024, 588)` — 1024 patches (448² / 14²) × 588 flat (3 × 14 × 14 RGB)
- output: `(1, 256, 2048)` — 256 vision tokens (merger 4× reduce) × 2048 (对接 LM hidden_size)
- 中间产物: `.visual.bc` 408M / `.visual_convert.bc` 409M / `.visual.hbo` 1.7G / `.hbm` 463M

**Fix**: N/A

**Alternatives considered**: N/A

**Prevention**:
- M4 unified compile (`locateanything-3b`) 需要同时编译 vision + language + fusion，参考本轮 vision + M2 language 单独编译的 hbm shape 对接接口设计
- Vision output `(1, 256, 2048)` 直接 concat 到 language prefill 的 embed 序列，无需额外投影
- 后续 host runtime 侧读入 vision.hbm 一次 forward → 256 tokens → prepend / merge 到 text embed → language prefill
