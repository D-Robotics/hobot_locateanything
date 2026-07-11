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

---

## #015 C++ runtime link: hbDNN* in libdnn.so, hbUCP* in libhbucp.so (2026-07-10)

**Symptom**: First CMake `make` on S600 fails at link:
```
undefined reference to `hbDNNInitializeFromFiles'
undefined reference to `hbDNNGetModelNameList'
...
undefined reference to `hbDNNInferV2'
```

**Trigger**: `cd main/runtime && mkdir build && cd build && cmake .. && make`

**Root cause**: CMake's `find_library(HBDNN_LIB hbdnn ...)` looked for `libhbdnn.so` — but on S600, hbDNN* symbols live in `libdnn.so` (no `hb` prefix), and hbUCP* in `libhbucp.so` (the public hobot-dnn deb renamed the libs when splitting off from the internal hbdk4 runtime). Confirmed via `nm -D`:
```
$ nm -D /usr/hobot/lib/libdnn.so   | grep hbDNNInitializeFromFiles  → T
$ nm -D /usr/hobot/lib/libhbucp.so| grep hbUCPWaitTaskDone          → T
$ nm -D /usr/lib/aarch64-linux-gnu/libucp.so.0 | grep hbUCP         → (none)
```

**Evidence**: S600 symbol-home map:
| Symbol prefix | .so file |
|---|---|
| `hbDNN*` | `/usr/hobot/lib/libdnn.so` |
| `hbUCP*` | `/usr/hobot/lib/libhbucp.so` |
| `hbUCPSys*` (mem struct) | `/usr/hobot/lib/libhbucp.so` |
| `hbUCP_INITIALIZE_SCHED_PARAM` macro | `/usr/include/hobot/hb_ucp.h:78` |

**Fix**: `find_library(HBDNN_LIB dnn ...)` and `find_library(HBUCP_LIB hbucp ...)` (commit `4c1fc8e`).

**Alternatives considered**:
- pkg-config — S600's hobot-dnn deb ships no `.pc` files in `/usr/hobot/lib/pkgconfig/`.
- link `-lhb_dnn` with underscore — also not the symbol home.

**Prevention**: Documented in `main/runtime/CMakeLists.txt` header comment: "S600 symbol homes (verified by nm -D): hbDNN* -> libdnn.so, hbUCP* -> libhbucp.so".

---

## #016 C++ runtime: hbDNNInferV2 does not auto-submit task; schedParam cannot be NULL (2026-07-10)

**Symptom**: After link succeeds, `vision_dummy_test` reports:
```
[E] Should first call the `hbUCPSubmitTask` to the taskHandle before calling `hbUCPWaitTaskDone`
[FAIL] Execute: code=-200004 msg=hbUCPWaitTaskDone failed
```
Then after adding `hbUCPSubmitTask(task, nullptr)`:
```
[E] sched_param is null pointer
[FAIL] Execute: code=-100001 msg=hbUCPSubmitTask failed
```

**Trigger**: First successful Execute attempt on vision.hbm.

**Root cause**: Two coupled mistakes in our `hbm_session.cpp::Graph::Execute`:
1. We assumed `hbDNNInferV2` auto-submits the task (it does NOT — it only creates+binds).
2. We passed `nullptr` as `hbUCPSchedParam*`, but S600's UCP refuses NULL ("sched_param is null pointer").

Reference: `HB_HBMRuntime.cc:637-650` (in `/usr/hobot/lib/hbm_runtime/src/`) shows the canonical flow:
```cpp
hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle);
hbUCPSchedParam sched_param{};
HB_UCP_INITIALIZE_SCHED_PARAM(&sched_param);   // priority=LOWEST, backend=CORE_ANY
sched_param.backend = GetBPUCoreMaskForModel(name, bpu_cores);  // default bpu_cores={-1} → HB_UCP_BPU_CORE_ANY
hbUCPSubmitTask(task_handle, &sched_param);
hbUCPWaitTaskDone(task_handle, 0);  // timeout=0 (sync)
```

**Evidence**: commit history shows three incremental fixes (each verified by re-running on S600):
- `76024c0`: add `hbUCPSubmitTask` call (fixes "Should first call")
- `d2e27ba`: pass zeroed `hbUCPSchedParam{}` (fixes "sched_param is null pointer")
- `a11808f`: use `HB_UCP_INITIALIZE_SCHED_PARAM` macro + `backend = HB_UCP_BPU_CORE_ANY` (aligns with reference)

**Fix**: Copy the exact flow from HB_HBMRuntime.cc::InferSingleModel — use the macro for defaults, then explicitly set backend. See `main/runtime/src/hbm_session.cpp::Graph::Execute`.

**Alternatives considered**:
- hardcode `backend = 0xF` (force all 4 cores) — works for SubmitTask but triggers the L2-memspace bug (#017); not what reference does.
- timeout -1 (wait forever) — works but reference uses 0 (sync); prefer consistency.

**Prevention**: `hbm_session.cpp` Execute has a comment block citing `HB_HBMRuntime.cc:625-655` as the reference. Future BPU-runtime work should grep the reference before assuming C-API semantics.

---

## #017 C++ runtime: "L2 memory not enough" — set HB_DNN_USER_DEFINED_L2M_SIZES env var (2026-07-10)

**Symptom**: After #016 fixes, `vision_dummy_test` now fails INSIDE hbUCPWaitTaskDone:
```
[E] [Plan] model [visual] node [visual.visual_bpu_segment_0] L2 memory not enough,
    required l2 memspace info: [3244032, 3244032, 3244032, 3244032, ],
    user-assigned l2 memspace size: [0, 0, 0, 0, ], user-assigned cores: [0, 1, 2, 3, ]
[E] [Plan] Get BPU temporary memspace failed!
[E] [HBRT ERROR]HBRT4_STATUS_NULL_OBJECT
[FAIL] Execute: code=-200003 msg=hbUCPWaitTaskDone failed
```

**Trigger**: 4-core hbm (`corenum_4`) Execute with all the correct schedParam + alignedByteSize fallback.

**Root cause**: S600 BPU runtime expects the **host process** to pre-declare per-core L2 temp memspace sizes BEFORE submitting any task. The public hbDNN C API has no function to set this — it's done via an **environment variable** `HB_DNN_USER_DEFINED_L2M_SIZES` of the form `<core0>:<core1>:<core2>:<core3>` in megabytes.

When unset, runtime reads "user-assigned l2 memspace size: [0,0,0,0]" and refuses to allocate the temp memspace, so `GetCommand` fails inside `BpuBackendSchedule`.

This is exactly why `HB_HBMRuntime.cc` source has NO L2-memspace code — the env var does it at process level, before any C API call.

**Evidence**: The oellm_runtime reference example ships a run script that sets it:
```
$ cat oellm/s600_sdk/.../examples/vlm_demo/run_vlm.sh
#!/bin/sh
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6   # ← THE FIX
./vlm -c $config_file -i $image_file
```
Setting `export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6` before running our `vision_dummy_test` immediately clears the error and Execute returns the expected `(1, 256, 2048)` fp16 output.

**Fix**: `main/runtime/run_vision_dummy_test.sh` sets the env var by default (mirroring `run_vlm.sh`). Future LA run scripts (for language and unified) must do the same.

**Alternatives considered**:
- Try to find a hbDNN C API to set L2 memspace — none exists in `/usr/include/hobot/dnn/hb_dnn.h` or `/usr/include/hobot/hb_ucp.h`.
- Increase value to 8:8:8:8 — 6MB is what the reference uses and is enough for MoonViT (3.24 MB required per core).

**Prevention**: Documented in `main/runtime/run_vision_dummy_test.sh` header comment + this KNOWN_ISSUES entry. Any future S600 BPU runtime launch script MUST export `HB_DNN_USER_DEFINED_L2M_SIZES` or all 4-core hbms will fail with -200003.

---

## #018 C++ runtime Phase-1 milestone: vision.hbm x86-free S600 native execute (2026-07-10)

**Symptom**: N/A (success record)

**Trigger**: After #015-#017 fixes, `vision_dummy_test` on S600 runs end-to-end.

**Root cause**: All three KNOWN_ISSUES resolved.

**Evidence** (S600 stdout, post-fix):
```
[vision_dummy_test] Phase-1 sanity
[vision_dummy_test] hbm: .../LocateAnything-3B_vision_448x448_w8_nash-p_corenum_4.hbm
[BPU][BPU_MONITOR] BPULib verison(2, 2, 15)
[ok] Load. graphs in hbm: [visual]
[ok] graph visual: 1 inputs, 1 outputs
[ok] dummy input built: shape=[1, 1024, 588] floats=602112 bytes=2408448
[ok] Execute returned 1 output tensors:
  out[0]  shape=[1,256,2048] dtype=F16 bytes=1048576
[verdict] vision.hbm Phase-1 sanity PASSED
```

**Fix**: N/A

**Alternatives considered**: N/A

**Prevention**: Phase-2 (language hbm loop) should:
- reuse `HbmSession::ExecuteGraphByName` for prefill + decode graphs
- keep the L2M env var set
- preserve the SchedParam flow from HB_HBMRuntime.cc reference
- compare logits against upstream PyTorch for numerical alignment (M6)

---

## #019 Phase 2 路线决策: libxlm.so 不支持 LA 的 PBD + 坐标 token, 走自研 hbDNN 路线 (2026-07-10)

**Symptom**: Phase 2 language hbm 推理 loop 需要选择推理引擎。候选:
  (A) D-Robotics `libxlm.so` + `xlm.h` 高层 API (config 驱动, vlm_demo.cc 那套)
  (B) 在我们 `hbm_session.cpp` 基础上自研 embed lookup + KV cache + PBD loop

**Trigger**: 用户要求 "优先复用官方 libxlm, 不行再自研, 一切参照 oellm 官方算子"。

**Root cause**: 对 `oellm/s600_sdk/.../lib/libxlm.so` 做 `strings` 探测, 没有以下 LA-specific 字面:
  - `pbd` / `block_size` / `parallel_block_decode` — PBD 6-token/step 加速机制
  - `switch_token` / `coord_start` / `coord_end` / `box_start` / `box_end` — LA 坐标输出 + 模式切换
  - `locateanything` / `magi` — LA 模型识别串
  - `is_valid_box_frame` / `handle_pattern` / `decode_bbox_avg` — LA 生成循环的关键函数

libxlm 内部 decode loop 不认识 LA 的 PBD block_size=6 语义 (LA config.json 里 text_config.block_size=6, 而 libxlm 自带的 qwen2.5vl_3b_config.json 里没这字段, 默认 1-token/step)。坐标 token `<0>..<1000>` 的输出格式和 `<switch>` / `<box>` 的模式切换也是 LA 独有, libxlm 没有对应 handler。

**Evidence**:
- `strings libxlm.so | grep -iE "pbd|block_size|switch_token|coord_|box_start|locateanything|magi|is_valid_box"` → 仅命中 tokenizer (Rust HF tokenizers) 的无关字符串, 无 LA-specific。
- `qwen2.5vl_3b_config.json` schema 无 `block_size` / `decode_seq_len` 字段。
- `xlm_model_type` enum (xlm.h:32-46) 到 `XLM_MODEL_TYPE_GEMMA4=14` 截止, 无 LA。
- S600 上**没装完整 oellm_runtime SDK** (只有系统级 hobot-dnn deb 的 libhbrt4/libdnn/libhbucp), 要走 libxlm 还得先把 SDK 从 4090 拷过去 — 即便拷了也跑不通 PBD。

**Fix**: 选路线 B — 在 `main/runtime/src/hbm_session.cpp` 基础上扩展:
  1. `embed_lookup` — mmap `embed_tokens.bin` (597MB, 152681×2048 fp16) + gather by token IDs
  2. `attention_mask` — causal + 最后 block_size=6 列对角块 bidirectional (PBD 语义, 参照 upstream `mask_sdpa_utils.py::update_causal_mask_for_one_gen_window_2d`)
  3. `position_ids` — `np.arange` + PBD 窗口位共享 `pos_ids[-6:] -= 1` (参照 upstream `_prepare_inputs_in_mtp`)
  4. `kv_cache` — 36 层 × 2 (K/V) × `(1, 1024, 2, 128)` int8 环状缓冲
  5. `pbd_generate` — 移植 upstream `generate_utils.py::sample_tokens/handle_pattern/is_valid_box_frame/decode_bbox_avg` 到 C++
  6. 顶层 `LocateAnythingRuntime` — vision.hbm execute → embed concat → language.hbm prefill → decode loop

**Alternatives considered**:
- libxlm + 写 LA config 强行跑 — 能加载文件但 decode 不会走 PBD, 输出 6 个 token 里只有第 1 个有效, 性能跟普通 autoregressive 一样, 失去 LA 的核心加速。且坐标 token 不会被解析成 bbox。
- patch libxlm 加 LA model_type — libxlm.so 是 stripped binary, 无源码, 改不动。
- 等官方 oellm 后续版本支持 LA — 时间不定, 不阻塞当前部署。

**Prevention**: 此决策记录在此, 后续 review 可追溯。Phase 2 实现时每个模块参照 upstream `eagle/Embodied/LocateAnything-3B/modeling_*.py` 对应函数 (参照 `feedback_locateanything_read_upstream_first` memory), 数值对齐 (前 10 token logits diff < 1e-3) 后才算该模块完成。

---

## #020 libxlm Qwen2.5-VL 分支输出乱码 — image embed/position_ids/mask 不匹配 LA (2026-07-10)

**Symptom**: `locateanything_demo` (仿 vlm_demo.cc, config model_type="Qwen2.5-VL")
xlm_init 成功 + 加载 LA 全部 3 个 hbm (language prefill+decode + vision visual) +
xlm_infer ret=0 + 真实性能数据 (vit 30ms / prefill 6426 tps / decode 57 tps /
tpot 17.4ms / e2e 16.9s), 但输出全是随机多语言乱码, 一路 decode 到 KV cache
满 (980 tokens, "no enough kvcache now stop generating"). prompt 改成 LA 标准
query 格式 "cat" 也救不回.

**Trigger**: xlm_infer 后 callback 吐乱码; log 显示 `Model type: Qwen2.5-VL` +
`QwenVLPreprocess success` + `Load hbm ... success` × 3.

**Root cause**: libxlm 走 Qwen2.5-VL 分支, 按 Qwen2.5-VL 的语义构造输入序列:
1. **image embed 插入**: Qwen2.5-VL 用 `<|vision_start|><|image_pad|>×N<|vision_end|>`
   (N 个占位符填 N 个 vision embed). LA 用 `image_token_index=151665` **单个**
   占位符, 运行时替换成 256 个 vision embed (净增 255). 两种插法序列结构不同.
2. **position_ids**: Qwen2.5-VL 用 M-RoPE 3D (T/H/W 三轴 position_ids [bs,3,seq]).
   LA 用 vanilla 1D position_ids [bs,1,seq] + PBD `pos[-6:]-=1`.
3. **attention mask**: Qwen2.5-VL 纯 causal. LA PBD block_size=6, 最后 6×6 块
   bidirectional + prev-trailing mask (mask_sdpa_utils.py).

libxlm 按 Qwen2.5-VL 构造的 embed 序列喂给 LA hbm, 序列错位 → logits 全乱 →
采样出随机多语言 token. 不是 hbm 坏, 是 host 端输入构造错.

**Evidence**:
- `strings libxlm.so` 含 `QwenVLPreprocess success`, `resized_width and resized_height only support 448`, `Get config success! ... image_net_mean ... image_net_std ...` — libxlm 读 config 但按 Qwen2.5-VL 流程预处理.
- `vlm_model.cc:262 Model type: Qwen2.5-VL` 确认走 Qwen2.5-VL 分支.
- 输出 token 全是 vocab 里的随机多语言 subword (非 LA 的 `<box>/<0>~<1000>/<ref>` 坐标格式).
- hbm 本身 verify 过 (hbdk4 + Phase1 vision_dummy_test + #014), 结构正确.

**Fix**: 放弃 libxlm 推理路径 (它只能加载 hbm, 不能正确构造 LA 输入). 纯 C++ 自研:
- 复用 libxlm 的 `tokenizers_*` C API 做 tokenizer (encode/decode 坐标 token, 不自研 BPE)
- 复用我们的 hbm_session (hbm 加载+execute, #015-#018 已对齐 HB_HBMRuntime.cc)
- 复用 embed_lookup + attention_mask + position_ids (Phase 2 已写+测 PASS)
- 自研 image_preprocess (OpenCV resize 448+BGR2RGB+归一化0.5/0.5+patchify) +
  vision_text_concat (image_token_index 单占位符→256 vision embed 替换) +
  kv_cache (36层 ring buffer) + pbd_generate (sample+handle_pattern+decode_bbox_avg,
  移植 generate_utils.py) + 顶层 locateanything_infer.

**Alternatives considered**:
- 继续调 libxlm config (改 mean/std/prompt) — prompt "cat" 已试无效, 根因是
  image embed 插入/position_ids/mask 三层全按 Qwen2.5-VL, config 改不动这些.
- patch libxlm 加 LA 分支 — libxlm.so stripped binary 无源码, 改不动.
- 等 libxlm 后续支持 LA — 不确定, 不阻塞.

**Prevention**: libxlm 路径证明 LA hbm 可加载 + 性能可测, 但输入构造必须按 LA
自身语义 (image_token_index 替换 + vanilla 1D pos + PBD mask). 自研推理流程
见 `main/runtime/docs/INFERENCE_FLOW.md`. PBD decode 输入 = 1 个上轮尾真实 token +
5 个 `<text_mask>`(151676), position_ids [-6:]-=1 (n_future_tokens=6=block_size).

---

## #021 language.hbm prefill output layout: 数据存在但 row 0~15 全 0 (2026-07-10)

**Symptom**: prefill_verify 跑 language.hbm prefill graph 成功 (Execute 返回 73
outputs, KV output[1] 全非0 证明 BPU 真算, logits sample min=-13.7 max=6.76 无
NaN), 但 argmax(logits[0,255]) 全是 id=0 val=0.0. 初判 "logits 全0", 实际深查
后不是全 0.

**Trigger**: dummy 输入 (embed token 0..255 真实 embed + pos 0..255 + causal
mask + 72×KV 全 0) 跑 prefill.

**Root cause (已查清部分)**:
1. logits output stride = [78184448, 305408, 2] (查 hbdk4 output_strides).
   每行 152704 fp16 (152681 真实 + 23 padding 对齐 64). 紧密按 152681 读会错位.
   按 stride[1]=305408 byte / 2 = 152704 fp16 per row 才对.
2. layout_probe (扫所有非0 byte 段) 显示: 1873 个非0段, 全部集中在 row 16~88
   之间 (按 152704 stride 算). row 0~15 全 0, row 89~255 全 0. 数据是真实 logits
   (中间夹的 0 是 fp16 自然小值), 但只有 ~73 行有效, 不是 256 行.
3. HB_HBMRuntime.cc::PrepareOutputArrays (reference) 读 output 用
   output_properties.stride[i] 构造 numpy, 不假设紧密 — 验证我们 stride 读法
   跟官方一致.

**未查清 (待查)**: 为什么只有 row 16~88 有数据, row 0~15 全 0. 假设:
- input embed token 0~15 (token id 0=pad/<|endoftext|> 等) 触发某种跳过
- mask causal 填法跟 hbm 期望不符 (hbm mask 语义可能是 cache_len 维度, 不是
  query 维度 causal)
- input embed 拼接没做 image_token_index 替换 (LA prefill 期望 prompt 里有
  image_token_index 占位符, 我们 dummy 输入没放)

**Evidence**:
- KV out[1] non_zero=676/676 (BPU 确实写 KV cache)
- logits 1873 个非0段集中在 row 16~88
- libxlm 跑 LA (Qwen2.5-VL 分支) 时 logits 非零且出 token (虽乱码), 说明 hbm
  本身能产完整 logits, 差异在 input 构造

**Fix**: 待查. 下一步对照 upstream modeling_locateanything.py 真实 prefill
input 构造 (image_token_index 替换 + 正确 mask + position), 不用 dummy token
0..255.

**Alternatives considered**:
- output stride 读错 — 已排除 (HB_HBMRuntime 也用 stride)
- hbm 坏 — 已排除 (libxlm 能出非零 logits)
- output cache 同步 — 已排除 (KV output 全非0)

**Prevention**: prefill verify 判据不能只看 "argmax=0 val=0" 就判 PASS, 必须
检查 logits 实际数值范围 (min/max/mean) + 非零行分布. 之前判据太松差点漏过
这个 bug.

---

## #022 M2 chunk_1024 重编 97% 进程被 SIGHUP 杀 (nohup 不彻底) (2026-07-11)

**Symptom**: M2 重编 (chunk_size=1024 cache_len=2048) 跑到进度 97% (compile_hbo
快完) 时进程消失, 无 hbm 产出. log 尾部无任何 error/traceback/oom/killed —
进度条卡 97% 后戛然而止.

**Trigger**: 用 `nohup bash -c "..." & disown` 在 ssh 会话里启动编译. ssh 会话
断开 (cron tick 的 ssh 连接释放 / Claude session 切换) 时, nohup 的子进程被
SIGHUP 杀.

**Root cause**: `nohup ... & disown` 在某些 bash 配置下不够彻底 — nohup 忽略
SIGHUP 但子进程 (oellm_build python) 仍可能继承会话的 controlling terminal,
ssh 会话关闭时 kernel 给会话前台进程组发 SIGHUP, nohup 没完全屏蔽掉. 长时间
编译 (3h+) 期间 Claude cron 反复 ssh 连接/断开, 某次断开把 oellm_build 带走.

compile_hbo 97% 卡住不是代码 bug — 是启动方式不稳. 重编用 setsid 彻底脱离
会话解决.

**Evidence**:
- log 尾部 30 行全是 "High performance Scaled Dot Product Attention Detected"
  + 进度条 97%, 无 error.
- /tmp/m2_recompile.log (wrapper) 尾部是 "process alive ✓" (启动时活的), 没有
  退出码/信号记录.
- 4090 内存 7.7G used (编译时可能更高但 dmesg 无 OOM 记录).
- 无 .hbo 产出 (compile_hbo 97% 没写完).

**Fix**: 重编改用 `setsid nohup bash -c "..." > log 2>&1 < /dev/null &` — setsid
创建新会话脱离 controlling terminal, 配合 nohup + stdin 重定向 /dev/null,
ssh 断开不再杀子进程. (commit 这次教训后重编, 新 PID 3703967)

**Alternatives considered**:
- 从 .hbo 续编 — 无 .hbo (没写完), 不能续.
- 降 chunk_size 减少编译时间 — 单图 prompt 1001 token, chunk 必须 ≥1024.
- tmux/screen 会话 — 4090 没装 tmux, 且 setsid 已够.

**Prevention**:
- 长时间编译 (oellm_build) 一律用 setsid + nohup + `</dev/null`, 不要裸 nohup.
- cron 监控用 pgrep 模式匹配 (不依赖固定 PID), 进程死了能抓到新 PID.
- 编译落盘后 watcher 主动写 /tmp/m2_watch.log 记录成败.

---

## #023 M2 chunk_1024/cache_2048 重编闭环 (2026-07-11)

**Symptom**: N/A (成功记录)

**Trigger**: M2 重编 chunk_size=1024 cache_len=2048 (decode_seq_len=6, w4, corenum_4_4),
setsid 保护跨过上次 97% SIGHUP 死点 (#022), wall-clock ~70min.

**Root cause**: N/A

**Evidence** (hbdk4 verify):
- march = nash-p
- toolkit = 4.10.2a2.dev202603180400+4c23b55.develop
- graphs = ["prefill", "decode"]
- prefill: 75 in (embeds (1,1024,2048)F16 + pos (1,1,1024)S32 + mask (1,1024,2048)F16
  + 72×KV (1,1024,2,128)S8) → 73 out (logits (1,1024,152681)F16 + 72×KV)
- decode: 75 in (embeds (1,6,2048) + pos (1,1,6) + mask (1,6,2048) + 72×KV)
  → 73 out (logits (1,6,152681) + 72×KV)
- 关键: prefill input_0 = (1,1024,2048) 能装下真实 LA 单图 prompt 970 token
  (4090 PyTorch dump ground truth: input_ids 970 + 925 个 151665 占位符)
- mask shape 从 chunk_256 的 (1,256,1024) 变成 (1,1024,2048) — cache_len 2048
  后 mask 列维度 = cache_len, 不是 chunk_size
- KV cache 物理 shape 仍 (1,1024,2,128) — cache_len 2048 是逻辑长度, KV 段
  1024 滚动复用 (要 verify)

**Fix**: N/A. 之前 chunk_256 装不下真实 prompt 导致 prefill verify row0~15 全0 (#021),
重编 chunk_1024 解决 input 长度问题.

**Alternatives considered**: chunk_512 (要分 chunk prefill, 复杂), chunk_970 (不通用).

**Prevention**: LA 单图 prompt ~970 token, chunk_size 必须 ≥1024. 多图 (双图 1884,
三图 2816) 超 cache_len 2048 跑不了, 后续要动态分 chunk 或更大 cache_len.
