"""M2 精度验证 — LocateAnything language HBM ↔ PyTorch baseline.

Compares:
  A. hbdk4.runtime 加载 language HBM，在给定 (inputs_embeds, position_ids,
     attention_mask, kv_cache) 输入下计算 logits + updated KV。
  B. PyTorch reference (transformers-based LocateAnythingForConditionalGeneration
     with `_attn_implementation="sdpa"`) 在同样输入下计算 logits + KV。
  C. 差值统计: |logits_A - logits_B| max/mean/p95、argmax token 一致率、
     KV cache max diff。

用法（M2 编译完成后运行）：

  python main/examples/verify_lm_hbm.py \\
      --hbm ~/oe_locateanything/main/language/baseline_outputs/locateanything-lm-3b_nash-p_w4/LocateAnything-3B_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm \\
      --embed_bin ~/oe_locateanything/main/language/baseline_outputs/locateanything-lm-3b_nash-p_w4/LocateAnything-3B_embed_tokens.bin \\
      --model_dir ~/oe_locateanything/eagle/Embodied/LocateAnything-3B \\
      --mode prefill  # 或 decode-pbd

TODO(编译完再填):
  - hbdk4 python runtime API 具体调用方式 (待查 leap_llm/nn/utils.py 里的
    verifier 例子或 oellm_runtime API)
  - PyTorch baseline 的 prefill/decode 输入 dtype 对齐 (fp16 vs bf16)
  - PBD decode 6-token 场景的 attention_mask 精确构造 (用 pbd_mask.py)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# HBM inference — TBD (placeholder while M2 compiles)
# ---------------------------------------------------------------------------
def run_hbm_prefill(hbm_path: str, embed_bin: str, input_ids: torch.Tensor,
                     position_ids: torch.Tensor, attention_mask: torch.Tensor,
                     kv_cache: list[torch.Tensor]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load language HBM and run one prefill.

    Placeholder — populate after M2 compile finishes and we can inspect the
    hbdk4.runtime API. Expected shape based on leap_input_types:
      logits: (bs, seq_len, vocab_size)
      new_kv: list[Tensor] of length 2 * num_layers
              each (bs, cache_len, num_kv_heads, head_dim)
    """
    raise NotImplementedError(
        "HBM prefill runner — to be implemented once M2 hbm lands. Use "
        "hbdk4.runtime + embed_bin memory-mapped table for embed lookup."
    )


def run_hbm_decode_pbd(hbm_path: str, embed_bin: str, block_ids: torch.Tensor,
                        position_ids: torch.Tensor, attention_mask: torch.Tensor,
                        kv_cache: list[torch.Tensor]) -> tuple[np.ndarray, list[np.ndarray]]:
    """One PBD decode step: block_ids shape (1, 6), returns logits (1, 6, vocab)."""
    raise NotImplementedError("HBM PBD decode runner — TBD after M2 compile")


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def load_pytorch_baseline(model_dir: str, dtype=torch.float16):
    """Load LocateAnythingForConditionalGeneration in eager SDPA mode."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    return model.eval()


def torch_prefill(model, input_ids: torch.Tensor, position_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> tuple[torch.Tensor, list]:
    """Run baseline prefill; returns (logits, past_key_values list)."""
    # LocateAnythingForConditionalGeneration has an internal Qwen2ForCausalLM.
    # We forward through it directly for a clean logits+kv comparison.
    with torch.no_grad():
        out = model.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    return out.logits, out.past_key_values


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
def compare_logits(hbm_logits: np.ndarray, ref_logits: torch.Tensor,
                    label: str = "logits") -> dict:
    ref = ref_logits.float().cpu().numpy()
    hbm = hbm_logits.astype(np.float32)
    diff = np.abs(hbm - ref)
    p95 = np.percentile(diff, 95)
    argmax_agree = float((hbm.argmax(-1) == ref.argmax(-1)).mean())
    stats = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(p95),
        "argmax_agreement": argmax_agree,
    }
    print(f"[{label}]")
    for k, v in stats.items():
        print(f"  {k:24s}: {v:.6e}" if isinstance(v, float) and abs(v) < 1
              else f"  {k:24s}: {v}")
    return stats


def compare_kv(hbm_kv: list[np.ndarray], ref_kv: list, label: str = "kv") -> dict:
    """Compare KV cache tensors layer by layer."""
    max_diffs = []
    for i, (hb, rf) in enumerate(zip(hbm_kv, ref_kv)):
        r = rf.float().cpu().numpy() if hasattr(rf, "float") else rf
        d = float(np.abs(hb.astype(np.float32) - r).max())
        max_diffs.append(d)
    stats = {
        "num_layers": len(max_diffs),
        "max_max_diff": max(max_diffs) if max_diffs else 0.0,
        "mean_max_diff": float(np.mean(max_diffs)) if max_diffs else 0.0,
    }
    print(f"[{label}] " + " ".join(f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in stats.items()))
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbm", required=True, help="path to language .hbm")
    ap.add_argument("--embed_bin", required=True, help="path to embed_tokens.bin (fp16)")
    ap.add_argument("--model_dir", required=True,
                    help="LocateAnything-3B checkpoint dir (for PyTorch baseline)")
    ap.add_argument("--mode", choices=["prefill", "decode-pbd"], default="prefill")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--cache_len", type=int, default=1024)
    ap.add_argument("--block_size", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Sanity: HBM file exists
    assert os.path.exists(args.hbm), f"hbm missing: {args.hbm}"
    assert os.path.exists(args.embed_bin), f"embed_bin missing: {args.embed_bin}"
    hbm_size = os.path.getsize(args.hbm) / 1e9
    print(f"[verify_lm_hbm] hbm size: {hbm_size:.2f} GB")

    # Build synthetic inputs (deterministic).
    input_ids = torch.randint(0, 152681, (1, args.chunk_size))
    position_ids = torch.arange(args.chunk_size).unsqueeze(0)
    attention_mask = torch.zeros(1, 1, args.chunk_size, args.cache_len,
                                  dtype=torch.float16)
    # Causal mask over new tokens region
    tri = torch.triu(torch.ones(args.chunk_size, args.chunk_size), diagonal=1).bool()
    causal = torch.zeros(args.chunk_size, args.chunk_size, dtype=torch.float16)
    causal.masked_fill_(tri, float("-inf"))
    attention_mask[:, :, :, :args.chunk_size] = causal

    # PyTorch baseline
    print("[verify_lm_hbm] loading PyTorch baseline...")
    model = load_pytorch_baseline(args.model_dir, dtype=torch.float16)

    if args.mode == "prefill":
        print("[verify_lm_hbm] PyTorch prefill...")
        ref_logits, ref_kv = torch_prefill(model, input_ids, position_ids, attention_mask)
        print(f"  ref logits shape: {ref_logits.shape}")

        # HBM path — TBD
        try:
            hbm_logits, hbm_kv = run_hbm_prefill(
                args.hbm, args.embed_bin, input_ids, position_ids,
                attention_mask, kv_cache=[],
            )
            compare_logits(hbm_logits, ref_logits)
            compare_kv(hbm_kv, ref_kv)
        except NotImplementedError as e:
            print(f"[verify_lm_hbm] SKIPPED HBM path: {e}")
            print("  --- next step: implement run_hbm_prefill using hbdk4.runtime API")

    elif args.mode == "decode-pbd":
        # Build a small prefill'd KV then run one PBD decode block
        raise NotImplementedError("decode-pbd mode — TBD after prefill path works")


if __name__ == "__main__":
    main()
