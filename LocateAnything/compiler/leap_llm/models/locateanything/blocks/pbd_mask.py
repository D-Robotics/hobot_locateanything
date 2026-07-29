"""PBD attention mask constructors — SDPA-friendly additive masks.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/mask_sdpa_utils.py:104
      (update_causal_mask_for_one_gen_window_2d)

The PBD decode HBM consumes a 4D attention_mask of shape
  (batch, 1, q_len, kv_len)
where entries are additive (0.0 = allow, -inf = mask).

Two variants matter for M2:

  1. build_pbd_decode_mask
       q_len = block_size (typ. 6). kv_len = past + block_size.
       Inside the last block_size×block_size region the mask is 0
       (bidirectional). Everything else is standard causal.

  2. build_causal_prefill_mask
       q_len = kv_len. Standard triangular; PBD does not touch prefill.

Both are pure tensor construction — no data dependency — so host-side
Python builds them and passes them into the compiled decode HBM as a
plain input tensor. The compile-side leap graph never branches on
causal_attn.
"""

from __future__ import annotations

from typing import Optional

import torch


def build_causal_prefill_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Standard additive causal mask for prefill.

    Returns:
      (batch_size, 1, seq_len, seq_len) with 0 on/below the diagonal
      and -inf above.
    """
    mask = torch.zeros(
        (batch_size, 1, seq_len, seq_len), dtype=dtype, device=device,
    )
    causal = torch.triu(
        torch.ones(seq_len, seq_len, device=device), diagonal=1,
    ).bool()
    tri = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    tri.masked_fill_(causal, float("-inf"))
    mask[:] = tri
    return mask


def build_pbd_decode_mask(
    batch_size: int,
    past_len: int,
    block_size: int = 6,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    causal_attn: bool = False,
    use_cache: bool = True,
) -> torch.Tensor:
    """Build the additive attention mask for one PBD decode step.

    Layout:
      q_len = block_size (the diffusion window)
      kv_len = past_len + block_size

    Behaviour when causal_attn=False (LocateAnything default):
      - The block_size × block_size lower-right region is 0.0 everywhere
        (bidirectional within the window).
      - The block attends freely to all past tokens (past region = 0.0).
      - use_cache=True masks the single token just before the window
        (see upstream comment: 'Mask the last token from previous round to
        prevent recomputation'). This is the position (past_len - 1).

    Behaviour when causal_attn=True:
      - Standard triangular over the window itself.
      - The pre-window mask column is not injected.

    Returns:
      (batch_size, 1, block_size, past_len + block_size)
    """
    kv_len = past_len + block_size

    # Start from an all-zero mask (past tokens fully attendable).
    mask = torch.zeros(
        (batch_size, 1, block_size, kv_len), dtype=dtype, device=device,
    )

    # Fill the window-vs-window region (last block_size columns).
    if not causal_attn:
        # Bidirectional inside the window.
        window = torch.zeros(block_size, block_size, dtype=dtype, device=device)
    else:
        # Strict causal inside the window.
        window = torch.zeros(block_size, block_size, dtype=dtype, device=device)
        upper = torch.triu(
            torch.ones(block_size, block_size, device=device), diagonal=1,
        ).bool()
        window.masked_fill_(upper, float("-inf"))
    mask[:, :, :, past_len:] = window

    # Optional: mask the last token from the previous round (upstream comment).
    # We only apply this when we actually have a previous round to mask
    # (past_len >= 1). The masked column is `past_len - 1`.
    if use_cache and past_len >= 1:
        mask[:, :, :, past_len - 1] = float("-inf")

    return mask
