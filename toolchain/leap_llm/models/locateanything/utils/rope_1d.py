"""Qwen2 vanilla 1D rotary position embedding — compile-friendly.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_qwen2.py:117 (Qwen2RotaryEmbedding)
    LocateAnything_hyphen_3B/modeling_qwen2.py:154 (rotate_half)
    LocateAnything_hyphen_3B/modeling_qwen2.py:162 (apply_rotary_pos_emb)

Design notes:

1. Qwen2's rope is *not* Qwen2.5-VL's mrope. It's the standard 1D
   position rope from Llama — a single position_ids tensor per token,
   not a 3-way (T, H, W) split.

2. The upstream implementation caches (cos, sin) tables inside a
   nn.Module buffer and grows them on demand. For the compile path we
   pre-compute the tables once at LocateAnythingConfig-driven max size,
   store them as float32 buffers, and gather at forward time.

3. `apply_rotary_pos_emb` here is byte-identical to upstream so the
   real->real transform works the same way. This differs from the
   MoonViT rope which was originally complex; here the upstream is
   already real-valued (rotate_half trick), so no expansion needed.
"""

from __future__ import annotations

import torch
from torch import nn


def precompute_rope_cos_sin(
    dim: int,
    max_position_embeddings: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute (cos, sin) tables of shape (max_seq_len, dim).

    Matches upstream Qwen2RotaryEmbedding._set_cos_sin_cache
    (modeling_qwen2.py:135):
        inv_freq = 1 / base ** (arange(0, dim, 2) / dim)
        t = arange(max_seq_len)
        freqs = t ⊗ inv_freq                     # (seq, dim/2)
        emb = cat([freqs, freqs], dim=-1)        # (seq, dim)  # NB: same freq twice
        cos = emb.cos()   sin = emb.sin()
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                          # (seq, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                   # (seq, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim by -1, matching upstream."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather (cos, sin) at position_ids and rotate q/k.

    Args:
      q, k          : (..., num_heads, seq_len, head_dim)
      cos, sin      : (max_seq, head_dim) — pre-computed table
      position_ids  : (batch, seq_len)
      unsqueeze_dim : 1 when q/k are (bs, heads, seq, hd);
                      2 when q/k are (bs, seq, heads, hd)

    Returns:
      q_embed, k_embed : same shapes as q, k
    """
    cos_at = cos[position_ids].unsqueeze(unsqueeze_dim)       # (bs, 1, seq, hd)
    sin_at = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = q * cos_at + rotate_half(q) * sin_at
    k_embed = k * cos_at + rotate_half(k) * sin_at
    return q_embed, k_embed


class Qwen2RotaryTable(nn.Module):
    """Static rope table for the compile pipeline.

    Registers `cos_cached` and `sin_cached` as non-persistent buffers so they
    move with the model on .to(device) but do not enter state_dict.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        cos, sin = precompute_rope_cos_sin(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(
            q, k, self.cos_cached, self.sin_cached, position_ids, unsqueeze_dim,
        )
