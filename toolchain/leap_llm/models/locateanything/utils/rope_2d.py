"""MoonViT 2D rotary position embedding — real-valued expansion of the
original complex implementation.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_vit.py:302 (Rope2DPosEmb)
    LocateAnything_hyphen_3B/modeling_vit.py:201 (apply_rope)

Why this file exists (report chapter 5.5 / pit #2):
  Upstream uses `torch.view_as_complex` / `torch.polar` which are
  unsupported by the leap DSL compiler. We expand every complex op into
  its (real, imag) pair and use only real-valued ops.

Complex-to-real translation table:
  c = a + b*i, w = cos + sin*i          (unit-modulus rotor)
  c * w = (a*cos - b*sin) + (a*sin + b*cos)*i
        = real:  a*cos - b*sin
          imag:  a*sin + b*cos

We store the RoPE tensor as a 2×-last-dim real tensor of shape
  freqs_cos_sin: [..., head_dim/2, 2]         (dim=-1 is [cos, sin])
and apply against tokens reshaped to
  x_pair:        [..., head_dim/2, 2]         (dim=-1 is [a, b])

The 2D nature (x_cis / y_cis) is baked into freqs_cos_sin by interleaving
even/odd channels — see `precompute_freqs_cos_sin` below.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Pre-compute freqs — pure real-valued math, no torch.polar.
# ---------------------------------------------------------------------------
def precompute_freqs_cos_sin(
    max_height: int,
    max_width: int,
    dim: int,
    theta_base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a real tensor of shape (max_height, max_width, dim/2, 2).

    Semantics identical to upstream `Rope2DPosEmb._precompute_freqs_cis`
    (modeling_vit.py:337) but stored as [cos, sin] pairs instead of
    complex numbers. The last-2 dims are:
        (dim/2, 2)  where the (dim/2) axis alternates x-channel / y-channel
                     and the last size-2 axis is [cos(theta), sin(theta)].

    x_freqs at even indices (2i)   correspond to the x (width) axis
    y_freqs at odd  indices (2i+1) correspond to the y (height) axis
    exactly matching the upstream cat + reshape pattern:
        cat([x_cis[..., None], y_cis[..., None]], dim=-1)
        .reshape(H, W, dim/2)   # so channel i pairs with axis (i % 2)
    """
    assert dim % 4 == 0, "MoonViT RoPE requires head_dim divisible by 4"
    N = max_height * max_width

    flat_pos = torch.arange(N, device=device, dtype=dtype)
    x_pos = flat_pos % max_width       # shape (N,)
    y_pos = flat_pos // max_width      # shape (N,)

    # Same frequency schedule as upstream: dim_range = 0, 4, 8, ..., dim-4
    # producing dim/4 channels; each channel is used *twice* (once for x, once
    # for y) so the final RoPE has dim/2 (cos,sin) pairs.
    dim_range = torch.arange(0, dim, 4, device=device, dtype=dtype)[: dim // 4]
    inv_freq = 1.0 / (theta_base ** (dim_range / dim))    # (dim/4,)

    x_theta = torch.outer(x_pos, inv_freq)                # (N, dim/4)
    y_theta = torch.outer(y_pos, inv_freq)                # (N, dim/4)

    x_cos, x_sin = torch.cos(x_theta), torch.sin(x_theta)
    y_cos, y_sin = torch.cos(y_theta), torch.sin(y_theta)

    # Interleave x/y channels to match upstream reshape order.
    # Stack shape: (N, dim/4, 2, 2)  = (N, dim/4, {x,y}, {cos,sin})
    cos_stack = torch.stack([x_cos, y_cos], dim=-1)      # (N, dim/4, 2)
    sin_stack = torch.stack([x_sin, y_sin], dim=-1)      # (N, dim/4, 2)
    freqs_cos_sin = torch.stack([cos_stack, sin_stack], dim=-1)  # (N, dim/4, 2, 2)
    #                                                              └cos/sin
    # Flatten x/y axis into channel axis to obtain the final (N, dim/2, 2) layout.
    freqs_cos_sin = freqs_cos_sin.reshape(N, dim // 2, 2)
    freqs_cos_sin = freqs_cos_sin.reshape(max_height, max_width, dim // 2, 2)
    return freqs_cos_sin


def gather_freqs_by_grid(
    freqs_cos_sin: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Slice the pre-computed table down to a specific grid.

    Static-shape slice suitable for compiled graphs: takes the top-left
    (grid_h, grid_w) block from a (max_height, max_width, dim/2, 2) table
    and flattens spatial dims.

    Args:
      freqs_cos_sin : (max_H, max_W, dim/2, 2)
      grid_h, grid_w: post-merge grid (i.e. after 2×2 patch merger)

    Returns:
      (grid_h*grid_w, dim/2, 2)
    """
    sliced = freqs_cos_sin[:grid_h, :grid_w]                  # (h, w, dim/2, 2)
    return sliced.reshape(grid_h * grid_w, sliced.shape[-2], 2)


# ---------------------------------------------------------------------------
# Apply — real-valued expansion of (a+bi)(cos+i·sin).
# ---------------------------------------------------------------------------
def apply_rope_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D RoPE to Q/K without touching complex numbers.

    Args:
      xq, xk        : (..., num_heads, head_dim)
      freqs_cos_sin : broadcastable to (..., 1, head_dim/2, 2)
                      the trailing axis is [cos(theta), sin(theta)]

    Returns:
      xq_out, xk_out : same shape as xq, xk

    Math (per pair):
      (a + b i)(cos + i sin) = (a cos - b sin) + (a sin + b cos) i
      -> new_a = a*cos - b*sin
         new_b = a*sin + b*cos
    """
    out_dtype = xq.dtype
    # Cast to float32 for numerical safety — upstream also promotes to fp32
    # inside view_as_complex.
    xq_f = xq.float()
    xk_f = xk.float()

    # (..., num_heads, head_dim) -> (..., num_heads, head_dim/2, 2)
    xq_pair = xq_f.reshape(*xq_f.shape[:-1], -1, 2)
    xk_pair = xk_f.reshape(*xk_f.shape[:-1], -1, 2)

    # Broadcast freqs to (..., 1, head_dim/2, 2) so it hits every head.
    if freqs_cos_sin.dim() == xq_pair.dim() - 1:
        freqs = freqs_cos_sin.unsqueeze(-3)     # insert num_heads axis
    else:
        freqs = freqs_cos_sin
    cos = freqs[..., 0]                          # (..., 1, head_dim/2)
    sin = freqs[..., 1]

    a_q, b_q = xq_pair[..., 0], xq_pair[..., 1]
    a_k, b_k = xk_pair[..., 0], xk_pair[..., 1]

    q_new_a = a_q * cos - b_q * sin
    q_new_b = a_q * sin + b_q * cos
    k_new_a = a_k * cos - b_k * sin
    k_new_b = a_k * sin + b_k * cos

    xq_out = torch.stack([q_new_a, q_new_b], dim=-1).flatten(-2)
    xk_out = torch.stack([k_new_a, k_new_b], dim=-1).flatten(-2)
    return xq_out.to(out_dtype), xk_out.to(out_dtype)


# ---------------------------------------------------------------------------
# Optional torch.nn.Module wrapper for calibration convenience.
# ---------------------------------------------------------------------------
class MoonViT2DRotary(torch.nn.Module):
    """Table-lookup 2D RoPE module.

    Registers the pre-computed table as a non-persistent buffer so it moves
    with the model on .to(device) but does not enter state_dict (the table
    is deterministic, no need to save weights).
    """

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        table = precompute_freqs_cos_sin(
            max_height=max_height,
            max_width=max_width,
            dim=dim,
            theta_base=theta_base,
        )
        self.register_buffer("freqs_cos_sin", table, persistent=False)

    def get_grid(self, grid_h: int, grid_w: int) -> torch.Tensor:
        return gather_freqs_by_grid(self.freqs_cos_sin, grid_h, grid_w)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_height={self.max_height}, "
            f"max_width={self.max_width}, theta_base={self.theta_base}"
        )
