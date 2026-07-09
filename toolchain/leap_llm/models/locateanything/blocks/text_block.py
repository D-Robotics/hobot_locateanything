"""Qwen2 DecoderLayer — pre-norm + GQA attention + pre-norm + SwiGLU MLP.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_qwen2.py:99  (Qwen2RMSNorm)
    LocateAnything_hyphen_3B/modeling_qwen2.py:927 (Qwen2DecoderLayer)

State-dict layout matches upstream exactly:
  input_layernorm.weight
  post_attention_layernorm.weight
  self_attn.q_proj.{weight,bias}
  self_attn.k_proj.{weight,bias}
  self_attn.v_proj.{weight,bias}
  self_attn.o_proj.weight
  mlp.gate_proj.weight
  mlp.up_proj.weight
  mlp.down_proj.weight
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .text_attention import Qwen2GQAAttentionStatic
from .text_mlp import Qwen2MLPStatic


class Qwen2RMSNormStatic(nn.Module):
    """Byte-identical to upstream Qwen2RMSNorm (modeling_qwen2.py:99)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x.to(in_dtype))


class Qwen2DecoderLayerStatic(nn.Module):
    """Single Qwen2 decoder layer for the compile pipeline.

    Interface aligned with the leap decode-HBM convention used by
    Qwen2_5_VLDecoderLayer (see qwen2_5_vl/blocks/transformer_block.py:62):

      forward(hidden_states, attention_mask, position_ids, cos, sin,
              cache_keys, cache_values)
      -> (hidden_states, new_keys, new_values)

    cos/sin are supplied by the caller (Model wrapper builds the rope table
    once and shares it across layers).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        mlp_bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_layernorm = Qwen2RMSNormStatic(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNormStatic(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen2GQAAttentionStatic(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
        )
        self.mlp = Qwen2MLPStatic(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=mlp_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h, new_keys, new_values = self.self_attn(
            hidden_states=h,
            cos=cos, sin=sin,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        hidden_states = residual + h

        return hidden_states, new_keys, new_values
