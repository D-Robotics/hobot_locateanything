from __future__ import annotations

import torch
from torch import nn
from hbdk4.compiler import leap
from leap_llm.nn.utils import Module
from leap_llm.nn.modules import DynamicQuantLinear, DynamicQuantMatmul

class Attention(Module):
    """Minimal DiT attention implementation for torch forward only."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        out_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs  # keep constructor compatible with previous calls

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head**-0.5
        self.upcast_attention = upcast_attention

        context_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = DynamicQuantLinear(query_dim, self.inner_dim, bias=bias)
        self.to_k = DynamicQuantLinear(context_dim, self.inner_dim, bias=bias)
        self.to_v = DynamicQuantLinear(context_dim, self.inner_dim, bias=bias)

        # Keep key names compatible with existing checkpoints: to_out.0 / to_out.1
        self.to_out = nn.ModuleList([DynamicQuantLinear(self.inner_dim, query_dim, bias=out_bias)])

        self.qk_matmul = DynamicQuantMatmul()
        self.sv_matmul = DynamicQuantMatmul()

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = tensor.shape
        return tensor.view(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs

        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = self._shape(self.to_q(hidden_states))
        key = self._shape(self.to_k(context))
        value = self._shape(self.to_v(context))

        if self.upcast_attention:
            query = query.float()
            key = key.float()
            value = value.float()

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            # Support [B, Q, K], [B, 1, Q, K], or [B, H, Q, K] additive masks
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores + attention_mask.to(dtype=attn_scores.dtype, device=attn_scores.device)

        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        attn_output = torch.matmul(attn_probs, value)

        # [B, H, Q, D] -> [B, Q, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        bsz, q_len, _, _ = attn_output.shape
        attn_output = attn_output.view(bsz, q_len, self.inner_dim)

        attn_output = self.to_out[0](attn_output)
        return attn_output

    def build(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        del kwargs

        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        bsz, q_len, _ = hidden_states.type.shape
        k_len = context.type.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)

        query = leap.reshape(query, [bsz, q_len, self.heads, self.dim_head])
        query = leap.transpose(query, [0, 2, 1, 3])

        key = leap.reshape(key, [bsz, k_len, self.heads, self.dim_head])
        key = leap.transpose(key, [0, 2, 1, 3])

        value = leap.reshape(value, [bsz, k_len, self.heads, self.dim_head])
        value = leap.transpose(value, [0, 2, 1, 3])

        if self.upcast_attention:
            query = leap.cast_type(query, output_type=leap.float32)
            key = leap.cast_type(key, output_type=leap.float32)
            value = leap.cast_type(value, output_type=leap.float32)

        attn_scores = self.qk_matmul(query, key)
        attn_scores = leap.mul(attn_scores, self.scale)

        if attention_mask is not None:
            if len(attention_mask.type.shape) == 3:
                attention_mask = leap.reshape(attention_mask, [bsz, 1, q_len, k_len])
            attn_scores = leap.add(attn_scores, attention_mask)

        # Match torch path: softmax in fp32 and cast back.
        attn_probs = leap.softmax(attn_scores, -1)
        value = leap.transpose(value, [0, 1, 3, 2])
        attn_output = self.sv_matmul(attn_probs, value)

        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [bsz, q_len, self.inner_dim])
        attn_output = self.to_out[0](attn_output)
        return attn_output

