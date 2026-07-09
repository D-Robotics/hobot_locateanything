import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import LayerNorm, RMSNorm
from leap_llm.nn.utils import Module

from . import MLP, Attention, Encoder, FeedForward


class DecoderLayer(Module):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        intermediate_size: int,
        rms_norm_eps: float,
        w_bits: int,
        has_scale: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            w_bits=w_bits,
            has_scale=has_scale,
        )
        self.mlp = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w_bits=w_bits,
            has_scale=has_scale,
        )
        self.layer_id = layer_id

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def build(self, hidden_states, cos, sin, cache_k, cache_v, mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, new_key, new_value = self.self_attn(
            hidden_states, cos, sin, cache_k, cache_v, mask
        )

        hidden_states = leap.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = leap.add(residual, hidden_states)
        return hidden_states, new_key, new_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        mask: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, new_key, new_value = self.self_attn(
            hidden_states, cos, sin, cache_k, cache_v, mask
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_key, new_value


class EncoderLayer(Module):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.attn = Encoder(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.layer_id = layer_id

        self.norm1 = LayerNorm(hidden_size, eps=layer_norm_eps, bias=True)
        self.norm2 = LayerNorm(hidden_size, eps=layer_norm_eps, bias=True)

        self.ls1 = torch.nn.Parameter(torch.ones(hidden_size))
        self.ls2 = torch.nn.Parameter(torch.ones(hidden_size))

    def build(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = leap.mul(hidden_states, self.ls1.data)
        hidden_states = leap.add(residual, hidden_states)
        residual = hidden_states

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.mul(hidden_states, self.ls2.data)
        hidden_states = leap.add(residual, hidden_states)

        return hidden_states

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states) * self.ls1
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states) * self.ls2
        return residual + hidden_states
