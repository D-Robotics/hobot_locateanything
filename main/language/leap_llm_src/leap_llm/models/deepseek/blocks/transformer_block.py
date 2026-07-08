import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    FakeQuantAdd,
    FakeQuantRMSNorm,
    RMSNorm,
)
from leap_llm.nn.utils import Module

from . import MLP, Attention


class DecoderLayer(Module):
    def __init__(
        self,
        layer_id: int,
        preserve_precision: bool,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        rope_theta: float,
        intermediate_size: int,
        rms_norm_eps: float,
        march: str,
        w_bits: int,
        has_scale: bool,
        fuse_norm: bool,
    ) -> None:
        super().__init__()
        self.march = march
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.preserve_precision = preserve_precision
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            w_bits=w_bits,
            has_scale=has_scale,
            march=march,
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w_bits=w_bits,
            has_scale=has_scale,
            march=march,
        )
        self.layer_id = layer_id

        if "nash-p" in self.march:
            self.input_layernorm = RMSNorm(
                hidden_size,
                eps=rms_norm_eps,
            )
            self.post_attention_layernorm = RMSNorm(
                hidden_size,
                eps=rms_norm_eps,
            )
        else:
            self.input_layernorm = FakeQuantRMSNorm(
                hidden_size,
                eps=rms_norm_eps,
                preserve_precision=preserve_precision,
                fuse_norm=fuse_norm,
            )
            self.post_attention_layernorm = FakeQuantRMSNorm(
                hidden_size,
                eps=rms_norm_eps,
                preserve_precision=preserve_precision,
                fuse_norm=fuse_norm,
            )

        quantized = False if preserve_precision else True
        self.add_res_hidden = FakeQuantAdd(quantized=quantized)
        self.add_res_hidden_mlp = FakeQuantAdd(quantized=quantized)

    def build(self, hidden_states, cos, sin, cache_k, cache_v, mask):
        if "nash-p" in self.march:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            hidden_states, new_key, new_value = self.self_attn(
                hidden_states, cos, sin, cache_k, cache_v, mask
            )
            hidden_states = leap.add(residual, hidden_states)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = leap.add(residual, hidden_states)
            return hidden_states, new_key, new_value
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            hidden_states, new_key, new_value = self.self_attn(
                hidden_states, cos, sin, cache_k, cache_v, mask
            )
            if self.preserve_precision:
                residual = leap.cast_type(residual, output_type=leap.float16)
                hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
                hidden_states = self.add_res_hidden(residual, hidden_states)
                hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                residual = leap.cast_type(residual, output_type=leap.float16)
                hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
                hidden_states = self.add_res_hidden_mlp(residual, hidden_states)
                hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)
            else:
                hidden_states = self.add_res_hidden(residual, hidden_states)
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = self.add_res_hidden_mlp(residual, hidden_states)
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

        hidden_states = self.add_res_hidden(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.add_res_hidden_mlp(residual, hidden_states)
        return hidden_states, new_key, new_value
