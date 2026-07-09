import torch

from leap_llm.nn.modules import FakeQuantAdd, FakeQuantRMSNorm
from leap_llm.nn.utils import Module
from hbdk4.compiler import leap
from . import MLP, Attention


class InternLM2DecoderLayer(Module):
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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            preserve_precision=preserve_precision,
        )
        self.feed_forward = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.layer_id = layer_id

        self.attention_norm = FakeQuantRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            preserve_precision=preserve_precision,
        )
        self.ffn_norm = FakeQuantRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            preserve_precision=preserve_precision,
        )

        quantized = False
        self.add_res_hidden = FakeQuantAdd(quantized=quantized)
        self.add_res_hidden_mlp = FakeQuantAdd(quantized=quantized)

    def build(self, hidden_states, cos, sin, cache_k, cache_v, mask):

        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)

        hidden_states, new_key, new_value = self.attention(
            hidden_states, cos, sin, cache_k, cache_v, mask
        )
        residual = leap.cast_type(residual, output_type=leap.float16)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
        hidden_states = self.add_res_hidden(residual, hidden_states)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        residual = leap.cast_type(residual, output_type=leap.float16)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
        hidden_states = self.add_res_hidden_mlp(residual, hidden_states)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)

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
        hidden_states = self.attention_norm(hidden_states)

        hidden_states, new_key, new_value = self.attention(
            hidden_states, cos, sin, cache_k, cache_v, mask
        )

        hidden_states = self.add_res_hidden(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.add_res_hidden_mlp(residual, hidden_states)

        return hidden_states, new_key, new_value
