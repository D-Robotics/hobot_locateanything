import torch
from typing import Optional, Tuple

from leap_llm.nn.modules.activation import FakeQuantGELU
from leap_llm.nn.modules import FakeQuantAdd, FakeQuantLinear, Clip
from leap_llm.nn.modules import Qwen2RMSNorm, LayerNormSplit
from leap_llm.nn.utils import Module
from hbdk4.compiler import leap
from .mlp import Qwen2_5OmniMLP, Qwen2_5OmniPatchMergerMLP
from .attention import (
    Qwen2_5OmniAudioAttention,
    Qwen2_5OmniVisionAttention,
    Qwen2_5OmniAttention,
)


class Qwen2_5OmniDecoderLayer(Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2_5OmniAttention(config, layer_idx)

        self.mlp = Qwen2_5OmniMLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        quantized = False
        self.add1 = FakeQuantAdd(quantized=quantized)
        self.add2 = FakeQuantAdd(quantized=quantized)

    def build(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_k: torch.Tensor = None,
        cache_v: torch.Tensor = None,
    ):

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, key_states_out, value_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_k=cache_k,
            cache_v=cache_v,
        )
        residual = leap.cast_type(residual, output_type=leap.float16)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
        hidden_states = self.add1(residual, hidden_states)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        residual = leap.cast_type(residual, output_type=leap.float16)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
        hidden_states = self.add2(residual, hidden_states)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)

        return hidden_states, key_states_out, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_k: torch.Tensor = None,
        cache_v: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, key_states_out, value_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_k=cache_k,
            cache_v=cache_v,
        )
        hidden_states = self.add1(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.add2(residual, hidden_states)

        outputs = (hidden_states, key_states_out, value_states)

        return outputs


class Qwen2_5OmniAudioEncoderLayer(Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.embed_dim = config.d_model
        self.self_attn = Qwen2_5OmniAudioAttention(config)
        self.self_attn_layer_norm = LayerNormSplit(self.embed_dim)
        self.activation_fn = FakeQuantGELU(quantized=True)

        quantized = True
        self.add1 = FakeQuantAdd(quantized=quantized)
        self.add2 = FakeQuantAdd(quantized=quantized)
        self.fc1 = FakeQuantLinear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = FakeQuantLinear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNormSplit(self.embed_dim)

        self.clip = Clip()

    def build(
        self,
        hidden_states,
        attention_mask,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.add1(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.add2(residual, hidden_states)

        element_type = hidden_states.type.element_type
        if str(element_type) == "f16":
            # clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            clamp_value = 64504.0
            hidden_states = self.clip(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = hidden_states

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.add1(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.add2(residual, hidden_states)

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = self.clip(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class Qwen2_5OmniVisionBlock(Module):
    def __init__(self, config) -> None:
        super().__init__()

        preserve_precision = True
        self.norm1 = Qwen2RMSNorm(
            config.hidden_size,
            eps=1e-6,
            preserve_precision=preserve_precision,
            fp16_tiny=False,
        )
        self.norm2 = Qwen2RMSNorm(
            config.hidden_size,
            eps=1e-6,
            preserve_precision=preserve_precision,
            fp16_tiny=False,
        )

        self.attn = Qwen2_5OmniVisionAttention(
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5OmniMLP(config, bias=True)
        quantized = False
        self.add1 = FakeQuantAdd(quantized=quantized)
        self.add2 = FakeQuantAdd(quantized=quantized)

    def build(
        self,
        hidden_states,
        window_attention,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ):
        residual = hidden_states
        hidden_states = self.attn(
            self.norm1(hidden_states),
            window_attention=window_attention,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        residual = leap.cast_type(residual, output_type=leap.float16)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
        hidden_states = self.add1(residual, hidden_states)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)
        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        residual = leap.cast_type(residual, output_type=leap.float16)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
        hidden_states = self.add2(residual, hidden_states)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)
        return hidden_states

    def forward(
        self,
        hidden_states,
        window_attention,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            window_attention=window_attention,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        hidden_states = self.add1(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))

        hidden_states = self.add2(residual, hidden_states)
        return hidden_states


class Qwen2_5OmniPatchMerger(Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(
            context_dim, eps=1e-6, preserve_precision=True, fp16_tiny=False
        )
        self.mlp = Qwen2_5OmniPatchMergerMLP(self.hidden_size, dim)

    def build(self, hidden_states):
        hidden_states = self.ln_q(hidden_states)
        hidden_states = leap.reshape(hidden_states, [-1, self.hidden_size])
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.mlp(hidden_states)
        return hidden_states
