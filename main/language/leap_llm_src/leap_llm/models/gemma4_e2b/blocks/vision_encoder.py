import torch
import torch.nn as nn
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.blocks.linear import Gemma4ClippableLinear
from leap_llm.models.gemma4_e2b.blocks.rmsnorm import Gemma4RMSNorm
from leap_llm.models.gemma4_e2b.blocks.vision_mlp import Gemma4VisionMLP
from leap_llm.models.gemma4_e2b.blocks.vision_rotary_embedding import Gemma4VisionRotaryEmbedding
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4VisionConfig
from leap_llm.nn.modules import (
    DynamicQuantMatmul,
)
from leap_llm.nn.utils import Module


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_leap(x):
    bs, dim1, dim2, head_dim = x.type.shape
    x1 = leap.slice(
        x,
        [0, 0, 0, 0],
        [bs, dim1, dim2, head_dim // 2],
        [1, 1, 1, 1],
    )
    x2 = leap.slice(
        x,
        [0, 0, 0, head_dim // 2],
        [bs, dim1, dim2, head_dim],
        [1, 1, 1, 1],
    )
    x2 = leap.mul(-1, x2)
    rotate_x = leap.concat([x2, x1], -1)
    return rotate_x


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_pos_emb_leap(x, cos, sin):
    x_embed = leap.mul(x, cos)
    x_embed = leap.add(x_embed, leap.mul(rotate_half_leap(x), sin))
    return x_embed


def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    ndim: int = 2,
) -> torch.Tensor:
    """
    x: (bs, #patch, #head, dim)
    cos: (bs, #patch, dim)
    sin: (bs, #patch, dim)
    """
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))

    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            "Invalid configuration: num_rotated_channels_per_dim must be > 0, got"
            f" {num_rotated_channels_per_dim} (num_input_channels={num_input_channels},"
            f" ndim={ndim})"
        )

    # Correctly split the input tensor into ndim parts
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        apply_rotary_pos_emb(
            x=x_parts[k],
            cos=cos_parts[k],
            sin=sin_parts[k],
        )
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


def apply_multidimensional_rope_leap(x, cos, sin, ndim=2, unsqueeze_dim=2):
    bs, num_patches, num_heads, head_dim = x.type.shape
    num_input_channels = x.type.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    # split_sizes = [num_rotated_channels_per_dim] * ndim

    y_parts = []

    for i in range(ndim):
        x_part = leap.slice(
            x,
            [0, 0, 0, i * num_rotated_channels_per_dim],
            [bs, num_patches, num_heads, (i + 1) * num_rotated_channels_per_dim],
            [1, 1, 1, 1],
        )
        cos_part = leap.slice(
            cos,
            [0, 0, 0, i * num_rotated_channels_per_dim],
            [bs, num_patches, 1, (i + 1) * num_rotated_channels_per_dim],
            [1, 1, 1, 1],
        )
        sin_part = leap.slice(
            sin,
            [0, 0, 0, i * num_rotated_channels_per_dim],
            [bs, num_patches, 1, (i + 1) * num_rotated_channels_per_dim],
            [1, 1, 1, 1],
        )
        y_part = apply_rotary_pos_emb_leap(x_part, cos_part, sin_part)
        y_parts.append(y_part)

    output = leap.concat(y_parts, dim=-1)
    return output


class Gemma4VisionAttention(Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = False
        self.q_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = Gemma4ClippableLinear(config, config.num_attention_heads * self.head_dim, config.hidden_size)

        self.q_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        self.qk_matmul = DynamicQuantMatmul()
        self.wv_matmul = DynamicQuantMatmul()

    def forward(self, hidden_states, position_embeddings):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope(query_states, cos, sin)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(key_states, cos, sin)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output

    def build(self, hidden_states, position_embeddings):
        input_shape = hidden_states.type.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states)
        query_states = leap.reshape(query_states, hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope_leap(query_states, cos, sin)
        # （bs, #patch, #head, #head_dim） -> （bs, #head, #patch, #head_dim）
        query_states = leap.transpose(query_states, (0, 2, 1, 3))

        key_states = self.k_proj(hidden_states)
        key_states = leap.reshape(key_states, hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope_leap(key_states, cos, sin)
        # （bs, #patch, #head, #head_dim） -> （bs, #head, #patch, #head_dim）
        key_states = leap.transpose(key_states, (0, 2, 1, 3))

        value_states = self.v_proj(hidden_states)
        value_states = leap.reshape(value_states, hidden_shape)
        value_states = self.v_norm(value_states)
        # （bs, #patch, #head, #head_dim） -> （bs, #head, #patch, #head_dim）-> (bs, #head, #head_dim, #patch)
        value_states = leap.transpose(value_states, (0, 2, 3, 1))

        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = leap.softmax(attn_weights, -1)  # (bs, #head, #patch, #patch)
        # value_states = leap.transpose(value_states, (0, 1, 3, 2)) # (bs, #head, #head_dim, #patch)
        attn_output = self.wv_matmul(attn_weights, value_states)  # (bs, #head, #patch, head_dim)
        attn_output = leap.transpose(attn_output, (0, 2, 1, 3))
        attn_output = leap.reshape(attn_output, (*input_shape, -1))

        attn_output = self.o_proj(attn_output)

        return attn_output


class Gemma4VisionEncoderLayer(Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4VisionAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings):
        """
        Args:
            hidden_states (torch.Tensor): Encoder input. Shape:
                ``(batch_size, num_patches, hidden_size)``, e.g.
                ``(batch_size, 2304, 768)`` for the 768x768 path.
            position_embeddings (bs, #patch, 1, #head_dim)

        Returns:
            torch.Tensor: Encoder output. Shape:
                ``(batch_size, num_patches, hidden_size)``, e.g.
                ``(batch_size, 2304, 768)``.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def build(self, hidden_states, position_embeddings):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        return hidden_states


class Gemma4VisionEncoder(Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config=config, layer_idx=i) for i in range(self.num_layers)]
        )

    def forward(self, inputs_embeds):
        """Run all vision encoder layers with the precomputed RoPE pair.

        Args:
            inputs_embeds (torch.Tensor): Patch embeddings from
                :class:`Gemma4VisionPatchEmbedder`. Shape:
                ``(batch_size, num_patches, hidden_size)``, e.g.
                ``(batch_size, 2304, 768)`` for the 768x768 resolution image.

        Returns:
            torch.Tensor: Encoder output. Shape:
                ``(batch_size, num_patches, hidden_size)``, e.g.
                ``(batch_size, 2304, 768)``.
        """
        hidden_states = inputs_embeds
        cos, sin = self.rotary_emb.vision_pe_cos, self.rotary_emb.vision_pe_sin
        position_embeddings = (cos.to(hidden_states), sin.to(hidden_states))

        # print(f"cos, sin: {cos.shape}, {sin.shape}")

        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(hidden_states, position_embeddings)

        return hidden_states

    def build(self, inputs_embeds):
        """Leap export path. See :meth:`forward` for shapes.

        Args:
            inputs_embeds: Shape ``(1, 2304, 768)`` for the 768x768 resolution.
        """
        hidden_states = inputs_embeds

        cos, sin = self.rotary_emb.vision_pe_cos, self.rotary_emb.vision_pe_sin

        cos = cos.to(device="cpu", dtype=torch.float16)
        sin = sin.to(device="cpu", dtype=torch.float16)

        position_embeddings = (cos, sin)

        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(hidden_states, position_embeddings)

        return hidden_states
