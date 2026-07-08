from typing import Optional, Tuple

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import RMSNorm
from leap_llm.nn.utils import Module

from .attention import Qwen2_5_VLAttention, Qwen2_5_VLVisionAttention
from .mlp import Qwen2_5_VLMLP, Qwen2_5_VLPatchMergerMLP


class Qwen2_5_VLDecoderLayer(Module):
    def __init__(self, config, layer_idx: int, use_plugin=False):
        super().__init__()
        self.use_plugin = use_plugin
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2_5_VLAttention(config, layer_idx, self.use_plugin)

        self.mlp = Qwen2_5_VLMLP(config, use_plugin=self.use_plugin)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_plugin=self.use_plugin,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_plugin=self.use_plugin,
        )

    def build(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_keys: torch.Tensor = None,
        cache_values: torch.Tensor = None,
    ):
        residual = hidden_states
        _, seq_len, hidden_size = hidden_states.type.shape
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_key, new_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        hidden_states = leap.add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        return hidden_states, new_key, new_value

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ):
        residual = hidden_states
        _, seq_len, hidden_size = hidden_states.shape
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_key, new_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        hidden_states = torch.add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.add(residual, hidden_states)

        return hidden_states, new_key, new_value


class Qwen2_5_VLVisionBlock(Module):
    def __init__(self, config, use_plugin) -> None:
        super().__init__()
        self.use_plugin = use_plugin
        self.norm1 = RMSNorm(
            config.hidden_size, eps=1e-6, use_plugin=self.use_plugin
        )
        self.norm2 = RMSNorm(
            config.hidden_size, eps=1e-6, use_plugin=self.use_plugin
        )
        self.attn = Qwen2_5_VLVisionAttention(
            config.hidden_size,
            num_heads=config.num_heads,
            use_plugin=self.use_plugin,
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True, use_plugin=self.use_plugin)

    def build(
        self,
        hidden_states,
        lengths,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ):
        residual = hidden_states
        hidden_states = self.attn(
            self.norm1(hidden_states),
            lengths=lengths,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        hidden_states = leap.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = leap.add(residual, hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states,
        lengths,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(
            self.norm1(hidden_states),
            lengths=lengths,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        hidden_states = torch.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = torch.add(residual, hidden_states)
        return hidden_states


class Qwen2_5_VLPatchMerger(Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        use_plugin: bool = False,
    ) -> None:
        super().__init__()
        self.use_plugin = use_plugin
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6, use_plugin=self.use_plugin)
        self.mlp = Qwen2_5_VLPatchMergerMLP(
            self.hidden_size, dim, use_plugin=self.use_plugin
        )

    def build(self, hidden_states):
        hidden_states = self.ln_q(hidden_states)
        hidden_states = leap.reshape(hidden_states, [1, -1, self.hidden_size])
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states)
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states = self.mlp(hidden_states)
        return hidden_states
