import torch
from hbdk4.compiler import leap

from typing import Optional, Tuple
import torch
from hbdk4.compiler import leap
from leap_llm.nn.modules import RMSNorm
from leap_llm.nn.utils import Module

from . import MLP, Attention


class DecoderLayer(Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        use_plugin=False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config, use_plugin=use_plugin)
        self.layer_idx = layer_idx
        # self.attention_type = config.layer_types[layer_idx]

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, use_plugin=use_plugin
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, use_plugin=use_plugin
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

        hidden_states, new_key, new_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        hidden_states = leap.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(residual, hidden_states)
        return hidden_states, new_key, new_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_keys: torch.Tensor = None,
        cache_values: torch.Tensor = None,
    ):
        residual = hidden_states
        _, seq_len, hidden_size = hidden_states.shape
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, new_key, new_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )

        hidden_states = torch.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.add(residual, hidden_states)
        return hidden_states, new_key, new_value
