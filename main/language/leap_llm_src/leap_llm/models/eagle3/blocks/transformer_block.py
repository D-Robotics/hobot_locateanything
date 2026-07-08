from typing import Optional, Tuple

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import RMSNorm
from leap_llm.nn.utils import Module

from . import Eagle3Attention, MLP


class Eagle3DecoderLayer(Module):
    """EAGLE3 draft model decoder layer.

    Unlike standard Qwen3 DecoderLayer, this layer:
    - Takes two inputs: input_emb (token embeddings) and hidden_states
    - Applies separate RMSNorm to each (input_layernorm for emb, hidden_norm for hidden)
    - Concatenates them along the last dim before attention: [emb, hidden] -> dim=hidden*2
    - Uses residual from original (pre-norm) hidden_states
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Eagle3Attention(config)
        self.mlp = MLP(config)

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def build(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_keys: torch.Tensor = None,
        cache_values: torch.Tensor = None,
    ):
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = leap.concat([input_emb, hidden_states], -1)

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
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_keys: torch.Tensor = None,
        cache_values: torch.Tensor = None,
    ):
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat([input_emb, hidden_states], dim=-1)

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
