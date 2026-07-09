import torch
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.blocks.rmsnorm import Gemma4RMSNorm
from leap_llm.models.gemma4_e2b.blocks.text_attention import Gemma4TextAttention
from leap_llm.models.gemma4_e2b.blocks.text_mlp import Gemma4TextMLP
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4TextConfig
from leap_llm.nn.modules import (
    DynamicQuantLinear,
)
from leap_llm.nn.utils import Module


class Gemma4TextDecoderLayer(Module):
    """Gemma4 decoder layer with 4 norms, KV sharing, PLE, and layer_scalar."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int, logger=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.logger = logger
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))  # FIXME: is this always 1 ?

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = "gelu_pytorch_tanh"
            self.per_layer_input_gate = DynamicQuantLinear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                w_bits=config.w_bits,
                has_scale=config.has_scale,
            )
            self.per_layer_projection = DynamicQuantLinear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                w_bits=config.w_bits,
                has_scale=config.has_scale,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        per_layer_input,
        fa_mask,
        sa_mask,
        fa_position_embeddings,
        sa_position_embeddings,
        past_key=None,
        past_value=None,
        shared_kv_states=None,
    ):
        """One text decoder layer: 4 RMSNorms + (sliding|full) attention + MLP (+ optional PLE).

        Args:
            hidden_states (torch.Tensor): Decoder input. Shape:
                ``(batch_size, q_len, hidden_size)`` — e.g.
                ``(1, chunk_size, 1536)`` for prefill or
                ``(1, 1, 1536)`` for decode.
            per_layer_input (torch.Tensor | None): Optional per-layer
                PLE input. Shape: ``(batch_size, q_len, hidden_size_per_layer_input)``.
                ``None`` when ``hidden_size_per_layer_input == 0``.
            fa_mask (torch.Tensor): Additive mask for full-attention
                layers. Shape: ``(batch_size, 1, q_len, cache_len)``.
                Padding/window positions are ``-inf``.
            sa_mask (torch.Tensor): Additive mask for sliding-attention
                layers. Shape: ``(batch_size, 1, q_len, 2 * sliding_window)``.
            fa_position_embeddings (tuple[torch.Tensor, torch.Tensor]):
                Per-position ``(cos, sin)`` for full attention. Each
                tensor: ``(batch_size, q_len, head_dim)``.
            sa_position_embeddings (tuple[torch.Tensor, torch.Tensor]):
                Same for sliding attention.
            past_key (torch.Tensor | None): Cached K. Shape:
                ``(batch_size, cache_len, num_kv_heads, head_dim)``.
                ``None`` for first chunk / KV-shared layers.
            past_value (torch.Tensor | None): Cached V, same shape.
            shared_kv_states (dict | None): ``{layer_idx: (k, v)}``
                cross-layer KV cache.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                ``(hidden_states, new_key, new_value)``.
                - ``hidden_states``: ``(batch_size, q_len, hidden_size)``.
                - ``new_key`` / ``new_value``:
                  ``(batch_size, q_len, num_kv_heads, head_dim)`` for
                  the cache (``None`` for KV-shared layers).
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.config.layer_types[self.layer_idx] == "full_attention":
            hidden_states, new_key, new_value = self.self_attn(
                hidden_states,
                fa_mask,
                fa_position_embeddings,
                past_key,
                past_value,
                shared_kv_states=shared_kv_states,
            )
        else:
            hidden_states, new_key, new_value = self.self_attn(
                hidden_states,
                sa_mask,
                sa_position_embeddings,
                past_key,
                past_value,
                shared_kv_states=shared_kv_states,
            )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = F.gelu(hidden_states, approximate="tanh")
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar

        return hidden_states, new_key, new_value

    def build(
        self,
        hidden_states,
        per_layer_input,
        fa_mask,
        sa_mask,
        fa_position_embeddings,
        sa_position_embeddings,
        past_key=None,
        past_value=None,
        shared_kv_states: dict = None,
    ):
        """Build method using leap operations for NPU compilation."""

        # print(f"Building Layer {self.layer_idx}")

        # Pre-attention layernorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Choose mask and position embeddings based on layer type
        if self.config.layer_types[self.layer_idx] == "full_attention":
            hidden_states, new_key, new_value, store_key, store_value = self.self_attn(
                hidden_states,
                fa_position_embeddings,
                fa_mask,
                shared_kv_states.get(self.self_attn.kv_shared_layer_index, (None, None))[0]
                if shared_kv_states
                else None,
                shared_kv_states.get(self.self_attn.kv_shared_layer_index, (None, None))[1]
                if shared_kv_states
                else None,
                past_key,
                past_value,
            )
            if (store_key is not None) and (store_value is not None) and self.self_attn.store_full_length_kv:
                shared_kv_states[self.layer_idx] = (store_key, store_value)
        else:
            hidden_states, new_key, new_value, store_key, store_value = self.self_attn(
                hidden_states,
                sa_position_embeddings,
                sa_mask,
                shared_kv_states.get(self.self_attn.kv_shared_layer_index, (None, None))[0]
                if shared_kv_states
                else None,
                shared_kv_states.get(self.self_attn.kv_shared_layer_index, (None, None))[1]
                if shared_kv_states
                else None,
                past_key,
                past_value,
            )
            if (store_key is not None) and (store_value is not None) and self.self_attn.store_full_length_kv:
                shared_kv_states[self.layer_idx] = (store_key, store_value)

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = leap.add(hidden_states, residual)
        # print(f"after attention: {hidden_states.type.shape}")
        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = leap.add(hidden_states, residual)
        # print(f"after mlp: {hidden_states.type.shape}")

        # PLE block
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            ple_gate = self.per_layer_input_gate(hidden_states)
            ple_act = leap.gelu(ple_gate, approximate="tanh")
            ple_combined = leap.mul(ple_act, per_layer_input)
            ple_proj = self.per_layer_projection(ple_combined)
            ple_normed = self.post_per_layer_input_norm(ple_proj)
            hidden_states = leap.add(ple_normed, residual)
            # print(f"after ple: {hidden_states.type.shape}")

        # layer_scalar
        if self.layer_scalar is not None and hasattr(self.layer_scalar, "item"):
            scalar_val = self.layer_scalar.item()
            if scalar_val != 1.0:
                hidden_states = leap.mul(hidden_states, scalar_val)
                # print(f"after layer scalar: {hidden_states.type.shape}")

        return hidden_states, new_key, new_value
