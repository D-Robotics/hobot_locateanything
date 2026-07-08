"""SmolVLM2 vision encoder layers (HF-compatible MLP activation)."""


import torch
from hbdk4.compiler import leap

from leap_llm.models.pi0.model_siglip import SiglipAttention
from leap_llm.models.smolvla.blocks.configuration_smolvlm import SmolVLMVisionConfig
from leap_llm.models.smolvla.blocks.gelu_tanh import GELUTanh
from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.modules.layer_norm import LayerNorm
from leap_llm.nn.utils import Module


class SmolVLMVisionMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = DynamicQuantLinear(config.hidden_size, config.intermediate_size)
        self.fc2 = DynamicQuantLinear(config.intermediate_size, config.hidden_size)
        self.activation_fn = GELUTanh()

    def build(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class SmolVLMVisionEncoderLayer(Module):
    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SmolVLMVisionMLP(config)

    def build(self, hidden_states, output_attentions=False):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = leap.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(residual, hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool | None = False,
    ) -> tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class SmolVLMVisionEncoder(Module):
    def __init__(self, config: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [SmolVLMVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def build(self, inputs_embeds):
        hidden_states = inputs_embeds
        attn_weight = None
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
            attn_weight = layer_outputs[1]
        return hidden_states, attn_weight

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        attn_weight = None
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
            attn_weight = layer_outputs[1]
        return hidden_states, attn_weight
