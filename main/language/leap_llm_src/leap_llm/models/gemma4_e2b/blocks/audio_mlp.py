import torch
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.blocks.linear import Gemma4ClippableLinear
from leap_llm.models.gemma4_e2b.blocks.rmsnorm import Gemma4RMSNorm
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4AudioConfig
from leap_llm.nn.utils import Module


class Gemma4AudioFeedForward(Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config

        self.ffw_layer_1 = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 4)
        self.ffw_layer_2 = Gemma4ClippableLinear(config, config.hidden_size * 4, config.hidden_size)

        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.act_fn = F.silu

        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

        self.clipping_absmax = min(
            self.gradient_clipping,
            torch.finfo(self.ffw_layer_1.linear.weight.dtype).max - 1000,  # FIXME: for avoiding quantization overflow
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Audio feed-forward block (SiLU/SwiGLU) with gradient clipping + residual.

        Args:
            hidden_states (torch.Tensor): Audio hidden states. Shape:
                ``(batch_size, seq_len, hidden_size)``. For the default
                config after the conv-subsample projection:
                ``(batch_size, 750, 1024)``.

        Returns:
            torch.Tensor: Output. Same shape as ``hidden_states``.
        """
        # This is needed to avoid any underflow/overflow issues when clipping
        # gradient_clipping = min(self.gradient_clipping, torch.finfo(self.ffw_layer_1.linear.weight.dtype).max)

        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states *= self.post_layer_scale
        hidden_states += residual

        return hidden_states

    def build(self, hidden_states):
        """Leap export path. See :meth:`forward` for shapes."""
        residual = hidden_states
        hidden_states = leap.clip(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = leap.swish(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = leap.clip(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states = leap.mul(hidden_states, self.post_layer_scale)
        hidden_states = leap.add(hidden_states, residual)

        return hidden_states
