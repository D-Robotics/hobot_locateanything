import functools

import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.blocks.linear import Gemma4ClippableLinear
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4VisionConfig
from leap_llm.nn.utils import Module


class Gemma4VisionMLP(Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Gemma4ClippableLinear(config, self.hidden_size, self.intermediate_size)
        self.up_proj = Gemma4ClippableLinear(config, self.hidden_size, self.intermediate_size)
        self.down_proj = Gemma4ClippableLinear(config, self.intermediate_size, self.hidden_size)
        self.act_fn = functools.partial(F.gelu, approximate="tanh")

    def forward(self, x):
        """SwiGLU/GeGLU MLP used inside each vision encoder layer.

        Args:
            x (torch.Tensor): Hidden states. Shape:
                ``(batch_size, num_patches, hidden_size)``, e.g.
                ``(batch_size, 2304, 768)`` for the 768x768 path.

        Returns:
            torch.Tensor: MLP output. Shape:
                ``(batch_size, num_patches, hidden_size)``, e.g.
                ``(batch_size, 2304, 768)``.
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def build(self, x):
        """Leap export path. See :meth:`forward` for shapes."""
        x_gated = self.gate_proj(x)
        x_act = leap.gelu(x_gated, approximate="tanh")
        x_up = self.up_proj(x)
        x_hidden = leap.mul(x_act, x_up)
        x_down = self.down_proj(x_hidden)
        return x_down
