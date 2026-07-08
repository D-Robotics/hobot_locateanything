import torch
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4VisionConfig,
)
from leap_llm.nn.modules import (
    DynamicQuantLinear,
)
from leap_llm.nn.utils import Module


class Gemma4ClippableLinear(Module):
    def __init__(
        self,
        config: Gemma4VisionConfig | Gemma4AudioConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = DynamicQuantLinear(in_features, out_features, bias=False)

        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Optional input/output clipping around a ``DynamicQuantLinear``.

        Args:
            hidden_states (torch.Tensor): Last dim must match
                ``in_features``. Shape is therefore
                ``(..., in_features)``.

        Returns:
            torch.Tensor: Linear output. Shape is
                ``(..., out_features)``.
        """
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)

        hidden_states = self.linear(hidden_states)

        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)

        return hidden_states

    def build(self, hidden_states):
        """Leap export path. See :meth:`forward` for shapes.

        Args:
            hidden_states: Shape ``(..., in_features)``.
        """
        if self.use_clipped_linears:
            hidden_states = leap.clip(
                hidden_states,
                self.input_min.item(),
                self.input_max.item(),
            )

        hidden_states = self.linear(hidden_states)

        if self.use_clipped_linears:
            hidden_states = leap.clip(
                hidden_states,
                self.output_min.item(),
                self.output_max.item(),
            )

        return hidden_states
