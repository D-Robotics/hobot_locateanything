import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4TextConfig
from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class Gemma4TextMLP(Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = DynamicQuantLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = DynamicQuantLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = DynamicQuantLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = "gelu_pytorch_tanh"

    def forward(self, x):
        """GeGLU MLP used inside each text decoder layer.

        Args:
            x (torch.Tensor): Hidden states. Shape:
                ``(batch_size, seq_len, hidden_size)``, e.g.
                ``(1, chunk_size, 1536)`` for prefill or ``(1, 1, 1536)``
                for decode.

        Returns:
            torch.Tensor: MLP output. Shape:
                ``(batch_size, seq_len, hidden_size)``, e.g.
                ``(1, chunk_size, 1536)`` for prefill or
                ``(1, 1, 1536)`` for decode.
        """
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

    def build(self, x):
        """Leap export path. See :meth:`forward` for shapes."""
        gate = leap.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        return self.down_proj(leap.mul(gate, up))
