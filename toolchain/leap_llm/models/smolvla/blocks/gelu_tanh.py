"""GELU activation matching transformers gelu_pytorch_tanh."""

import torch
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.nn.modules.const_fake_quant import ConstFakeQuant
from leap_llm.nn.utils import Module


class GELUTanh(Module):
    """Same as transformers ACT2FN['gelu_pytorch_tanh'] with FakeQuant output."""

    def __init__(self, quantized: bool = True, quant_bits: int = 16):
        super().__init__()
        self.quantized = quantized
        self.quant_bits = quant_bits
        self.out_quant = ConstFakeQuant(quant_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(x, approximate="tanh")
        if self.quantized:
            out = self.out_quant(out)
        return out

    def build(self, x):
        out = leap.gelu(x, approximate="tanh")
        if self.quantized:
            out = self.out_quant(out)
        return out
