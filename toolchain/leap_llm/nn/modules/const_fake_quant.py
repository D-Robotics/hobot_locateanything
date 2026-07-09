import torch
from hbdk4.compiler import leap

from leap_llm.nn.utils import Module


class ConstFakeQuant(Module):
    def __init__(self, bits=8, quantized=True) -> None:
        super().__init__()
        self.bits = bits
        absmax = torch.tensor(0.0)
        self.quantized = quantized
        self.register_buffer("absmax", absmax, persistent=False)

    def build(self, x):
        if not self.quantized:
            return x
        # print(f"build absmax: {self.absmax.item()}, self.bits: {self.bits}")
        x = leap.const_fake_quant(
            x,
            [-self.absmax.item()],
            [self.absmax.item()],
            self.bits,
            True,
        )
        return x

    def forward(self, x):
        if not self.quantized:
            return x
        curr_absmax = x.abs().max()
        self.absmax = torch.maximum(self.absmax, curr_absmax)
        # print(f"absmax: {self.absmax.item()}")
        return x
