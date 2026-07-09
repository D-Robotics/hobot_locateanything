import torch
from hbdk4.compiler import leap

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant


class FakeQuantSoftmax(Module):
    def __init__(
        self,
        quant_bits: int = 16,
        quantized: bool = True,
    ) -> None:
        super().__init__()
        self.quant_bits = quant_bits
        self.quantized = quantized
        self.out_quant = ConstFakeQuant(quant_bits)

    def build(self, x):
        if self.quantized:
            out = leap.softmax(x, -1)
            out = self.out_quant(out)
            return out

        return leap.softmax(x, -1)

    def forward(self, x):
        out = torch.softmax(x, -1)
        out = self.out_quant(out)
        return out


class FakeQuantSwish(Module):
    def __init__(
        self,
        quantized: bool = True,
        quant_bits: int = 16,
    ) -> None:
        super().__init__()
        self.quant_bits = quant_bits
        self.out_quant = ConstFakeQuant(quant_bits)

        self.act_fn = torch.nn.functional.silu
        self.quantized = quantized

    def build(self, x):
        if self.quantized:
            x = leap.swish(x)
            x = self.out_quant(x)
            return x
        return leap.swish(x)

    def forward(self, x):
        if self.quantized:
            x = self.act_fn(x)
            x = self.out_quant(x)
            return x

        return self.act_fn(x)


class FakeQuantGELU(Module):
    def __init__(
        self,
        quantized: bool = True,
        quant_bits: int = 16,
    ) -> None:
        super().__init__()
        self.quant_bits = quant_bits
        self.out_quant = ConstFakeQuant(quant_bits)

        self.act_fn = torch.nn.functional.gelu
        self.quantized = quantized

    def build(self, x):
        if self.quantized:
            out = leap.gelu(x)
            out = self.out_quant(out)
            return out

        return leap.gelu(x)

    def forward(self, x):
        if self.quantized:
            out = self.act_fn(x)
            out = self.out_quant(out)
            return out
        out = self.act_fn(x)
        return out


class FakeQuantTanh(Module):
    def __init__(
        self,
        quantized: bool = True,
        quant_bits: int = 16,
    ) -> None:
        super().__init__()
        self.quant_bits = quant_bits
        self.in_quant = ConstFakeQuant(quant_bits)
        # self.out_quant = ConstFakeQuant(quant_bits)

        self.act_fn = torch.nn.functional.tanh
        self.quantized = quantized

    def build(self, x):
        if self.quantized:
            x = self.in_quant(x)
            out = leap.tanh(x)
            # out = self.out_quant(out)
            return out

        return leap.tanh(x)

    def forward(self, x):
        if self.quantized:
            x = self.in_quant(x)
            out = self.act_fn(x)
            # out = self.out_quant(out)
            return out
        out = self.act_fn(x)
        return out
