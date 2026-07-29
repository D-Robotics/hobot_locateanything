import torch
from hbdk4.compiler import leap

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant


class FakeQuantAdd(Module):
    def __init__(self, quant_bits: int = 16, quantized=False):
        super().__init__()
        self.quant_bits = quant_bits
        self.out_fake_quant = ConstFakeQuant(self.quant_bits)
        self.quantized = quantized

    def build(self, x, y):
        if not self.quantized:
            return leap.add(x, y)

        out = leap.add(x, y)
        out = self.out_fake_quant(out)
        return out

    def forward(self, x, y):

        if not self.quantized:
            return torch.add(x, y)

        out = torch.add(x, y)
        out = self.out_fake_quant(out)
        return out


class FakeQuantMul(Module):
    def __init__(self, quantized=False):
        super().__init__()
        self.out_fake_quant = ConstFakeQuant(16)
        self.quantized = quantized

    def build(self, x, y):

        if not self.quantized:
            return leap.mul(x, y)

        out = leap.mul(x, y)
        out = self.out_fake_quant(out)
        return out

    def forward(self, x, y):

        if not self.quantized:
            return torch.mul(x, y)

        out = torch.mul(x, y)
        out = self.out_fake_quant(out)
        return out


class FakeQuantRsqrt(Module):
    def __init__(self, quantized=True):
        super().__init__()
        self.out_fake_quant = ConstFakeQuant(16)
        self.quantized = quantized

    def build(self, x):
        if not self.quantized:
            return leap.rsqrt(x)

        out = leap.rsqrt(x)
        out = self.out_fake_quant(out)
        return out

    def forward(self, x: torch.Tensor):
        if not self.quantized:
            return x.rsqrt()

        out = x.rsqrt()
        out = self.out_fake_quant(out)
        return out


class FakeQuantReduceMean(Module):
    def __init__(self, quantized=True):
        super().__init__()
        self.out_fake_quant = ConstFakeQuant(16)
        self.quantized = quantized

    def build(self, x, dim=[-1], keepdim=True):
        if not self.quantized:
            return leap.reduce_mean(x, dim)

        out = leap.reduce_mean(x, dim)
        out = self.out_fake_quant(out)
        return out

    def forward(self, x: torch.Tensor, dim=-1, keepdim=True):
        if not self.quantized:
            return x.mean(dim, keepdim=keepdim)

        out = x.mean(dim, keepdim=keepdim)
        out = self.out_fake_quant(out)
        return out


class FakeQuantPow(Module):
    def __init__(self, quantized=True):
        super().__init__()
        self.out_fake_quant = ConstFakeQuant(16)
        self.quantized = quantized

    def build(self, x, exponent=None):
        if not self.quantized:
            return leap.pow(x, exponent)
        # output_type=leap.float16
        out = leap.pow(x, exponent)
        out = self.out_fake_quant(out)
        return out

    def forward(self, x: torch.Tensor, exponent=None):
        if not self.quantized:
            return x.pow(exponent)
        # output_type=leap.float16
        out = x.pow(exponent)
        out = self.out_fake_quant(out)
        return out


class Clip(Module):
    def __init__(self):
        super().__init__()

    def build(self, x, min, max):
        out = leap.clip(x, min, max)
        return out

    def forward(self, x: torch.Tensor, min, max):
        return torch.clamp(x, min, max)
