import torch
from hbdk4.compiler import leap

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant


class FakeQuantMatmul(Module):
    def __init__(self, x_bits=8, y_bits=8, out_bits=16):
        super().__init__()
        self.x_bits = x_bits
        self.y_bits = y_bits

        self.x_fake_quant = ConstFakeQuant(self.x_bits)
        self.y_fake_quant = ConstFakeQuant(self.y_bits)
        if out_bits is not None:
            self.out_fake_quant = ConstFakeQuant(out_bits)
        else:
            self.out_fake_quant = None

    def build(self, x, y):
        if self.x_bits:
            x = self.x_fake_quant(x)
        if self.y_bits:
            y = self.y_fake_quant(y)
        out = leap.matmul(x, y)
        if self.out_fake_quant is not None:
            out = self.out_fake_quant(out)
        return out

    def forward(self, x, y):
        if self.x_bits:
            x = self.x_fake_quant(x)
        if self.y_bits:
            y = self.y_fake_quant(y)
        out = torch.matmul(x, y)
        if self.out_fake_quant is not None:
            out = self.out_fake_quant(out)
        return out


class DynamicQuantMatmul(Module):
    def __init__(self) -> None:
        super().__init__()

    def build(self, x, y):
        # return leap.matmul(x, y)
        x_q, x_s = leap.dynamic_quantize(x, blockSize=-1)
        y_q, y_s = leap.dynamic_quantize(y, blockSize=-1)
        v_f = leap.block_quantized_matmul(x_q, y_q, x_s, y_s, mmaAlpha=1024.0)
        return v_f

    def forward(self, x, y):
        return torch.matmul(x, y)
