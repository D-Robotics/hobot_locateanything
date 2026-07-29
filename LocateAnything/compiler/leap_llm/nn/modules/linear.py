import torch
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant


class FakeQuantLinear(Module):
    """ColumnParallelLinear & RowParallelLinear"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_bits: int = 16,
        w_bits: int = 8,
        x_input_bits: int = 16,
        has_scale: bool = False,
    ) -> None:
        super().__init__()

        self.absmax_weight = None

        row_parallel_size = 1
        self.in_features = in_features // row_parallel_size
        column_parallel_size = 1
        self.out_features = out_features // column_parallel_size
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.w_bits = w_bits
        self.has_scale = has_scale
        if has_scale:
            self.register_buffer("scales", torch.ones(self.out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.bias = None

        self.x_fake_quant = ConstFakeQuant(x_input_bits)

        self.quant_bits = quant_bits
        self.out_fake_quant = ConstFakeQuant(self.quant_bits)

    def build(self, x):
        x_quant = self.x_fake_quant(x)
        weight_min = (-self.absmax_weight).tolist()
        weight_max = self.absmax_weight.tolist()

        weight_data = self.weight.data.to(torch.float32)
        bias = self.bias.data.to(torch.float32) if self.bias is not None else None

        # name = self.get_full_name()

        if self.w_bits == 8:
            # print("=== name:", name, self.w_bits)
            weight_quant = leap.const_fake_quant(
                weight_data,
                weight_min,
                weight_max,
                8,
                True,
                axis=0,
            )
        elif self.w_bits == 4:
            # print("=== name:", name, self.w_bits)
            qmax = 2 ** (self.w_bits - 1) - 1  # 7
            qmin = -qmax  # -7

            scale_list = self.scales.flatten().tolist()
            zeros = [0] * len(scale_list)
            q_weight = torch.clip(torch.round(weight_data / self.scales), qmin, qmax)
            q_weight = q_weight.to(torch.int8)
            # print("=== q_weight:", q_weight.min().item(), q_weight.max().item())
            weight_quant = leap.dequantize(
                q_weight, scales=scale_list, zeros=zeros, axis=0
            )

        out = leap.linear(
            x_quant,
            weight_quant,
            bias=bias,
        )
        out = self.out_fake_quant(out)
        return out

    def forward(self, x):
        x = self.x_fake_quant(x)
        weight = self.weight.data

        if x.dtype != weight.dtype:
            x = x.to(weight.data)

        if self.absmax_weight is None:
            # per_channel_max 形状：(1536,)
            per_channel_max, _ = torch.max(weight.abs(), dim=1)
            self.absmax_weight = per_channel_max

        bias = self.bias.data if self.bias is not None else None
        out = torch.nn.functional.linear(x, weight, bias=bias)
        out = self.out_fake_quant(out)
        return out


class DynamicQuantLinear(Module):
    """ColumnParallelLinear & RowParallelLinear"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bits: int = 8,
        has_scale: bool = False,
    ) -> None:
        super().__init__()
        row_parallel_size = 1
        self.in_features = in_features // row_parallel_size
        column_parallel_size = 1
        self.out_features = out_features // column_parallel_size
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.w_bits = w_bits
        self.has_scale = has_scale
        if has_scale:
            self.register_buffer("scales", torch.zeros(self.out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.bias = None

    def build(self, x):
        x_q, x_s = leap.dynamic_quantize(x, blockSize=-1)

        int_max = 2 ** (self.w_bits - 1) - 1
        if self.has_scale:
            w_s = self.scales
        else:
            w_max = self.weight.data.abs().max(dim=-1, keepdim=True).values
            w_s = w_max / int_max
        w_q = torch.round(self.weight.data / w_s)
        w_q = torch.clamp(w_q, -int_max - 1, int_max).to(torch.int8)

        v_f = leap.block_quantized_matmul(x_q, w_q, x_s, w_s, mmaAlpha=1024.0)
        if self.bias is not None:
            v_f = leap.add(v_f, self.bias.data)
        return v_f

    def forward(self, x):
        x = nn.functional.linear(
            x, self.weight.data, bias=self.bias.data if self.bias is not None else None
        )
        return x
