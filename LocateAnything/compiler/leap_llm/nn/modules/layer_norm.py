import math

import torch
import torch.nn.functional as F
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.utils import Module


class LayerNorm(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(dim))
            if bias:
                self.bias = nn.Parameter(torch.empty(dim))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def build(self, x):
        # x = leap.cast_type(x, output_type=leap.float32)
        if self.elementwise_affine:
            bias = leap.reshape(self.bias.data, [1, 1, self.bias.shape[-1]])
            weight = leap.reshape(self.weight.data, [1, 1, self.weight.shape[-1]])
            res = leap.layernorm(x, [-1], self.eps, bias=bias, weight=weight)
        else:
            res = leap.layernorm(x, [-1], self.eps)
        return res

    def forward(self, hidden_states: torch.Tensor):
        weight = self.weight.data if self.elementwise_affine else None
        bias = self.bias.data if self.elementwise_affine and self.bias is not None else None
        h_norm = F.layer_norm(
            hidden_states,
            normalized_shape=[hidden_states.shape[-1]],
            weight=weight,
            bias=bias,
            eps=self.eps,
        )
        return h_norm


class LayerNormSplit(Module):
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None

        self.summax_hidden = None
        self.max_float16 = 65504.0
        self.scale = 1.0
        self.i_scale = 1.0

    def build(self, x):

        x = leap.cast_type(x, output_type=leap.float16)

        weight = self.weight.data.to(torch.float16)
        bias = self.bias.data.to(torch.float16)

        input = leap.mul(x, self.i_scale)
        mean = leap.reduce_mean(input, [-1])
        # hidden_sub = leap.sub(input, mean)
        hidden_sub = leap.add(input, leap.mul(mean, -1))
        pow_ = leap.pow(hidden_sub, 2)
        var = leap.reduce_mean(pow_, [-1])
        eps_scaled = self.eps * (self.i_scale * self.i_scale)

        var_add = leap.add(var, eps_scaled)
        inv_std = leap.rsqrt(var_add)
        normalized = leap.mul(hidden_sub, inv_std)
        output = leap.mul(normalized, weight)
        output = leap.add(output, bias)

        output = leap.cast_type(output, output_type=leap.float32)

        return output

    def forward(self, hidden_states: torch.Tensor):

        orig_dtype = hidden_states.dtype

        hidden_fp32_for_pow = hidden_states.to(torch.float32)

        h_pow = torch.sum(hidden_fp32_for_pow**2, dim=-1)  # float32

        curr_absmax = h_pow.max().item()
        if (self.summax_hidden is None) or (curr_absmax > self.summax_hidden):
            self.summax_hidden = curr_absmax

        raw_scale = math.sqrt(self.summax_hidden / self.max_float16) * 2.0
        self.scale = raw_scale if (raw_scale > 1.0) else 1.0
        self.i_scale = 1.0 / self.scale

        hidden_fp32 = hidden_states.to(torch.float32) * self.i_scale

        mean = hidden_fp32.mean(dim=-1, keepdim=True)
        var = ((hidden_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)

        eps_scaled = self.eps * (self.i_scale * self.i_scale)
        inv_std = torch.rsqrt(var + eps_scaled)

        normalized = (hidden_fp32 - mean) * inv_std

        weight_fp32 = self.weight.to(hidden_fp32.device)
        bias_fp32 = self.bias.to(hidden_fp32.device)
        out_fp32 = normalized * weight_fp32 + bias_fp32

        out = out_fp32.to(orig_dtype)

        return out
