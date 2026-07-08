import math

import torch
from hbdk4.compiler import leap

from leap_llm.nn.utils import Module


class GemmaRMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(dim))
        self.scale = 1.0
        i_scale = torch.tensor(1.0)
        i_scale_pow = torch.tensor(1.0)
        self.summax_hidden = None
        self.register_buffer("i_scale", i_scale, persistent=False)
        self.register_buffer("i_scale_pow", i_scale_pow, persistent=False)
        # max float16 sqrt
        self.max_float16 = 65504.0

    def build(self, x):
        i_scale = self.i_scale.item()
        i_scale_pow = self.i_scale_pow.item()
        x = leap.mul(x, i_scale)
        eps = self.eps * i_scale_pow
        ndim = len(x.type.shape)
        if ndim == 3:
            weight = leap.reshape(self.weight.data, [1, 1, self.weight.shape[-1]])
            seq_len = x.type.shape[1]
        else:
            weight = leap.reshape(self.weight.data, [1, self.weight.shape[-1]])
            seq_len = x.type.shape[0]
        if seq_len % 32 == 0 or seq_len == 1:
            output = leap.rms_norm(x, [-1], eps, weight=leap.add(weight, 1.0))
        else:
            squared = leap.pow(x, 2)
            variance = leap.reduce_mean(squared, [-1])

            adjusted_variance = leap.add(variance, eps)
            inv_sqrt = leap.rsqrt(adjusted_variance)
            hidden_states = leap.mul(x, inv_sqrt)
            output = leap.mul(1.0 + self.weight.data, hidden_states)
        return output

    def forward(self, hidden_states: torch.Tensor):
        # for caculate scale
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        h_pow = torch.sum(hidden_states**2, dim=-1)
        curr_absmax = h_pow.max()
        # 更新全局峰值
        if self.summax_hidden is None or curr_absmax > self.summax_hidden:
            self.summax_hidden = curr_absmax
        # 2) 动态算出 raw scale, mul 2 for more robust
        raw_scale = math.sqrt(self.summax_hidden / self.max_float16) * 2
        self.scale = raw_scale if raw_scale > 1.0 else 1.0
        self.i_scale = torch.tensor(1 / self.scale)
        self.i_scale_pow = torch.tensor(1 / (self.scale * self.scale))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        output = (self.weight + 1.0) * hidden_states
        output = output.to(dtype)
        return output
