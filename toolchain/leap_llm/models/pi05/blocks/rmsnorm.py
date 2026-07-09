import math

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class GemmaRMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int = None):
        super().__init__()
        self.eps = eps
        if cond_dim is not None:
            #self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = DynamicQuantLinear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            self.dense.weight.data.zero_()
        else:
            self.weight = torch.nn.Parameter(torch.empty(dim))
            self.dense = None
        self.scale = 1.0
        i_scale = torch.tensor(1.0)
        i_scale_pow = torch.tensor(1.0)
        self.summax_hidden = None
        self.register_buffer("i_scale", i_scale, persistent=False)
        self.register_buffer("i_scale_pow", i_scale_pow, persistent=False)
        # max float16 sqrt
        self.max_float16 = 65504.0

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs
        
    def build(self, x, cond=None):
        i_scale = self.i_scale.item()
        i_scale_pow = self.i_scale_pow.item()
        x = leap.mul(x, i_scale)
        eps = self.eps * i_scale_pow
        ndim = len(x.type.shape)

        if self.dense is not None:
            modulation = self.dense(cond)
            modulation = leap.reshape(modulation, [1, 1, 1024*3])
            scale = leap.slice(modulation, [0, 0, 0], [1, 1, 1024], [1, 1, 1])
            shift = leap.slice(modulation, [0, 0, 1024], [1, 1, 2048], [1, 1, 1])
            gate = leap.slice(modulation, [0, 0, 2048], [1, 1, 3072], [1, 1, 1])
            if ndim == 3:
                seq_len = x.type.shape[1]
            else:
                seq_len = x.type.shape[0]
            if seq_len % 32 == 0 or seq_len == 1:
                output = leap.rms_norm(x, [-1], eps, weight=leap.add(scale, 1.0))
                output = leap.add(output, shift)
            else:
                squared = leap.pow(x, 2)
                variance = leap.reduce_mean(squared, [-1])
                adjusted_variance = leap.add(variance, eps)
                inv_sqrt = leap.rsqrt(adjusted_variance)
                hidden_states = leap.mul(x, inv_sqrt)
                output = leap.mul(leap.add(scale, 1.0), hidden_states)
                output = leap.add(output, shift)
            return output, gate
        else:
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


    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        # for caculate scale
        hidden_states = x.to(torch.float32)
        h_pow = torch.sum(hidden_states**2, dim=-1)
        curr_absmax = h_pow.max()
        if self.summax_hidden is None or curr_absmax > self.summax_hidden:
            self.summax_hidden = curr_absmax
        raw_scale = math.sqrt(self.summax_hidden / self.max_float16) * 2
        self.scale = raw_scale if raw_scale > 1.0 else 1.0
        self.i_scale = torch.tensor(1 / self.scale)
        self.i_scale_pow = torch.tensor(1 / (self.scale * self.scale))

        dtype = x.dtype
        normed_inputs = self._norm(x)

        if self.dense is not None:
            modulation = self.dense(cond)
            if len(x.shape) == 3:
                modulation = modulation.unsqueeze(1)
            scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
            normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
            return normed_inputs.to(dtype), gate.to(dtype)
        else:
            output = (self.weight.float() + 1.0) * normed_inputs
            return output.to(dtype)