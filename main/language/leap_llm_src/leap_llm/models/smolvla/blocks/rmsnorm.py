"""Llama-style RMSNorm for SmolLM2 (not Gemma's (1 + weight) form)."""

import math

import torch
from hbdk4.compiler import leap

from leap_llm.nn.utils import Module


class SmolLM2RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(dim))
        self.scale = 1.0
        self.summax_hidden = None
        self.register_buffer("i_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("i_scale_pow", torch.tensor(1.0), persistent=False)
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
        # B30 VPU RMSNorm: batch dim must be power-of-2 or multiple of 64.
        # chunk_size=50 fails; use manual path like Pi0 GemmaRMSNorm fallback.
        if seq_len % 32 == 0 or seq_len == 1:
            return leap.rms_norm(x, [-1], eps, weight=weight)
        variance = leap.reduce_mean(leap.pow(x, 2), [-1])
        hidden_states = leap.mul(x, leap.rsqrt(leap.add(variance, eps)))
        return leap.mul(weight, hidden_states)

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        h_pow = torch.sum(hidden_states**2, dim=-1)
        curr_absmax = h_pow.max()
        if self.summax_hidden is None or curr_absmax > self.summax_hidden:
            self.summax_hidden = curr_absmax
        raw_scale = math.sqrt(self.summax_hidden / self.max_float16) * 2
        self.scale = raw_scale if raw_scale > 1.0 else 1.0
        self.i_scale = torch.tensor(1 / self.scale)
        self.i_scale_pow = torch.tensor(1 / (self.scale * self.scale))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


__all__ = ["SmolLM2RMSNorm"]
