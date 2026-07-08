import math

import torch
import torch.nn as nn
from hbdk4.compiler import leap

from leap_llm.nn.utils import Module


class Gemma4RMSNorm(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        with_scale: bool = True,
        use_plugin: bool = False,
    ):
        super().__init__()
        self.use_plugin = use_plugin
        self.eps = eps
        self.with_scale = with_scale
        # if self.with_scale:
        #     self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.scale = 1.0
        i_scale = torch.tensor(1.0)
        i_scale_pow = torch.tensor(1.0)
        self.summax_hidden = None
        self.register_buffer("i_scale", i_scale, persistent=False)
        self.register_buffer("i_scale_pow", i_scale_pow, persistent=False)
        # max float16 sqrt
        self.max_float16 = 65504.0

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to addess
        # compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm in float32, then cast back to the input dtype.

        Also updates the per-module ``i_scale`` / ``i_scale_pow`` calibration
        statistics that the leap ``build`` path consumes.

        Args:
            hidden_states (torch.Tensor): Input tensor with the
                normalization dim at ``-1``. Shape is one of:
                ``(..., dim)`` — e.g. ``(batch, seq, hidden_size)`` for
                the LLM/vision/audio paths or ``(1, 1, dim)`` for a
                single-vector norm.

        Returns:
            torch.Tensor: Normalized output with the same shape and dtype
                as ``hidden_states``.
        """
        # Match HF: compute norm in float32, cast back to input dtype
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()

        # Track calibration stats for build() (i_scale/i_scale_pow)
        h_pow = torch.sum(hidden_states.float() ** 2, dim=-1)
        curr_absmax = h_pow.max()
        if (self.summax_hidden is None) or (curr_absmax > self.summax_hidden):
            self.summax_hidden = curr_absmax
        raw_scale = math.sqrt(self.summax_hidden / self.max_float16) * 2
        self.scale = raw_scale if raw_scale > 1.0 else 1.0
        self.i_scale = torch.tensor(1 / self.scale)
        self.i_scale_pow = torch.tensor(1 / (self.scale**2))

        return normed_output.type_as(hidden_states)

    def build(self, x):
        i_scale = self.i_scale.item()
        i_scale_pow = self.i_scale_pow.item()
        x = leap.mul(x, i_scale)
        eps = self.eps * i_scale_pow
        ndim = len(x.type.shape)
        padding_size = None
        # if self.with_scale:
        if ndim == 3:
            weight = leap.reshape(self.weight.data, [1, 1, self.weight.shape[-1]])
            seq_len = x.type.shape[1]
        elif ndim == 4:
            weight = leap.reshape(self.weight.data, [1, 1, 1, self.weight.shape[-1]])
            seq_len = x.type.shape[1]
            if seq_len == 1:
                bs, seq_len, h, w = x.type.shape
                if h % 32 != 0:
                    print(f"padding h to be 32 divisible: {h} -> {32 - h % 32 + h}")
                    padding_size = 32 - h % 32
                    padding_zeros = (
                        torch.zeros((bs, seq_len, padding_size, w), dtype=torch.float16).contiguous().numpy()
                    )
                    x = leap.concat([x, padding_zeros], -2)
                    h = h + padding_size
                x = leap.reshape(x, [bs, h, w])
                weight = leap.reshape(weight, [1, 1, self.weight.shape[-1]])
        else:
            weight = leap.reshape(self.weight.data, [1, self.weight.shape[-1]])
            seq_len = x.type.shape[0]

        if seq_len % 32 == 0 or seq_len == 1:
            output = leap.rms_norm(x, [-1], eps, weight=weight)
        else:
            x_pow = leap.pow(x, 2)
            x_mean = leap.reduce_mean(x_pow, [-1])
            varience = leap.rsqrt(leap.add(x_mean, eps))
            x = leap.mul(x, varience)
            output = leap.mul(x, weight)

        if ndim == 4 and seq_len == 1:
            seq_len, h, w = output.type.shape
            if padding_size is not None:
                print("unpadding after rms_norm")
                h = h - padding_size
                output = leap.slice(
                    output,
                    [0, 0, 0],
                    [seq_len, h, w],
                    [1, 1, 1],
                )
            output = leap.reshape(output, [1, seq_len, h, w])
        return output
