import warnings
from typing import Optional

import torch
from hbdk4.compiler import leap

from leap_llm.nn.utils import Module


class Qwen3VLTextRotaryEmbedding(Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        self.rope_type = getattr(config, "rope_type", "default")
        if self.rope_type != "default":
            warnings.warn(
                f"rope_type is not set as default, instead as {self.rope_type}",
                "results could be different when not default rope is applied",
            )
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(
            config, device
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.mrope_section = getattr(config, "mrope_section", [24, 20, 20])
        self.h_indices = torch.tensor(
            list(range(1, self.mrope_section[1] * 3, 3)), device=device
        )
        self.w_indices = torch.tensor(
            list(range(2, self.mrope_section[2] * 3, 3)), device=device
        )
        self.freq_dim: int = getattr(config, "head_dim", None) // 2
        self.h_mask = torch.zeros(
            self.freq_dim, dtype=torch.bool, device=device
        ).contiguous()
        self.w_mask = torch.zeros(
            self.freq_dim, dtype=torch.bool, device=device
        ).contiguous()
        self.h_mask[self.h_indices] = True
        self.w_mask[self.w_indices] = True
        # self.max_pos_len: int = config.max_position_embeddings
        self.max_pos_len: int = 4096
        self.ctx_position_ids = torch.arange(
            0, self.max_pos_len, 1, dtype=torch.int64, device=device
        )
        self.ctx_freqs = (
            inv_freq[:, None].float() @ self.ctx_position_ids[None, :].float()
        )
        self.ctx_freqs = self.ctx_freqs.transpose(0, 1)
        cache_cos = self.ctx_freqs.cos().contiguous()
        cache_sin = self.ctx_freqs.sin().contiguous()  # (ctx_len, freq_dim)
        # print(f"cache_cos/sin.shape = {cache_cos.shape}")
        self.register_buffer("cache_cos", cache_cos, persistent=False)
        self.register_buffer("cache_sin", cache_sin, persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[any] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        base = getattr(config, "rope_theta", 5_000_000)
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    # @dynamic_rope_update  # power user: used with advanced RoPE types
    # (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float()
        inv_freq_expanded = inv_freq_expanded.expand(3, position_ids.shape[1], -1, 1)
        # shape (3, bs, 1, positions)
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = inv_freq_expanded.float() @ position_ids_expanded.float()
            freqs = freqs.transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def build(self, position_ids):
        """calculate mrope according to position_ids

        Args:
            position_ids (int32): [3, bs, seq_len]
        """
        mm_types, bs, seq_len = position_ids.type.shape
        if seq_len == 1:
            # decode stage, 1-D text PE now
            pos_ids_t = leap.slice(position_ids, [0, 0, 0], [1, bs, seq_len], [1, 1, 1])
            pos_ids_t = leap.reshape(pos_ids_t, [bs, seq_len, 1])
            sin_t = leap.gather_nd(self.cache_sin.numpy(), pos_ids_t, 0)
            cos_t = leap.gather_nd(self.cache_cos.numpy(), pos_ids_t, 0)
            cos = leap.concat((cos_t, cos_t), dim=-1)
            sin = leap.concat((sin_t, sin_t), dim=-1)
            return cos, sin

        # (1, bs, seq_len)
        pos_ids_t = leap.slice(position_ids, [0, 0, 0], [1, bs, seq_len], [1, 1, 1])
        # (bs, seq_len, 1)
        pos_ids_t = leap.reshape(pos_ids_t, [bs, seq_len, 1])
        pos_ids_h = leap.slice(position_ids, [1, 0, 0], [2, bs, seq_len], [1, 1, 1])
        pos_ids_h = leap.reshape(pos_ids_h, [bs, seq_len, 1])
        pos_ids_w = leap.slice(position_ids, [2, 0, 0], [3, bs, seq_len], [1, 1, 1])
        pos_ids_w = leap.reshape(pos_ids_w, [bs, seq_len, 1])

        # print(f"pos_ids_t.shape = {pos_ids_t.type.shape}")
        # (bs, seq_len, freq_dim)
        sin_t = leap.gather_nd(self.cache_sin.numpy(), pos_ids_t, 0)
        sin_h = leap.gather_nd(self.cache_sin.numpy(), pos_ids_h, 0)
        sin_w = leap.gather_nd(self.cache_sin.numpy(), pos_ids_w, 0)
        cos_t = leap.gather_nd(self.cache_cos.numpy(), pos_ids_t, 0)
        cos_h = leap.gather_nd(self.cache_cos.numpy(), pos_ids_h, 0)
        cos_w = leap.gather_nd(self.cache_cos.numpy(), pos_ids_w, 0)
        # print(f"sin_t.shape = {sin_t.type.shape}")

        # interleave apply here, to THWTHW...THW
        cos_t = leap.where(self.h_mask.numpy(), cos_h, cos_t)
        cos_t = leap.where(self.w_mask.numpy(), cos_w, cos_t)
        sin_t = leap.where(self.h_mask.numpy(), sin_h, sin_t)
        sin_t = leap.where(self.w_mask.numpy(), sin_w, sin_t)

        cos = leap.concat((cos_t, cos_t), dim=-1)
        sin = leap.concat((sin_t, sin_t), dim=-1)
        # print(f"sin.shape = {sin.type.shape}")
        return cos, sin
