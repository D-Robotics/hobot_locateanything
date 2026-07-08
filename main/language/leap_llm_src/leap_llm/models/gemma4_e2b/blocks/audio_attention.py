import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.blocks.linear import Gemma4ClippableLinear
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4AudioConfig
from leap_llm.nn.modules import (
    DynamicQuantLinear,
    DynamicQuantMatmul,
)
from leap_llm.nn.utils import Module


class Gemma4AudioAttention(Module):
    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)

        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.q_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
        self.post = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)

        self.relative_k_proj = DynamicQuantLinear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))  # NOTE: weight

        self.ac_matmul = DynamicQuantMatmul()
        self.bd_matmul = DynamicQuantMatmul()
        self.wv_matmul = DynamicQuantMatmul()

        # self.tanh = FakeQuantTanh()

        self.register_buffer("softcap", torch.tensor(self.attention_logits_soft_cap), persistent=False)

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Splits a `(batch_size, seq_len, num_heads, head_dim)` tensor
        into non-overlapping blocks of `chunk_size` along the sequence dim.
        """
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim).contiguous()

    def _convert_to_block_leap(self, hidden_states):
        """ "

        Args:
            hidden_states (qtype): (bs = 1, seq_len, #heads, head_dim)

        Returns:
            _type_: (#block, chunk_size, #heads, head_dim)
        """
        bs, seq_len, num_heads, head_dim = hidden_states.type.shape
        assert bs == 1, "batch size should only be 1"
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        pad_zeros = torch.zeros((bs, pad, num_heads, head_dim), dtype=torch.float16)
        hidden_states = leap.concat([hidden_states, pad_zeros], dim=1)
        hidden_states = leap.reshape(hidden_states, (num_blocks, self.chunk_size, num_heads, head_dim))
        return hidden_states

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts overlapping context windows of `context_size` for every block, strided by `chunk_size`."""
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        hidden_states = F.pad(
            hidden_states, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1)
        )
        hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
        hidden_states = torch.movedim(hidden_states, -1, 2)
        return hidden_states.contiguous()

    def _extract_block_context_leap(self, hidden_states):
        bs, seq_len, num_heads, head_dim = hidden_states.type.shape

        assert bs == 1, "batch size should only be 1"

        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        padded_len = pad_left + seq_len + pad_right

        pad_zero_left = torch.zeros((bs, pad_left, num_heads, head_dim), dtype=torch.float16)
        pad_zero_right = torch.zeros((bs, pad_right, num_heads, head_dim), dtype=torch.float16)

        # (bs, seq_len, H, D) → (bs, padded_len, H, D)
        hidden_states = leap.concat([pad_zero_left, hidden_states], dim=1)
        hidden_states = leap.concat([hidden_states, pad_zero_right], dim=1)
        hidden_states = leap.reshape(
            hidden_states,
            (padded_len, num_heads, head_dim),
        )

        # unfold along padded_len dim, with self.context_size window and self.chunk_size as stride
        # last blk is not considered as the valid seq_len is always accounted due to the padding
        num_blk = (padded_len - self.context_size) // self.chunk_size + 1

        result = []

        for idx in range(num_blk):
            result.append(
                leap.slice(
                    hidden_states,
                    [idx * self.chunk_size, 0, 0],
                    [idx * self.chunk_size + self.context_size, num_heads, head_dim],
                    [1, 1, 1],
                )
            )

        result = leap.stack(result, dim=0)

        return result

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Relative position shift for blocked attention. See appendix B of https://huggingface.co/papers/1901.02860."""
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def _rel_shift_leap(self, x):
        num_heads, num_chunk, chunk_size, position_len = x.type.shape
        pad_zero = torch.zeros(
            (num_heads, num_chunk, chunk_size, self.context_size + 1 - position_len),
            dtype=torch.float16,
        )
        x = leap.concat([x, pad_zero], dim=-1)
        x = leap.reshape(x, (num_heads, num_chunk, chunk_size * (self.context_size + 1)))
        x = leap.slice(
            x,
            [0, 0, 0],
            [num_heads, num_chunk, chunk_size * self.context_size],
            [1, 1, 1],
        )
        x = leap.reshape(x, (num_heads, num_chunk, chunk_size, self.context_size))
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ):
        """Chunked local multi-head self-attention with relative position bias.

        Computes ``softmax(QK^T + R + softcap) V`` with a sliding-window
        context and the Gemma-style chunked mask layout. Uses the
        Conformer-style log-scale Q/K scaling and ``softcap`` tanh clipping.

        Args:
            hidden_states (torch.Tensor): Layer-normed audio hidden
                states. Shape: ``(batch_size, seq_len, hidden_size)``,
                e.g. ``(1, 750, 1024)``.
            position_embeddings (torch.Tensor): Sinusoidal relative
                position embeddings. Shape:
                ``(1, 2 * context_size - 1, hidden_size)`` — split into
                ``sin``/``cos`` halves internally for ``relative_k_proj``.
            attention_mask (torch.BoolTensor | None): 5D blocked mask
                of valid positions. Shape:
                ``(batch_size, num_blocks, chunk_size, context_size)``.
                For ``seq_len = 750``, ``chunk_size = 12``:
                ``(1, 63, 12, 24)``. ``True`` marks allowed positions.

        Returns:
            torch.Tensor: Attention output projected by ``self.post``.
                Shape: ``(batch_size, seq_len, hidden_size)``, e.g.
                ``(1, 750, 1024)``.
        """
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale

        print("[before unfolding]")
        print(f"query_states shape: {query_states.shape}")
        print(f"key_states shape: {key_states.shape}")
        print(f"value_states shape: {value_states.shape}")

        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        print("[after unfolding]")
        print(f"query_states shape: {query_states.shape}")
        print(f"key_states shape: {key_states.shape}")
        print(f"value_states shape: {value_states.shape}")

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)

        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(), self.config.attention_invalid_logits_value
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))

        return attn_output

    def build(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
    ):
        bs, seq_len, _ = hidden_states.type.shape
        assert bs == 1, "only support batch_size = 1"
        hidden_shape = (bs, seq_len, self.num_heads, self.head_dim)

        # (bs, seq_len, #heads, head_dim)
        query_states = leap.reshape(self.q_proj(hidden_states), hidden_shape)
        key_states = leap.reshape(self.k_proj(hidden_states), hidden_shape)
        value_states = leap.reshape(self.v_proj(hidden_states), hidden_shape)

        q_scaling = self.q_scale * F.softplus(self.per_dim_scale)

        query_states = leap.mul(query_states, q_scaling)
        key_states = leap.mul(key_states, self.k_scale)

        print("[before unfolding]")
        print(f"query_states shape: {query_states.type.shape}")
        print(f"key_states shape: {key_states.type.shape}")
        print(f"value_states shape: {value_states.type.shape}")

        # (#blk, blk_size, #heads, head_dim)
        query_states = self._convert_to_block_leap(query_states)
        # (#blk, ctx_size, #head, head_dim)
        key_states = self._extract_block_context_leap(key_states)
        value_states = self._extract_block_context_leap(value_states)

        print("[after unfolding]")
        print(f"query_states shape: {query_states.type.shape}")
        print(f"key_states shape: {key_states.type.shape}")
        print(f"value_states shape: {value_states.type.shape}")

        num_blocks = query_states.type.shape[0]

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = leap.reshape(relative_key_states, (-1, self.num_heads, self.head_dim))

        # (#blk, blk_size, #heads, head_dim) -> (#heads, #blk, blk_size, head_dim)
        queris = leap.transpose(query_states, (2, 0, 1, 3))
        # (#blk, ctx_size, #heads, head_dim) -> (#heads, #blk, head_dim, ctx_size) for fq_matmaul
        # key_states = leap.transpose(key_states, (2, 0, 3, 1))
        # (#blk, ctx_size, #heads, head_dim) -> (#heads, #blk, ctx_size, head_dim) for dq_matmul
        key_states = leap.transpose(key_states, (2, 0, 1, 3))

        # (#heads, #blk, blk_size, ctx_size)
        matrix_ac = self.ac_matmul(queris, key_states)
        # (#heads, seq_len, head_dim)
        queries_flat = leap.reshape(queris, (self.num_heads, -1, self.head_dim))
        # (#heads, blk_size, head_dim)
        relative_key_states = leap.transpose(relative_key_states, (1, 0, 2))
        # (#heads, seq_len, blk_size)
        matrix_bd = self.bd_matmul(queries_flat, relative_key_states)
        # (#heads, #blk, blk_size, blk_size)
        matrix_bd = leap.reshape(matrix_bd, (self.num_heads, num_blocks, self.chunk_size, -1))
        # (#heads, #blk, blk_size, ctx_size)
        matrix_bd = self._rel_shift_leap(matrix_bd)

        attn_weights = leap.add(matrix_ac, matrix_bd)
        attn_weights = leap.div(attn_weights, self.softcap)
        attn_weights = leap.cast_type(attn_weights, output_type=leap.int16)
        attn_weights = leap.tanh(attn_weights)
        # attn_weights = leap.cast_type(attn_weights, output_type=leap.float16)
        attn_weights = leap.mul(attn_weights, self.softcap)
        attn_weights = leap.add(attn_weights, attention_mask)

        attn_weights = leap.softmax(attn_weights, -1)
        # (#blk, ctx_size, #heads, head_dim) -> (#heads, #blk, head_dim, ctx_size)
        value_states = leap.transpose(value_states, (2, 0, 3, 1))
        # (#heads, #blk, blk_size, head_dim)
        attn_output = self.wv_matmul(attn_weights, value_states)

        attn_output = leap.transpose(attn_output, (1, 2, 0, 3))
        attn_output = leap.reshape(attn_output, (bs, -1, self.num_heads * self.head_dim))
        attn_output = leap.slice(
            attn_output,
            [0, 0, 0],
            [bs, seq_len, self.num_heads * self.head_dim],
            [1, 1, 1],
        )

        attn_output = self.post(attn_output)

        return attn_output
