import math
import torch
import torch.nn as nn
from torch.quantization import DeQuantStub
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    ConstFakeQuant,
    DynamicQuantLinear,
    DynamicQuantMatmul,
    FakeQuantMatmul,
)
from leap_llm.nn.utils import Module

try:
    from horizon_plugin_pytorch.quantization import QuantStub
except ImportError:
    QuantStub = None


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_leap(x):
    shape = x.type.shape
    if len(shape) == 3:
        n_local_head, seq_len, head_dim = shape
        x1 = leap.slice(
            x, [0, 0, 0], [n_local_head, seq_len, head_dim // 2], [1, 1, 1]
        )
        x2 = leap.slice(
            x, [0, 0, head_dim // 2], [n_local_head, seq_len, head_dim], [1, 1, 1]
        )
        x2 = leap.mul(-1, x2)
        return leap.concat([x2, x1], 2)

    bs, n_heads, seq_len, head_dim = shape
    x1 = leap.slice(
        x,
        [0, 0, 0, 0],
        [bs, n_heads, seq_len, head_dim // 2],
        [1, 1, 1, 1],
    )
    x2 = leap.slice(
        x,
        [0, 0, 0, head_dim // 2],
        [bs, n_heads, seq_len, head_dim],
        [1, 1, 1, 1],
    )
    x2 = leap.mul(-1, x2)
    return leap.concat([x2, x1], 3)


def apply_rope_torch_1d(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rope_leap_1d(query_states, key_states, cos, sin):
    q_embed = leap.mul(query_states, cos)
    q_embed = leap.add(q_embed, leap.mul(rotate_half_leap(query_states), sin))
    k_embed = leap.mul(key_states, cos)
    k_embed = leap.add(k_embed, leap.mul(rotate_half_leap(key_states), sin))
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LocateAnythingTextAttention(Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx, use_plugin):
        super().__init__()
        self.use_plugin = use_plugin
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.q_mul_value = 1.0 / math.sqrt(self.head_dim)
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads"
                f"(got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if not self.use_plugin:
            self.qk_matmul = FakeQuantMatmul(8, 8, None)
            self.wv_matmul = FakeQuantMatmul(8, 8, None)
            self.cache_k_fq = ConstFakeQuant(8)
            self.cache_v_fq = ConstFakeQuant(8)
            self.q_proj = DynamicQuantLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=True,
                w_bits=config.w_bits,
                has_scale=config.has_scale,
            )
            self.k_proj = DynamicQuantLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=True,
                w_bits=config.w_bits,
                has_scale=config.has_scale,
            )
            self.v_proj = DynamicQuantLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=True,
                w_bits=config.w_bits,
                has_scale=config.has_scale,
            )
            self.o_proj = DynamicQuantLinear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                w_bits=config.w_bits,
                has_scale=config.has_scale,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=True
            )
            self.k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim, self.hidden_size, bias=False
            )
            self.cache_k_fq = QuantStub()
            self.cache_v_fq = QuantStub()
            # self.quant_values = QuantStub()
            # self.quant_keys = QuantStub()
            self.dequant = DeQuantStub()

    def build(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ):
        bsz, q_len, _ = hidden_states.type.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(query_states, [bsz, q_len, -1, self.head_dim])
        query_states = leap.transpose(query_states, [0, 2, 1, 3])

        key_states = leap.reshape(key_states, [bsz, q_len, -1, self.head_dim])
        key_states = leap.transpose(key_states, [0, 2, 1, 3])

        value_states = leap.reshape(value_states, [bsz, q_len, -1, self.head_dim])
        value_states = leap.transpose(value_states, [0, 2, 1, 3])

        cos, sin = position_embeddings
        query_states, key_states = apply_rope_leap_1d(
            query_states, key_states, cos, sin
        )
        key_states = leap.cast_type(key_states, output_type=leap.float32)
        value_states = leap.cast_type(value_states, output_type=leap.float32)

        new_key = key_states
        new_value = value_states
        if cache_keys is not None and cache_values is not None:
            cache_keys = self.cache_k_fq(cache_keys)
            cache_values = self.cache_v_fq(cache_values)
            cur_len = key_states.type.shape[2]
            _, c_len, num_heads, embed_dim = cache_keys.type.shape
            cache_keys = leap.slice(
                cache_keys,
                [0, cur_len, 0, 0],
                [bsz, c_len, num_heads, embed_dim],
                [1, 1, 1, 1],
            )
            cache_keys = leap.transpose(cache_keys, [0, 2, 1, 3])
            key_states = leap.concat([cache_keys, key_states], 2)

            cache_values = leap.slice(
                cache_values,
                [0, cur_len, 0, 0],
                [bsz, c_len, num_heads, embed_dim],
                [1, 1, 1, 1],
            )
            cache_values = leap.transpose(cache_values, [0, 2, 1, 3])
            value_states = leap.concat([cache_values, value_states], 2)

        kv_len = key_states.type.shape[2]
        if q_len >= 1024:
            query_states = leap.reshape(
                query_states, [bsz, self.num_key_value_heads, -1, self.head_dim]
            )
            key_states = leap.reshape(
                key_states, [bsz, self.num_key_value_heads, 1, kv_len, self.head_dim]
            )
            key_states = leap.tile(key_states, [1, 1, self.num_key_value_groups, 1, 1])
            key_states = leap.reshape(
                key_states,
                [
                    bsz,
                    self.num_key_value_heads * self.num_key_value_groups,
                    kv_len,
                    self.head_dim,
                ],
            )
            value_states = leap.reshape(
                value_states, [bsz, self.num_key_value_heads, 1, kv_len, self.head_dim]
            )
            value_states = leap.tile(
                value_states, [1, 1, self.num_key_value_groups, 1, 1]
            )
            value_states = leap.reshape(
                value_states,
                [
                    bsz,
                    self.num_key_value_heads * self.num_key_value_groups,
                    kv_len,
                    self.head_dim,
                ],
            )
        else:
            query_states = leap.reshape(
                query_states, [bsz, self.num_key_value_heads, -1, self.head_dim]
            )

        key_states = leap.transpose(key_states, [0, 1, 3, 2])
        query_states = leap.cast_type(query_states, output_type=leap.float32)
        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = leap.cast_type(
            attn_weights, output_type=hidden_states.type.element_type
        )

        if q_len < 1024:
            attn_weights = leap.reshape(attn_weights, [bsz, self.num_heads, q_len, -1])
        attn_weights = leap.mul(attn_weights, self.q_mul_value)

        if attention_mask is not None:
            kv_len_full = attn_weights.type.shape[-1]
            attention_mask = leap.reshape(
                attention_mask, [bsz, 1, q_len, kv_len_full]
            )
            attn_weights = leap.add(attn_weights, attention_mask)
        attn_weights = leap.softmax(attn_weights, -1)
        attn_weights = leap.cast_type(attn_weights, output_type=leap.float32)
        if q_len < 1024:
            attn_weights = leap.reshape(
                attn_weights,
                [bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1],
            )
        attn_output = self.wv_matmul(attn_weights, value_states)
        attn_output = leap.cast_type(
            attn_output, output_type=hidden_states.type.element_type
        )
        attn_output = leap.reshape(attn_output, [bsz, -1, q_len, self.head_dim])
        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)

        bs, num_heads, seq_len, _ = new_key.type.shape
        new_key = leap.transpose(new_key, [0, 2, 1, 3])
        new_value = leap.transpose(new_value, [0, 2, 1, 3])

        new_key = self.cache_k_fq(new_key)
        new_value = self.cache_v_fq(new_value)
        return attn_output, new_key, new_value

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states = query_states.reshape(-1, q_len, self.head_dim)
        key_states = key_states.reshape(-1, q_len, self.head_dim)
        query_states, key_states = apply_rope_torch_1d(
            query_states, key_states, cos, sin
        )
        key_states = key_states.reshape(bsz, -1, q_len, self.head_dim)
        new_key = key_states
        new_value = value_states
        if cache_keys is not None and cache_values is not None:
            cache_keys = self.cache_k_fq(cache_keys)
            cache_values = self.cache_v_fq(cache_values)
            cur_len = key_states.shape[2]
            cache_keys = cache_keys[:, cur_len:].transpose(1, 2)
            key_states = torch.cat([cache_keys, key_states], dim=2)
            cache_values = cache_values[:, cur_len:].transpose(1, 2)

            value_states = torch.cat([cache_values, value_states], dim=2)

        query_states = query_states.reshape(
            bsz, self.num_key_value_heads, -1, self.head_dim
        )

        if self.use_plugin:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        else:
            attn_weights = self.qk_matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, -1)
        attn_weights = torch.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
        if attention_mask is not None:
            attn_weights = torch.add(attn_weights, attention_mask)
        attn_weights = torch.softmax(attn_weights, -1).to(query_states.dtype)

        attn_weights = attn_weights.reshape(
            bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1
        )
        if self.use_plugin:
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            attn_output = self.wv_matmul(attn_weights, value_states)

        attn_output = attn_output.view(bsz, -1, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        new_key = new_key.transpose(1, 2)
        new_value = new_value.transpose(1, 2)
        new_key = self.cache_k_fq(new_key)
        new_value = self.cache_v_fq(new_value)
        if self.use_plugin:
            new_key = self.dequant(new_key)
            new_value = self.dequant(new_value)
        return attn_output, new_key, new_value


class Qwen2_5_VLVisionAttention(Module):
    def __init__(self, dim: int, num_heads: int = 16, use_plugin: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_plugin = use_plugin
        if self.use_plugin:
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)
        else:
            self.qkv = DynamicQuantLinear(dim, dim * 3, bias=True, w_bits=8)
            self.proj = DynamicQuantLinear(dim, dim, w_bits=8)
            self.qk_matmuls = nn.ModuleList([DynamicQuantMatmul() for _ in range(16)])
            self.wv_matmuls = nn.ModuleList([DynamicQuantMatmul() for _ in range(16)])
        self.q_mul_value = 1.0 / math.sqrt(self.head_dim)

    def build(
        self,
        hidden_states,
        lengths,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ):
        seq_length = hidden_states.type.shape[1]
        qkv = self.qkv(hidden_states)
        qkv = leap.reshape(qkv, [seq_length, 3, self.num_heads, -1])
        qkv = leap.transpose(qkv, [1, 0, 2, 3])
        query_states = leap.select(qkv, 0, 0)
        key_states = leap.select(qkv, 0, 1)
        value_states = leap.select(qkv, 0, 2)
        query_states, key_states = apply_rope_leap_1d(
            query_states, key_states, rotary_pos_emb_cos, rotary_pos_emb_sin
        )
        query_states = leap.transpose(query_states, [1, 0, 2])
        key_states = leap.transpose(key_states, [1, 0, 2])
        value_states = leap.transpose(value_states, [1, 0, 2])

        query_states = leap.reshape(query_states, [1, self.num_heads, seq_length, -1])
        key_states = leap.reshape(key_states, [1, self.num_heads, seq_length, -1])
        value_states = leap.reshape(value_states, [1, self.num_heads, seq_length, -1])

        query_states_split = []
        key_states_split = []
        values_states_split = []

        lengths = lengths[1:] - lengths[:-1]
        num_splits = torch.unique_consecutive(lengths).tolist()

        diffs = torch.cat((torch.tensor([1]), torch.diff(lengths).abs()))
        group_ids = (diffs != 0).cumsum(dim=0) - 1
        result = torch.zeros_like(lengths)
        result.scatter_add_(0, group_ids, lengths)

        unique_groups = torch.unique_consecutive(group_ids)
        lengths = result[unique_groups]

        lengths = torch.cat((torch.tensor([0]), lengths))
        lengths = lengths.cumsum(dim=0)

        lengths = lengths.tolist()

        for i in range(len(lengths) - 1):
            query_states_split.append(
                leap.slice(
                    query_states,
                    [0, 0, lengths[i], 0],
                    [1, 16, lengths[i + 1], 80],
                    [1, 1, 1, 1],
                )
            )
            key_states_split.append(
                leap.slice(
                    key_states,
                    [0, 0, lengths[i], 0],
                    [1, 16, lengths[i + 1], 80],
                    [1, 1, 1, 1],
                )
            )
            values_states_split.append(
                leap.slice(
                    value_states,
                    [0, 0, lengths[i], 0],
                    [1, 16, lengths[i + 1], 80],
                    [1, 1, 1, 1],
                )
            )
        attn_outputs = []

        for q, k, v, num_split, qk_matmul, wv_matmul in zip(
            query_states_split,
            key_states_split,
            values_states_split,
            num_splits,
            self.qk_matmuls,
            self.wv_matmuls,
        ):
            if num_split != 1:
                bs, num_heads, num_tokens, num_embeds = q.type.shape
                q = leap.reshape(q, [bs * self.num_heads, -1, num_split, num_embeds])
                k = leap.reshape(k, [bs * self.num_heads, -1, num_split, num_embeds])
                v = leap.reshape(v, [bs * self.num_heads, -1, num_split, num_embeds])
            attn_weights = qk_matmul(q, k)
            attn_weights = leap.mul(attn_weights, self.q_mul_value)
            attn_weights = leap.softmax(attn_weights, -1)

            v = leap.transpose(v, [0, 1, 3, 2])
            attn_output_tmp = wv_matmul(attn_weights, v)
            attn_output_tmp = leap.reshape(
                attn_output_tmp, [bs, num_heads, -1, num_embeds]
            )
            attn_output_tmp = leap.transpose(attn_output_tmp, [0, 2, 1, 3])
            attn_outputs.append(attn_output_tmp)
        attn_output = leap.concat(attn_outputs, dim=1)
        attn_output = leap.reshape(attn_output, [1, seq_length, -1])
        attn_output = self.proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor = None,
        rotary_pos_emb_sin: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[1]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query_states, key_states = apply_rope_torch_1d(
            query_states, key_states, rotary_pos_emb_cos, rotary_pos_emb_sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = lengths[1:] - lengths[:-1]

        num_splits = torch.unique_consecutive(lengths)

        diffs = torch.cat((torch.tensor([1]), torch.diff(lengths).abs()))
        group_ids = (diffs != 0).cumsum(dim=0) - 1
        result = torch.zeros_like(lengths)
        result.scatter_add_(0, group_ids, lengths)

        unique_groups = torch.unique_consecutive(group_ids)
        lengths = result[unique_groups]

        splits = [
            torch.split(tensor, lengths.tolist(), dim=2)
            for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = []
        if self.use_plugin:
            for q, k, v, num_split in zip(*splits, num_splits):
                if num_split != 1:
                    bs, num_heads, _, num_embeds = q.shape
                    q = q.view(bs * self.num_heads, -1, num_split, num_embeds)
                    k = k.view(bs * self.num_heads, -1, num_split, num_embeds)
                    v = v.view(bs * self.num_heads, -1, num_split, num_embeds)
                k = k.transpose(2, 3)
                attn_weights = torch.matmul(q, k)
                attn_weights = torch.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
                attn_weights = torch.softmax(attn_weights, -1)

                attn_output_tmp = torch.matmul(attn_weights, v)
                attn_output_tmp = attn_output_tmp.view(bs, num_heads, -1, num_embeds)
                attn_output_tmp = attn_output_tmp.transpose(1, 2)
                attn_outputs.append(attn_output_tmp)
        else:
            for q, k, v, num_split, qk_matmul, wv_matmul in zip(
                *splits, num_splits, self.qk_matmuls, self.wv_matmuls
            ):
                if num_split != 1:
                    bs, num_heads, _, num_embeds = q.shape
                    q = q.view(bs * self.num_heads, -1, num_split, num_embeds)
                    k = k.view(bs * self.num_heads, -1, num_split, num_embeds)
                    v = v.view(bs * self.num_heads, -1, num_split, num_embeds)
                k = k.transpose(2, 3)
                attn_weights = qk_matmul(q, k)
                attn_weights = torch.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
                attn_weights = torch.softmax(attn_weights, -1)

                attn_output_tmp = wv_matmul(attn_weights, v)
                attn_output_tmp = attn_output_tmp.view(bs, num_heads, -1, num_embeds)
                attn_output_tmp = attn_output_tmp.transpose(1, 2)
                attn_outputs.append(attn_output_tmp)
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output
