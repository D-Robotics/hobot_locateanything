import torch
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.nn.modules.linear import DynamicQuantLinear
from leap_llm.nn.modules.matmul import DynamicQuantMatmul
from leap_llm.nn.utils import Module


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def rotate_half_leap(x):
    bs, dim1, dim2, head_dim = x.type.shape
    x1 = leap.slice(x, [0, 0, 0, 0], [bs, dim1, dim2, head_dim // 2], [1, 1, 1, 1])
    x2 = leap.slice(
        x, [0, 0, 0, head_dim // 2], [bs, dim1, dim2, head_dim], [1, 1, 1, 1]
    )
    x2 = leap.mul(-1, x2)
    rotate_x = leap.concat([x2, x1], -1)
    return rotate_x


def apply_rotary_pos_emb_vision_leap(query_states, key_states, cos, sin):
    """
    states: (bs, seqlen, #head, head_dim)
    pe: (1, seqlen, 1, head_dim)
    """
    q_embed = leap.mul(query_states, cos)
    q_embed = leap.add(q_embed, leap.mul(rotate_half_leap(query_states), sin))
    k_embed = leap.mul(key_states, cos)
    k_embed = leap.add(k_embed, leap.mul(rotate_half_leap(key_states), sin))
    return q_embed, k_embed


class Qwen3VLVisionAttention(Module):
    """
    (attn): Qwen3VLVisionAttention(
    (qkv): Linear(in_features=1024, out_features=3072, bias=True)
    (proj): Linear(in_features=1024, out_features=1024, bias=True)
    )
    """

    def __init__(self, config, use_plugin: bool = False):
        super().__init__()
        self.use_plugin = use_plugin
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = DynamicQuantLinear(self.dim, self.dim * 3)
        self.proj = DynamicQuantLinear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.qk = DynamicQuantMatmul()
        self.sv = DynamicQuantMatmul()

    def build(self, hidden_states, position_embeddings):
        """
        hidden_states: (bs, seqlen, hsize)
        pe: ()
        """
        bs, seq_len, hsz = hidden_states.type.shape
        assert hsz == self.num_heads * self.head_dim, "hidden size mismatch"
        qkv_states = self.qkv(hidden_states)
        qkv_states = leap.reshape(qkv_states, [seq_len, 3, self.num_heads, -1])
        qkv_states = leap.transpose(qkv_states, [1, 0, 2, 3])
        q_states = leap.slice(
            qkv_states,
            [0, 0, 0, 0],
            [1, seq_len, self.num_heads, self.head_dim],
            [1, 1, 1, 1],
        )
        k_states = leap.slice(
            qkv_states,
            [1, 0, 0, 0],
            [2, seq_len, self.num_heads, self.head_dim],
            [1, 1, 1, 1],
        )
        v_states = leap.slice(
            qkv_states,
            [2, 0, 0, 0],
            [3, seq_len, self.num_heads, self.head_dim],
            [1, 1, 1, 1],
        )

        cos, sin = position_embeddings

        # print(f"q_states.shape = {q_states.type.shape}") # (1, 784, 16, 64)
        # print(f"cos.shape = {cos.shape}") # (784, 1, 64)

        # (1, seq_len, #head, head_dim)
        q_states, k_states = apply_rotary_pos_emb_vision_leap(
            q_states, k_states, cos, sin
        )
        q_states = leap.transpose(q_states, [0, 2, 1, 3])
        k_states = leap.transpose(k_states, [0, 2, 1, 3])
        attn_wt = leap.mul(self.qk(q_states, k_states), self.scaling)
        attn_wt = leap.softmax(attn_wt, -1)
        v_states = leap.transpose(v_states, [0, 2, 3, 1])
        attn_output = self.sv(attn_wt, v_states)
        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [bs, seq_len, -1])
        attn_output = self.proj(attn_output)
        return attn_output

    def forward(self, hidden_states, position_embeddings):
        """
        hidden_states: (seq_len, hidden_size)
        """
        seq_length = hidden_states.shape[0]
        # (seq_len, num_head, head_dim)
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)  # (1, seq_len, num_head, head_dim)
        cos, sin = position_embeddings  # (1, seqlen, 1, head_dim)
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        # (1, num_head, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # (1, num_head, seq_len, seq_len)
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        # (1, num_head, seq_len, head_dim)
        # ->(1, seq_len, num_head, head_dim)
        # ->(1, seq_len, hidden_size)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output
