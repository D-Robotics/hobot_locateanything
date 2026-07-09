import torch
from hbdk4.compiler import leap

from leap_llm.models.pi0.blocks.configuration_gemma import GemmaConfig
from leap_llm.nn.modules import DynamicQuantLinear, DynamicQuantMatmul
from leap_llm.nn.utils import Module


class RotaryPosEmb(Module):
    def __init__(self):
        super().__init__()

    def rotate_half(self, x):
        # [n_local_head, seqlen, head_dim]
        n_local_head, seq_len, head_dim = x.type.shape
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        x1 = leap.slice(x, [0, 0, 0], [n_local_head, seq_len, head_dim // 2], [1, 1, 1])
        x2 = leap.slice(
            x, [0, 0, head_dim // 2], [n_local_head, seq_len, head_dim], [1, 1, 1]
        )
        x2 = leap.mul(-1, x2)
        rotate_x = leap.concat([x2, x1], 2)
        return rotate_x

    def apply_rotary_pos_emb(self, query_states, key_states, cos, sin):
        """
        # query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        # key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        """
        q_embed = leap.mul(query_states, cos)
        q_embed = leap.add(q_embed, leap.mul(self.rotate_half(query_states), sin))
        k_embed = leap.mul(key_states, cos)
        k_embed = leap.add(k_embed, leap.mul(self.rotate_half(key_states), sin))
        return q_embed, k_embed

    def rotate_half_torch(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]  # noqa: E203
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_torch(self, query_states, key_states, cos, sin):
        """
        # query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        # key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        """
        q_embed = torch.mul(query_states, cos)
        q_embed = torch.add(
            q_embed, torch.mul(self.rotate_half_torch(query_states), sin)
        )
        k_embed = torch.mul(key_states, cos)
        k_embed = torch.add(k_embed, torch.mul(self.rotate_half_torch(key_states), sin))
        return q_embed, k_embed

    def build(self, query_states, key_states, cos, sin):
        return self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    def forward(self, query_states, key_states, cos, sin):
        return self.apply_rotary_pos_emb_torch(query_states, key_states, cos, sin)


class GemmaAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.q_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = DynamicQuantLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.qk = DynamicQuantMatmul()
        self.sv = DynamicQuantMatmul()
        self.apply_rotary_pos_emb = RotaryPosEmb()

    def build(self, hidden_states, cos, sin, attention_mask):
        seqlen = hidden_states.type.shape[1]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(
            query_states, [seqlen, self.num_attention_heads, self.head_dim]
        )
        query_states = leap.transpose(query_states, [1, 0, 2])
        key_states = leap.reshape(
            key_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )

        key_states = leap.transpose(key_states, [1, 0, 2])
        value_states = leap.reshape(
            value_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )
        value_states = leap.transpose(value_states, [1, 0, 2])

        # xk, xv
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        new_k = key_states
        new_v = value_states

        H, W, C = query_states.type.shape

        query_states = leap.reshape(
            query_states,
            [
                self.num_key_value_heads,
                self.num_key_value_groups * W,
                self.head_dim,
            ],
        )

        attn_weights = self.qk(query_states, key_states)
        attn_weights = leap.reshape(attn_weights, [H, seqlen, seqlen])
        attn_weights = leap.mul(attn_weights, self.scaling)

        if attention_mask is not None:
            if len(attention_mask.type.shape) == len(attn_weights.type.shape) - 1:
                print("attn_weights:", attn_weights.type)
                attention_mask = leap.reshape(attention_mask, [1, seqlen, seqlen])
                print("attention_mask:", attention_mask.type)
            attn_weights = leap.add(attn_weights, attention_mask)

        attn_weights = leap.softmax(attn_weights, -1)
        attn_weights = leap.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, seqlen],
        )
        value_states = leap.transpose(value_states, [0, 2, 1])
        attn_output = self.sv(attn_weights, value_states)
        attn_output = leap.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output = leap.transpose(attn_output, [1, 0, 2])
        attn_output = leap.reshape(attn_output, [seqlen, self.hidden_size])
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, new_k, new_v

    def forward(self, hidden_states, cos, sin, attention_mask):
        assert hidden_states.ndim == 3
        batch_size, seqlen, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            [seqlen, self.num_attention_heads, self.head_dim]
        )
        query_states = query_states.transpose(1, 0)

        key_states = key_states.reshape(
            [seqlen, self.num_key_value_heads, self.head_dim]
        )

        key_states = key_states.transpose(1, 0)

        value_states = value_states.reshape(
            [seqlen, self.num_key_value_heads, self.head_dim]
        )
        value_states = value_states.transpose(1, 0)

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        new_k = key_states
        new_v = value_states
        key_states_t = key_states.transpose(2, 1)
        H, W, C = query_states.shape
        query_states = query_states.reshape(
            [
                self.num_key_value_heads,
                self.num_key_value_groups * W,
                self.head_dim,
            ]
        )
        attn_weights = self.qk(query_states, key_states_t)
        attn_weights = attn_weights.reshape([H, seqlen, seqlen])
        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.softmax(attn_weights, -1)

        attn_weights = torch.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, seqlen],
        )

        attn_output = self.sv(attn_weights, value_states)

        attn_output = torch.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output = torch.transpose(attn_output, 1, 0)
        attn_output = torch.reshape(attn_output, [batch_size, seqlen, self.hidden_size])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, new_k, new_v
