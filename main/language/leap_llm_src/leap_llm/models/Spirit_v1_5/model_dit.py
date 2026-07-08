import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from hbdk4.compiler import leap
from safetensors.torch import load_file as safe_load_file

from leap_llm.models.Spirit_v1_5.blocks.dit_attention import Attention
from leap_llm.models.Spirit_v1_5.blocks.dit_feedforward import FeedForward
from leap_llm.models.Spirit_v1_5.config.dit_config import DiTConfig
from leap_llm.models.Spirit_v1_5.model import PolicyFeature, SpiritVLAConfig
from leap_llm.nn.modules import DynamicQuantLinear, FakeQuantSwish, LayerNorm
from leap_llm.nn.utils import Model, Module, timeit

DIT_PREFIX = "dit."

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class TimestepEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: str | None = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = DynamicQuantLinear(in_channels, time_embed_dim, bias=sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = DynamicQuantLinear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = FakeQuantSwish()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = DynamicQuantLinear(time_embed_dim, time_embed_dim_out, bias=sample_proj_bias)


    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        return sample

    def build(self, sample, condition=None):
        if condition is not None:
            sample = leap.add(sample, self.cond_proj(condition))
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Timesteps(Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
        table_size: int = 10,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.table_size = table_size

        # Keep original Spirit timestep semantics: 1.0, 0.9, ..., 0.1
        table_timesteps = torch.linspace(1.0, 0.1, steps=table_size, dtype=torch.float32)
        table = get_timestep_embedding(
            table_timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        self.register_buffer("table_timesteps", table_timesteps, persistent=False)
        self.register_buffer("table", table, persistent=False)


    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.table[timesteps]
        return t_emb

    def build(self, timesteps):
        t_emb = leap.gather_nd(self.table, timesteps, 0)
        t_emb = leap.reshape(t_emb, [1, -1])
        t_emb = leap.cast_type(t_emb, output_type=leap.float16)
        return t_emb


class TimestepEncoder(Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb

    def build(self, timesteps):
        timesteps_proj = self.time_proj(timesteps)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb


class SinusoidalPositionalEmbedding(Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x

    def build(self, x):
        _, seq_length, _ = x.type.shape
        pe = leap.slice(self.pe, [0, 0, 0], [1, seq_length, 1536], [1, 1, 1])
        x = leap.add(x, pe)
        return x

class AdaLayerNorm(Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = FakeQuantSwish()
        self.linear = DynamicQuantLinear(embedding_dim, output_dim)
        # Spirit checkpoint does not contain `norm1.norm.{weight,bias}`.
        # Use non-affine LayerNorm here to keep state_dict keys aligned.
        self.norm = LayerNorm(output_dim // 2, eps=norm_eps, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x

    def build(self, x, temb=None):
        temb = self.linear(self.silu(temb))
        scale = leap.slice(temb, [0, 0], [1, 1536], [1, 1])
        shift = leap.slice(temb, [0, 1536], [1, 1536*2], [1, 1])
        scale = leap.reshape(scale, [1, 1, 1536])
        shift = leap.reshape(shift, [1, 1, 1536])
        x1 = self.norm(x)
        x2 = leap.mul(x1, leap.add(1, scale))
        x3 = leap.add(x2, shift)
        return x3

class BasicTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type
        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        self.norm3 = LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states

    def build(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, temb=None):
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = leap.add(hidden_states, attn_output)
        if len(hidden_states.type.shape) == 4:
            hidden_states = leap.reshape(hidden_states, [1, -1])
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = leap.add(hidden_states, ff_output)
        if len(hidden_states.type.shape) == 4:
            hidden_states = leap.reshape(hidden_states, [1, -1])
        return hidden_states

class BaseDiT(Model):
    def __init__(
        self, config: DiTConfig
    ):
        super().__init__()
        self.config = config
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, compute_dtype=self.config.compute_dtype)
        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and self.config.interleave_self_attention
            curr_cross_attention_dim = self.config.cross_attention_dim if not use_self_attn else None
            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=self.config.norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=self.config.positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)
        self.norm_out = LayerNorm(self.inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = DynamicQuantLinear(self.inner_dim, 2 * self.inner_dim)
        self.silu = FakeQuantSwish()


        self.state_proj = DynamicQuantLinear(self.config.max_state_dim, config.dit_hidden_size)
        self.action_in_proj = DynamicQuantLinear(self.config.max_action_dim, config.dit_hidden_size)
        self.action_out_proj = DynamicQuantLinear(config.dit_hidden_size, self.config.max_action_dim)

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.int32,
        vlm_last_embed: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):

        embs = []

        #外面做这步
        #state[:, :, [2, 9]] = 0
        state_emb = self.state_proj(state)
        embs.append(state_emb)
        action_emb = self.action_in_proj(x_t)
        embs.append(action_emb)
        hidden_states = torch.cat(embs, dim=1)


        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        vlm_last_embed = vlm_last_embed.contiguous()
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=encoder_attention_mask,
                    encoder_hidden_states=vlm_last_embed,
                    temb=temb,
                )
        conditioning = temb
        shift, scale = self.proj_out_1(self.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        suffix_out = hidden_states[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        x_t = x_t - 0.1 * v_t
        return x_t

    def build(self, state, x_t, timestep, vlm_last_embed, encoder_attention_mask):

        embs = []
        state_emb = self.state_proj(state)
        embs.append(state_emb)
        action_emb = self.action_in_proj(x_t)
        embs.append(action_emb)
        hidden_states = leap.concat(embs, 1)

        temb = self.timestep_encoder(timestep)
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=encoder_attention_mask,
                    encoder_hidden_states=vlm_last_embed,
                    temb=temb,
                )
        conditioning = temb
        conditioning = self.proj_out_1(self.silu(conditioning))
        shift = leap.slice(conditioning, [0, 0], [1, 1536], [1, 1])
        scale = leap.slice(conditioning, [0, 1536], [1, 1536*2], [1, 1])
        shift = leap.reshape(shift, [1, 1, 1536])
        scale = leap.reshape(scale, [1, 1, 1536])
        x1 = self.norm_out(hidden_states)
        x2 = leap.mul(x1, leap.add(1, scale))
        x3 = leap.add(x2, shift)

        suffix_out = leap.slice(x3, [0, 1, 0], [1, self.config.n_action_steps + 1, 1536], [1, 1, 1])
        v_t = self.action_out_proj(suffix_out)
        v_t = leap.mul(v_t, -0.1)
        x_t = leap.add(x_t, v_t)
        return x_t



class SpiritDitModel:
    @staticmethod
    @timeit
    def build(config_path: str, model_dir: str) -> "SpiritDitModel":
        def ensure_exists(path: Path, description: str) -> None:
            if not path.exists():
                raise FileNotFoundError(f"{description} not found: {path}")


        def filter_dataclass_fields(cfg_cls: type, data: dict[str, Any]) -> dict[str, Any]:
            valid_keys = {field.name for field in fields(cfg_cls)}
            return {key: value for key, value in data.items() if key in valid_keys}


        def to_policy_features(feature_map: dict[str, Any] | None) -> dict[str, PolicyFeature]:
            if not isinstance(feature_map, dict):
                return {}

            converted: dict[str, PolicyFeature] = {}
            for key, value in feature_map.items():
                if isinstance(value, PolicyFeature):
                    converted[key] = value
                elif isinstance(value, dict):
                    converted[key] = PolicyFeature(type=value["type"], shape=tuple(value["shape"]))
                else:
                    raise TypeError(f"Unsupported feature type for key `{key}`: {type(value)}")
            return converted
        def load_spirit_config(config_path: Path) -> SpiritVLAConfig:
            ensure_exists(config_path, "Spirit config")

            with config_path.open() as file:
                raw_config = json.load(file)

            filtered_config = filter_dataclass_fields(SpiritVLAConfig, raw_config)
            filtered_config["input_features"] = to_policy_features(filtered_config.get("input_features"))
            filtered_config["output_features"] = to_policy_features(filtered_config.get("output_features"))
            return SpiritVLAConfig(**filtered_config)


        def build_dit_model(config: SpiritVLAConfig) -> BaseDiT:
            dit_config = DiTConfig()
            dit_config.num_attention_heads = config.dit_num_heads
            dit_config.attention_head_dim = config.dit_hidden_size // config.dit_num_heads
            dit_config.num_layers = config.dit_num_layers
            dit_config.interleave_self_attention = config.dit_interleave_self_attention
            dit_config.cross_attention_dim = config.dit_cross_attention_dim
            dit_config.max_state_dim = config.max_state_dim
            dit_config.max_action_dim = config.max_action_dim
            dit_config.dit_hidden_size = config.dit_hidden_size
            dit_config.n_action_steps = config.n_action_steps

            return BaseDiT(dit_config)


        def extract_dit_state_dict(weight_path, prefix):
            ensure_exists(weight_path, "Spirit checkpoint")

            full_state_dict = safe_load_file(str(weight_path))
            dit_state_dict = {
                key[len(prefix) :]: value for key, value in full_state_dict.items() if key.startswith(prefix)
            }
            if not dit_state_dict:
                raise KeyError(f"No DiT weights with prefix `{prefix}` found in {weight_path}")

            # state_proj / action_in_proj / action_out_proj 在原始 SpiritVLAPolicy 顶层，
            # 并不在 `dit.` 前缀下面；这里单独合并进 dit_state_dict。
            extra_modules = ("state_proj", "action_in_proj", "action_out_proj")
            for module_name in extra_modules:
                for suffix in (".weight", ".bias"):
                    key = module_name + suffix
                    if key in full_state_dict:
                        dit_state_dict[key] = full_state_dict[key]
            return dit_state_dict


        def load_dit_weights(model, state_dict):
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    f"DiT state_dict load mismatch. Missing: {missing_keys}, Unexpected: {unexpected_keys}"
                )
        spirit_config = load_spirit_config(config_path)
        dit_model = build_dit_model(spirit_config)
        dit_state_dict = extract_dit_state_dict(model_dir, DIT_PREFIX)
        load_dit_weights(dit_model, dit_state_dict)
        return SpiritDitModel(dit_model, spirit_config)

    def __init__(self, model: BaseDiT, model_args: SpiritVLAConfig):
        self.model = model
        self.model_args = model_args


    def get_leap_input_types(
        self, action_dim, action_horizon, tokens_num
    ) -> list[leap.TensorType]:
        q_len = 1 + action_horizon
        input_types = [
            leap.TensorType([1, 1, action_dim], leap.float16),
            leap.TensorType([1, action_horizon, action_dim], leap.float16),
            leap.TensorType([1], leap.int32),
            leap.TensorType([1, tokens_num, 2560], leap.float16),
            leap.TensorType([1, 1, q_len, tokens_num], leap.float16),
        ]
        return input_types

    def compile(
        self,
        output_model_path: str,
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        inputs = self.get_leap_input_types(32, 60, 320)
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "spirit_dit", bc_path)
        # 编译 HBO 模型并链接成最终模型
        hbos = []
        bc_path = str(Path(output_model_path).with_suffix(".convert.bc"))
        mlir_module = self.model.convert_mlir(
            bc_module,
            save_path=bc_path,
            march=kwargs["march"],
            dynamic_quant=True,
        )

        kwargs["core_num"] = 4
        kwargs["max_l2m_size"] = 25165824
        print(f"kwargs : {kwargs}")
        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        hbo_model = self.model.compile_hbo(
            mlir_module,
            hbo_path,
            **kwargs,
        )
        hbos.append(hbo_model)

        hbm_path = str(Path(output_model_path).with_suffix(".hbm"))
        return self.model.link_models(hbos, hbm_path)