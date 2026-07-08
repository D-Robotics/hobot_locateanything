import json
import math
import os
import shutil
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from hbdk4.compiler import leap, save
from torch import nn
from torch.nn import functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

from leap_llm.nn.modules import (
    AvgPool1d,
    ConstFakeQuant,
    Conv1d,
    Embedding,
    FakeQuantEmbedding,
    FakeQuantGELU,
    FakeQuantLinear,
    FakeQuantMul,
    LayerNormSplit,
    Qwen2_5_VisionPatchEmbed,
    Qwen2RMSNorm,
)
from leap_llm.nn.utils import Model, timeit

from .blocks.transformer_block import (
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniDecoderLayer,
    Qwen2_5OmniPatchMerger,
    Qwen2_5OmniVisionBlock,
)


@dataclass
class Qwen2_5OmniVisionEncoderConfig:
    depth: int = 32
    hidden_size: int = 2048
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    out_hidden_size: int = 2048
    fullatt_block_indexes = [7, 15, 23, 31]


@dataclass
class Qwen2_5OmniAudioEncoderConfig:
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    n_window: int = 100
    output_dim: int = 2048


@dataclass
class Qwen2_5OmniTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    head_dim: int = int(hidden_size // num_attention_heads)
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    use_sliding_window: bool = False
    sliding_window: int = 32768
    max_window_layers: int = 28
    pad_token_id: int = 151643
    cache_len: int = 1024
    input_embedding: bool = True

    prefill_seq_len: int = 256
    decode_seq_len: int = 1
    cache_len: int = 2048


@dataclass
class Qwen2_5OmniThinkerConfig:
    audio_token_index: int = 151646
    image_token_index: int = 151655
    video_token_index: int = 151656
    vision_start_token_id: int = 151652
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    user_token_id: int = 872

    vision_config: Qwen2_5OmniVisionEncoderConfig = None
    audio_config: Qwen2_5OmniAudioEncoderConfig = None
    text_config: Qwen2_5OmniTextConfig = None


@dataclass
class Qwen2_5OmniTalkerConfig:
    pass


@dataclass
class Qwen2_5OmniToken2WavConfig:
    pass


@dataclass
class Qwen2_5OmniConfig:
    thinker_config: Qwen2_5OmniThinkerConfig = None
    talker_config: Qwen2_5OmniTalkerConfig = None
    token2wav_config: Qwen2_5OmniToken2WavConfig = None

    max_batch_size: int = 1


@dataclass
class CompileArgs:
    name: str = "qwen2_5_omni"  # function name for compile


@dataclass
class QuantizeArgs:
    pass


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisionRotary(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.theta = theta
        self.rotary_dim = dim

    # support [h_pos, w_pos]
    def forward(self, position_ids):
        # [2, patch_len, 1]
        position_ids = position_ids.float().unsqueeze(-1)
        idx_theta = position_ids * self.theta
        # [patch_len, rotary_dim]
        idx_theta = idx_theta.permute(1, 0, 2).reshape(-1, self.rotary_dim)
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2)
        ).float()
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.positional_embedding = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniRotaryEmbeddingV2(nn.Module):
    def __init__(self, config, cache_len=None, device=None):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.rotary_dim = config.head_dim
        self.theta = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
            self.mrope_section = config.rope_scaling["mrope_section"]
            self.theta_sections = self.theta.unsqueeze(0).split(
                self.mrope_section, dim=-1
            )

        else:
            self.rope_type = "default"

        self.max_positions = cache_len

        try:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        except KeyError:
            rope_init_fn = Qwen2_5OmniRotaryEmbedding.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(self.max_positions, device=inv_freq.device).float()
        freqs = torch.einsum("p,f->pf", positions, inv_freq.float())
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_cache = emb.cos() * self.attention_scaling
        sin_cache = emb.sin() * self.attention_scaling

        self.register_buffer(
            "cos_cache", cos_cache.unsqueeze(0).repeat(3, 1, 1), persistent=False
        )
        self.register_buffer(
            "sin_cache", sin_cache.unsqueeze(0).repeat(3, 1, 1), persistent=False
        )

    @torch.no_grad()
    def forward(self, position_ids):
        cos_cache = self.cos_cache.unsqueeze(1)  # was (3, max_positions, emb_dim)
        sin_cache = self.sin_cache.unsqueeze(1)

        ids_exp = position_ids.unsqueeze(-1).expand(-1, -1, -1, cos_cache.size(-1))
        cos = cos_cache.gather(dim=2, index=ids_exp)
        sin = sin_cache.gather(dim=2, index=ids_exp)

        return cos, sin


class Qwen2_5OmniRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5OmniThinkerConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        try:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        except KeyError:
            self.rope_init_fn = self.compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        """Default RoPE initialization for transformers 5.x compatibility."""
        if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
            base = config.rope_parameters.get("rope_theta", 10000.0)
        elif hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            base = config.rope_scaling.get("rope_theta", 10000.0)
        elif hasattr(config, "rope_theta"):
            base = config.rope_theta
        else:
            base = 10000.0

        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / head_dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)

            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5OmniThinkerTextModel(Model):
    def __init__(self, config: Qwen2_5OmniTextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = int(self.hidden_size // self.num_attention_heads)
        self.embed_tokens = FakeQuantEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                Qwen2_5OmniDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cache_len = config.cache_len
        self.mrope_section = self.config.rope_scaling["mrope_section"]

        self.rotary_emb = Qwen2_5OmniRotaryEmbeddingV2(
            config=config, cache_len=self.cache_len
        )
        self.lm_head = FakeQuantLinear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.cos = self.rotary_emb.cos_cache.unsqueeze(1)
        self.sin = self.rotary_emb.sin_cache.unsqueeze(1)

        self.inputs_embeds_fq = ConstFakeQuant(16)
        self.mask_fq = ConstFakeQuant(16)
        self.cos_fq = ConstFakeQuant(16)
        self.sin_fq = ConstFakeQuant(16)

    def build(self, inputs, attention_mask, cos, sin, *caches):
        if caches is not None:
            caches_k = caches[: len(caches) // 2]
            caches_v = caches[len(caches) // 2 :]
        else:
            caches_k = [None] * self.num_hidden_layers
            caches_v = [None] * self.num_hidden_layers

        # prefill: inputs_embeds not None
        # decode: input_ids not None
        if self.config.input_embedding:
            inputs_embeds = inputs
        else:
            input_ids = inputs
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.inputs_embeds_fq(inputs_embeds)
        attention_mask = self.mask_fq(attention_mask)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # position_ids = leap.reshape(position_ids, position_ids.type.shape + [1])
        # cos = leap.gather_nd(self.cos, position_ids, 2)
        # sin = leap.gather_nd(self.sin, position_ids, 2)

        cos = self.cos_fq(cos)
        sin = self.sin_fq(sin)

        position_embeddings = (cos, sin)

        new_keys = []
        new_values = []

        for decoder_layer, cache_k, cache_v in zip(self.layers, caches_k, caches_v):
            if cache_k is not None:
                cache_k = leap.transpose(cache_k, [0, 2, 1, 3])
                cache_v = leap.transpose(cache_v, [0, 2, 1, 3])

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_k=cache_k,
                cache_v=cache_v,
            )

            hidden_states = layer_outputs[0]
            key_states = layer_outputs[1]
            value_states = layer_outputs[2]

            key_states = leap.transpose(key_states, [0, 2, 1, 3])
            value_states = leap.transpose(value_states, [0, 2, 1, 3])

            new_keys.append(key_states)
            new_values.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, *new_keys, *new_values

    def get_rotary_pos_emb(self, position_ids, unsqueeze_dim=1):
        position_ids = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.cos.size(-1))

        cos = self.cos.to(device=position_ids.device).float()
        sin = self.sin.to(device=position_ids.device).float()

        cos = cos.gather(dim=2, index=position_ids)
        sin = sin.gather(dim=2, index=position_ids)

        mrope_section = self.mrope_section * 2
        cos = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
        ).unsqueeze(unsqueeze_dim)
        sin = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
        ).unsqueeze(unsqueeze_dim)
        # rotary_pos_emb = torch.cat([cos, sin], dim=0)
        # return rotary_pos_emb
        return (cos, sin)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        caches: List[torch.Tensor] = None,
    ):
        if caches is not None:
            caches_k = caches[: len(caches) // 2]
            caches_v = caches[len(caches) // 2 :]
        else:
            caches_k = [None] * self.num_hidden_layers
            caches_v = [None] * self.num_hidden_layers

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.inputs_embeds_fq(inputs_embeds)
        attention_mask = self.mask_fq(attention_mask)

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers

        if position_ids is not None:
            rotary_pos_emb = self.get_rotary_pos_emb(position_ids)
        else:
            cos, sin = rotary_pos_emb

            cos = self.cos_fq(cos)
            sin = self.sin_fq(sin)
            rotary_pos_emb = (cos, sin)

        new_keys = []
        new_values = []

        for decoder_layer, cache_k, cache_v in zip(self.layers, caches_k, caches_v):
            if cache_k is not None:
                cache_k = cache_k.transpose(2, 1)
                cache_v = cache_v.transpose(2, 1)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=rotary_pos_emb,
                cache_k=cache_k,
                cache_v=cache_v,
            )

            hidden_states = layer_outputs[0]
            key_states = layer_outputs[1]
            value_states = layer_outputs[2]

            key_states = key_states.transpose(2, 1)
            value_states = value_states.transpose(2, 1)

            new_keys.append(key_states)
            new_values.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, *new_keys, *new_values

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def save_cos_sin(self, output_dir):
        cos_path = os.path.join(output_dir, "cos.bin")
        sin_path = os.path.join(output_dir, "sin.bin")
        cos = self.cos.detach().cpu().numpy()
        cos.tofile(cos_path)
        sin = self.sin.detach().cpu().numpy()
        sin.tofile(sin_path)

    def get_leap_input_types(self, seq_len) -> List[leap.TensorType]:
        batch_size = 1
        input_types = []

        if self.config.input_embedding:
            inputs_embeds = leap.TensorType(
                [1, seq_len, self.hidden_size], leap.float32
            )
            inputs = inputs_embeds
        else:
            input_ids = leap.TensorType([1, seq_len], leap.int32)
            inputs = input_ids

        input_types.append(inputs)
        attention_mask = leap.TensorType(
            [batch_size, 1, seq_len, self.cache_len], leap.float32
        )
        rotary_pos_emb = leap.TensorType(
            [batch_size, 1, seq_len, self.head_dim], leap.float32
        )

        input_types.append(attention_mask)
        input_types.append(rotary_pos_emb)
        input_types.append(rotary_pos_emb)

        for _ in range(self.config.num_hidden_layers * 2):
            input_types.append(
                leap.TensorType(
                    [
                        batch_size,
                        self.config.cache_len,
                        self.config.num_key_value_heads,
                        int(self.config.hidden_size // self.config.num_attention_heads),
                    ],
                    leap.float32,
                )
            )

        return input_types

    def compile(
        self,
        stage: str,
        output_model_path: str,
        enable_vpu=True,
        **kwargs,
    ):
        assert self.is_compiled, "Model must be compiled before compiling."

        model_list = []
        stages = []

        if stage in {"text_model", "all"}:
            stages.append("text_model_prefill")
            stages.append("text_model_decode")

        for stage_name in stages:
            seq_len = (
                self.config.prefill_seq_len
                if stage_name == "text_model_prefill"
                else self.config.decode_seq_len
            )
            high_precision_qpp = True

            inputs = self.get_leap_input_types(seq_len)
            bc_path = str(Path(output_model_path).with_suffix(f".{stage_name}.bc"))
            bc_module = self.export_module(
                inputs, stage_name, bc_path, high_precision_qpp=high_precision_qpp
            )
            model_list.append(bc_module)

        hbos = []

        for bc_module in model_list:
            func_name = bc_module.functions[0].name
            convert_bc_path = str(
                Path(output_model_path).with_suffix(f".{func_name}_convert.bc")
            )
            mlir_module = self.convert_mlir(
                bc_module, convert_bc_path, enable_vpu=enable_vpu, march=kwargs["march"]
            )

            func = mlir_module.functions[0]
            func.remove_io_op(["Dequantize", "Quantize"])
            convert_removed_bc_path = str(
                Path(output_model_path).with_suffix(f".{func_name}_convert_removed.bc")
            )
            save(mlir_module, convert_removed_bc_path)
            print(f"kwargs : {kwargs}")
            hbo_path = str(Path(output_model_path).with_suffix(f".{func_name}.hbo"))
            hbo_model = self.compile_hbo(mlir_module, save_path=hbo_path, **kwargs)
            hbos.append(hbo_model)

        return self.link_models(hbos, output_model_path)


class Qwen2_5OmniVisionEncoder(Model):
    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig):
        super().__init__()
        self.mask_min_value = -512

        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.merge_size = config.spatial_merge_size
        self.temporal_patch_size = config.temporal_patch_size
        self.image_grid_thw = []

        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.resized_height = 448
        self.resized_width = 448
        # NOTE: 448x448
        # grid_thw = [1, 32, 32]
        grid_thw_h = self.resized_height // self.patch_size
        grid_thw_w = self.resized_width // self.patch_size
        grid_thw = [1, grid_thw_h, grid_thw_w]

        self.grid_thw = torch.tensor([grid_thw])
        self.seq_len = grid_thw[1] * grid_thw[2]

        window_index, cu_window_seqlens = self.get_window_index(self.grid_thw)
        self.window_index = window_index
        self.reverse_indices = torch.argsort(self.window_index)

        # fullatt_attention_mask = self.vision_attention_mask(
        #     self.grid_thw, min_value=self.mask_min_value
        # )
        # normal_attention_mask = self.vision_attention_mask(
        #     self.grid_thw, cu_window_seqlens, min_value=self.mask_min_value
        # )

        # self.attention_mask = torch.stack(
        #     [fullatt_attention_mask, normal_attention_mask], dim=0
        # )

        rotary_pos_emb_cos_sin = self.get_rotary_pos_emb_cos_sin(seq_len=self.seq_len)

        self.rotary_pos_emb_cos = rotary_pos_emb_cos_sin[0]
        self.rotary_pos_emb_sin = rotary_pos_emb_cos_sin[1]

        self.blocks = nn.ModuleList(
            [Qwen2_5OmniVisionBlock(config) for _ in range(config.depth)]
        )

        self.merger = Qwen2_5OmniPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

        self.hidden_states_in_fq = ConstFakeQuant(8)
        self.hidden_states_out_fq = ConstFakeQuant(16)
        self.cos_fq = ConstFakeQuant(16)
        self.sin_fq = ConstFakeQuant(16)

    def rot_pos_emb(self, grid_thw=None):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def vision_position_ids(self, grid_thw):
        pos_ids = []
        for _t, h, w in grid_thw:
            llm_h, llm_w = h // self.merge_size, w // self.merge_size
            # compute pos_ids
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(llm_h, self.merge_size, llm_w, self.merge_size)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(llm_h, self.merge_size, llm_w, self.merge_size)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids]))
            # pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        position_ids = torch.cat(pos_ids, dim=0)
        return position_ids

    def vision_attention_mask(self, grid_thw, cu_window_seqlens=None, min_value=-512):
        seq_len = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        if cu_window_seqlens is None:
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        else:
            cu_seqlens = cu_window_seqlens
        attention_mask = torch.full([1, seq_len, seq_len], min_value)

        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0
        attention_mask = attention_mask.to(dtype=torch.float32)
        return attention_mask

    def vision_reshape(self, images):
        images = [images] * self.temporal_patch_size
        patches = torch.concat(images, axis=0)
        _, channel, height, width = patches.shape
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])
        self.image_grid_thw.append([grid_t, grid_h, grid_w])
        return flatten_patches, grid_thw

    def get_rotary_pos_emb_cos_sin(self, seq_len=1024):
        grid_thw = self.grid_thw
        position_ids = self.vision_position_ids(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        pos_ids = position_ids.transpose(1, 0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        cos = cos.unsqueeze(1).repeat(1, 1, 2).float()
        sin = sin.unsqueeze(1).repeat(1, 1, 2).float()
        return cos, sin

    def build(self, hidden_states):
        hidden_states = self.hidden_states_in_fq(hidden_states)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.hidden_states_out_fq(hidden_states)

        seq_len, _ = hidden_states.type.shape

        rotary_pos_emb_cos = self.rotary_pos_emb_cos
        rotary_pos_emb_sin = self.rotary_pos_emb_sin
        rotary_pos_emb_cos = self.cos_fq(rotary_pos_emb_cos)
        rotary_pos_emb_sin = self.sin_fq(rotary_pos_emb_sin)
        window_index = self.window_index
        # attention_mask = self.attention_mask

        hidden_states = leap.reshape(
            hidden_states,
            [seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1],
        )

        window_index_len = window_index.shape[0]
        window_index = leap.reshape(window_index, [window_index_len, 1, 1])
        hidden_states = leap.gather_nd(hidden_states, window_index, 0)
        hidden_states = leap.reshape(hidden_states, [seq_len, -1])

        for layer_num, blk in enumerate(self.blocks):
            window_attention = (
                layer_num not in self.fullatt_block_indexes
            )

            hidden_states = blk(
                hidden_states,
                window_attention=window_attention,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        hidden_states = self.merger(hidden_states)
        self.reverse_indices = leap.reshape(self.reverse_indices, [window_index_len, 1])
        hidden_states = leap.gather_nd(hidden_states, self.reverse_indices, 0)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.hidden_states_in_fq(hidden_states)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.hidden_states_out_fq(hidden_states)
        seq_len, _ = hidden_states.size()

        rotary_pos_emb_cos = self.rotary_pos_emb_cos.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        rotary_pos_emb_sin = self.rotary_pos_emb_sin.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        rotary_pos_emb_cos = self.cos_fq(rotary_pos_emb_cos)
        rotary_pos_emb_sin = self.sin_fq(rotary_pos_emb_sin)

        window_index = self.window_index

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        for layer_num, blk in enumerate(self.blocks):
            window_attention = (
                layer_num not in self.fullatt_block_indexes
            )
            hidden_states = blk(
                hidden_states,
                window_attention=window_attention,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[self.reverse_indices, :]
        return hidden_states

    def get_leap_input_types(self) -> List[leap.TensorType]:
        seq_len = self.seq_len
        dtype = leap.float32
        audio_input_types = [
            leap.TensorType(
                [
                    seq_len,
                    self.config.temporal_patch_size
                    * self.config.patch_size
                    * self.config.patch_size
                    * self.config.in_channels,
                ],
                dtype,
            ),
        ]
        return audio_input_types

    def compile(
        self,
        stage: str,
        output_model_path: str,
        enable_vpu=True,
        **kwargs,
    ):
        inputs = self.get_leap_input_types()
        bc_path = str(Path(output_model_path).with_suffix(f".{stage}.bc"))
        bc_module = self.export_module(inputs, stage, bc_path, high_precision_qpp=True)
        func_name = bc_module.functions[0].name
        convert_bc_path = str(
            Path(output_model_path).with_suffix(f".{func_name}_convert.bc")
        )
        mlir_module = self.convert_mlir(
            bc_module, convert_bc_path, enable_vpu=enable_vpu, march=kwargs["march"]
        )
        print(f"kwargs : {kwargs}")
        hbo_path = str(Path(output_model_path).with_suffix(f".{func_name}.hbo"))
        hbo_model = self.compile_hbo(mlir_module, save_path=hbo_path, **kwargs)
        return self.link_models([hbo_model], output_model_path)


class Qwen2_5OmniAudioEncoder(Model):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.embed_dim = embed_dim
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.gelu1 = FakeQuantGELU(quantized=True)
        self.gelu2 = FakeQuantGELU(quantized=True)

        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )

        self.audio_bos_eos_token = Embedding(2, config.output_dim)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = LayerNormSplit(config.d_model)
        self.avg_pooler = AvgPool1d(2, stride=2)
        self.proj = FakeQuantLinear(config.d_model, config.output_dim)
        self.mul = FakeQuantMul(quantized=True)

        self.padded_feature_fq = ConstFakeQuant(16)
        self.padded_mask_fq = ConstFakeQuant(16)
        self.attention_mask_fq = ConstFakeQuant(16)

    def build(
        self,
        padded_feature,
        padded_mask,
        attention_mask,
    ):
        padded_mask = leap.cast_type(padded_mask, output_type=leap.float32)
        padded_mask = self.padded_mask_fq(padded_mask)
        padded_feature = self.padded_feature_fq(padded_feature)
        attention_mask = self.attention_mask_fq(attention_mask)

        padded_feature = leap.transpose(padded_feature, [0, 2, 1])
        padded_mask = leap.transpose(padded_mask, [0, 2, 1])

        padded_embed = self.gelu1(self.conv1(padded_feature))
        padded_embed = self.mul(padded_embed, padded_mask)
        padded_embed = self.gelu2(self.conv2(padded_embed))

        _, seqlen, _ = padded_embed.type.shape
        slice_positional_embedding = leap.slice(
            self.positional_embedding.positional_embedding,
            [0, 0],
            [seqlen, self.embed_dim],
            [1, 1],
        )

        padded_embed = leap.add(padded_embed, slice_positional_embedding)
        padded_embed = leap.reshape(padded_embed, [-1, self.embed_dim])
        hidden_states = padded_embed

        for _idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs

        each_audio_states = self.avg_pooler(hidden_states)
        each_audio_states = self.ln_post(each_audio_states)
        token_audio = self.proj(each_audio_states)
        return token_audio

    def forward(
        self,
        padded_feature,
        padded_mask,
        attention_mask,
    ):
        padded_feature = self.padded_feature_fq(padded_feature)
        padded_mask = self.padded_mask_fq(padded_mask)
        attention_mask = self.attention_mask_fq(attention_mask)

        padded_embed = self.gelu1(self.conv1(padded_feature))
        padded_embed = self.mul(padded_embed, padded_mask)
        padded_embed = self.gelu2(self.conv2(padded_embed)).transpose(1, 2)

        slice_positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
            .to(padded_embed.device)
        )

        padded_embed = padded_embed + slice_positional_embedding
        hidden_states = padded_embed[0]

        for _idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        each_audio_states = self.avg_pooler(hidden_states.transpose(0, 1)).transpose_(
            0, 1
        )
        each_audio_states = self.ln_post(each_audio_states)
        token_audio = self.proj(each_audio_states)
        return token_audio

    def get_leap_input_types(self) -> List[leap.TensorType]:
        batch_size = 1
        dtype = leap.float32
        audio_input_types = [
            leap.TensorType(
                [batch_size, self.config.num_mel_bins, self.config.n_window * 2], dtype
            ),
            leap.TensorType([batch_size, 1, self.config.n_window * 2], leap.int32),
            leap.TensorType(
                [batch_size, self.config.n_window, self.config.n_window], dtype
            ),
        ]
        return audio_input_types

    def compile(
        self,
        stage: str,
        output_model_path: str,
        enable_vpu=True,
        **kwargs,
    ):
        inputs = self.get_leap_input_types()
        bc_path = str(Path(output_model_path).with_suffix(f".{stage}.bc"))
        bc_module = self.export_module(inputs, stage, bc_path, high_precision_qpp=True)
        func_name = bc_module.functions[0].name
        convert_bc_path = str(
            Path(output_model_path).with_suffix(f".{func_name}_convert.bc")
        )
        mlir_module = self.convert_mlir(
            bc_module, convert_bc_path, enable_vpu=enable_vpu, march=kwargs["march"]
        )
        print(f"kwargs : {kwargs}")
        hbo_path = str(Path(output_model_path).with_suffix(f".{func_name}.hbo"))
        hbo_model = self.compile_hbo(mlir_module, save_path=hbo_path, **kwargs)
        return self.link_models([hbo_model], output_model_path)

    def padded_and_mask_function(
        self, tensor_list, tensor_len, padding_value=0, padding_side="right"
    ):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=torch.float32,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )


class Qwen2_5OmniThinkerForConditionalGeneration(Model):
    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__()
        self.config = config
        self.audio_tower = Qwen2_5OmniAudioEncoder(config.audio_config)
        self.visual = Qwen2_5OmniVisionEncoder(config.vision_config)
        self.vocab_size = config.text_config.vocab_size
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.model = Qwen2_5OmniThinkerTextModel(config.text_config)


class Qwen2_5OmniModel(Model):
    def __init__(
        self,
        config: Qwen2_5OmniConfig,
    ):
        super().__init__()
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)


def dataclass_from_dict(cls, dikt):
    """
    Recursively instantiate `cls` (a @dataclass) from the dict `dikt`.
    """
    if not is_dataclass(cls):
        # not a dataclass: just return the raw value
        return dikt

    init_kwargs = {}
    for f in fields(cls):
        raw_value = dikt.get(f.name, {})
        if is_dataclass(f.type) and isinstance(raw_value, dict):
            init_kwargs[f.name] = dataclass_from_dict(f.type, raw_value)
        else:
            init_kwargs[f.name] = raw_value if raw_value != {} else f.default

    return cls(**init_kwargs)


class Qwen2_5Omni:
    @staticmethod
    @timeit
    def build(model_dir: str, chunk_size=256, cache_len=1024) -> "Qwen2_5Omni":
        assert os.path.isdir(
            model_dir
        ), f"Checkpoint directory '{model_dir}' does not exist."

        checkpoints = sorted(Path(model_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {model_dir}"
        ckpt_path = checkpoints[0]  # No parallel
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        config_path = os.path.join(model_dir, "config.json")
        assert os.path.exists(config_path), f"config.json not found in {model_dir}"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        model_args: Qwen2_5OmniConfig = dataclass_from_dict(Qwen2_5OmniConfig, config)
        model_args.thinker_config.text_config.prefill_seq_len = chunk_size
        model_args.thinker_config.text_config.cache_len = cache_len
        model = Qwen2_5OmniModel(model_args)

        mapping = {
            "thinker.visual.merger.mlp.0.weight": "thinker.visual.merger.mlp.proj0.weight",  # noqa: E501
            "thinker.visual.merger.mlp.0.bias": "thinker.visual.merger.mlp.proj0.bias",
            "thinker.visual.merger.mlp.2.weight": "thinker.visual.merger.mlp.proj1.weight",  # noqa: E501
            "thinker.visual.merger.mlp.2.bias": "thinker.visual.merger.mlp.proj1.bias",
            "thinker.lm_head.weight": "thinker.model.lm_head.weight",
        }

        new_state_dict = {}
        for key, value in checkpoint.items():
            if key in mapping:
                key = mapping[key]
            new_key = key
            new_state_dict[new_key] = value

        # model.load_state_dict(new_state_dict)
        model.load_state_dict(new_state_dict, False)

        return Qwen2_5Omni(model, model_args)

    def __init__(self, model: Qwen2_5OmniModel, model_args: Qwen2_5OmniConfig):
        self.model = model
        self.model_args = model_args

    def get_input_embeddings(self):
        return self.model.thinker.model.get_input_embeddings()

    def save_input_embeddings(self, output_dir):
        embed_tokens = self.get_input_embeddings()
        embed_tokens_path = os.path.join(output_dir, "embed_tokens.bin")
        if not os.path.exists(embed_tokens_path):
            weights = embed_tokens.weight
            weights = weights.detach().cpu().numpy()
            weights.tofile(embed_tokens_path)

    def get_audio_tower(self):
        return self.model.thinker.audio_tower

    def get_visual(self):
        return self.model.thinker.visual

    def get_text_model(self):
        return self.model.thinker.model


def save_model_checkpoint(model_dir, dir_path):
    ckpt_dir = os.path.join(dir_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model_checkpoint.pth")

    if not os.path.exists(ckpt_path):
        device = "cpu"
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float32
        ).to(device)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Save checkpoint path: {ckpt_path}")

    config_json_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_json_path):
        shutil.copyfile(os.path.join(model_dir, "config.json"), config_json_path)

    return ckpt_dir
