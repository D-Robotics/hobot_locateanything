import json
import os
import shutil
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import List

import torch
from hbdk4.compiler import leap
from torch import nn
from torch.nn import functional as F
from torch.quantization import DeQuantStub
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as hf_Qwen2_5_VLForConditionalGeneration,
)

from leap_llm.nn.modules import (
    DynamicQuantLinear,
    Embedding,
    Qwen2_5_VisionPatchEmbed,
    RMSNorm,
)
from leap_llm.nn.utils import Model, load_safetensors_state_dict, timeit

from .blocks.transformer_block import (
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLVisionBlock,
)

try:
    from horizon_plugin_pytorch.quantization import QuantStub
except ImportError:
    QuantStub = None


def dataclass_from_dict(cls, dikt):
    """
    Recursively instantiate `cls` (a @dataclass) from the dict `dikt`.
    """
    if not is_dataclass(cls):
        return dikt

    init_kwargs = {}
    for f in fields(cls):
        raw_value = dikt.get(f.name, {})
        if is_dataclass(f.type) and isinstance(raw_value, dict):
            init_kwargs[f.name] = dataclass_from_dict(f.type, raw_value)
        else:
            init_kwargs[f.name] = raw_value if raw_value != {} else f.default
    return cls(**init_kwargs)


@dataclass
class Qwen2_5_VLVisionConfig:
    depth: int = 32
    hidden_size: int = 3584
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    out_hidden_size: int = 3584
    fullatt_block_indexes = [7, 15, 23, 31]
    tokens_per_second = 4
    image_width = 952
    image_height = 420
    mask_min_value = -32768
    w_bits: int = 8
    has_scale: bool = False


@dataclass
class Qwen2_5_VLTextConfig:
    vocab_size: int = 152064
    hidden_size: int = 8192
    intermediate_size: int = 29568
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    attention_dropout = 0.0
    rope_scaling = {
        "type": "default",
        "mrope_section": [16, 24, 24],
        "rope_type": "default",
    }
    max_prefill_text_token = 514
    max_new_tokens = 512
    max_lm_tokens = 4096

    prefill_seq_len: int = 256
    decode_seq_len: int = 1
    cache_len: int = 4096
    batch_size: int = 1
    w_bits: int = 8
    has_scale: bool = False


@dataclass
class Qwen2_5_VLConfig:
    vision_config: Qwen2_5_VLVisionConfig = None
    text_config: Qwen2_5_VLTextConfig = None
    image_token_id = 151655
    vision_start_token_id = 151652
    eos_token_id = 151645


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


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # transformers version compatibility:
        # - 4.x: ROPE_INIT_FUNCTIONS contains 'default', can be used directly
        # - 5.x: ROPE_INIT_FUNCTIONS removed 'default', need compute_default_rope_parameters
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
            .expand(1, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = (
            position_ids[:, :, None, :].float().to(device=inv_freq_expanded.device)
        )
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            cos = cos.squeeze()
            sin = sin.squeeze()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5_VLVisionModel(Model):
    def __init__(
        self,
        config: Qwen2_5_VLVisionConfig,
        use_plugin: False,
    ):
        super().__init__()
        self.config = config
        self.use_plugin = use_plugin
        self.mask_min_value = config.mask_min_value
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.merge_size = config.spatial_merge_size
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            use_conv2d=True,
            quant_output=False,
            use_plugin=use_plugin,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        grid_thw = [
            1,
            config.image_height // config.patch_size,
            config.image_width // config.patch_size,
        ]
        self.grid_thw = torch.tensor([grid_thw])
        self.seq_len = grid_thw[1] * grid_thw[2]

        rotary_pos_emb_cos_sin = self.get_rotary_pos_emb_cos_sin(seq_len=self.seq_len)
        self.rotary_pos_emb_cos = rotary_pos_emb_cos_sin[0]
        self.rotary_pos_emb_sin = rotary_pos_emb_cos_sin[1]

        window_index, cu_window_seqlens = self.get_window_index(self.grid_thw)
        self.window_index = window_index

        fullatt_lengths = self.vision_mask_lengths(
            self.grid_thw, min_value=self.mask_min_value
        )
        normal_lengths = self.vision_mask_lengths(
            self.grid_thw, cu_window_seqlens, min_value=self.mask_min_value
        )
        self.lengths = [fullatt_lengths, normal_lengths]
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VLVisionBlock(config, use_plugin=self.use_plugin)
                for _ in range(config.depth)
            ]
        )

        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            use_plugin=self.use_plugin,
        )
        if self.use_plugin:
            self.quant_hiddenstates = QuantStub()
            self.quant_cos = QuantStub()
            self.quant_sin = QuantStub()
        self.dequant = DeQuantStub()

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
        for t, h, w in grid_thw:
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
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        position_ids = torch.cat(pos_ids, dim=0)
        return position_ids

    def vision_mask_lengths(self, grid_thw, cu_window_seqlens=None, min_value=-512):
        seq_len = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        if cu_window_seqlens is None:
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        else:
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens,
                device=seq_len.device,
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_seqlens = torch.unique_consecutive(cu_window_seqlens)
        return cu_seqlens

    def get_rotary_pos_emb_cos_sin(self, seq_len):
        grid_thw = self.grid_thw
        position_ids = self.vision_position_ids(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        pos_ids = position_ids
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
        batch_size, seq_len, _ = hidden_states.type.shape
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb_cos = self.rotary_pos_emb_cos.to(torch.float16)
        rotary_pos_emb_sin = self.rotary_pos_emb_sin.to(torch.float16)
        window_index = self.window_index
        lengths = self.lengths
        fullattn_lengths = lengths[0]
        normal_lengths = lengths[1]

        hidden_states = leap.reshape(
            hidden_states,
            [seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1],
        )
        window_index_len = window_index.shape[0]
        window_index = leap.reshape(window_index, [window_index_len, 1, 1])
        hidden_states = leap.gather_nd(hidden_states, window_index, 0)
        hidden_states = leap.reshape(hidden_states, [batch_size, seq_len, -1])
        for layer_num, blk in enumerate(self.blocks):
            lengths_now = fullattn_lengths if layer_num in self.fullatt_block_indexes else normal_lengths

            hidden_states = blk(
                hidden_states,
                lengths=lengths_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )
        hidden_states = leap.clip(hidden_states, -65504.0, 65504.0)
        hidden_states = self.merger(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bs, seq_len, _ = hidden_states.size()
        if self.use_plugin:
            hidden_states = self.quant_hiddenstates(hidden_states)
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb_cos = self.rotary_pos_emb_cos.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        rotary_pos_emb_sin = self.rotary_pos_emb_sin.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        if self.use_plugin:
            rotary_pos_emb_cos = self.quant_cos(rotary_pos_emb_cos)
            rotary_pos_emb_sin = self.quant_sin(rotary_pos_emb_sin)

        window_index = self.window_index

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(1, seq_len, -1)

        lengths = self.lengths
        fullattn_lengths = lengths[0]
        normal_lengths = lengths[1]
        for layer_num, blk in enumerate(self.blocks):
            lengths_now = fullattn_lengths if layer_num in self.fullatt_block_indexes else normal_lengths

            hidden_states = blk(
                hidden_states,
                lengths=lengths_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )
        hidden_states = self.merger(hidden_states)
        hidden_states = self.dequant(hidden_states)
        return hidden_states

    def get_leap_input_types(self) -> List[leap.TensorType]:
        seq_len = self.seq_len
        dtype = leap.float16
        vision_input_types = [
            leap.TensorType(
                [
                    1,
                    seq_len,
                    self.config.patch_size
                    * self.config.patch_size
                    * self.config.in_channels,
                ],
                dtype,
            ),
        ]
        return vision_input_types


class Qwen2_5_VLTextModel(Model):
    def __init__(self, config: Qwen2_5_VLTextConfig, use_plugin: False):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.use_plugin = use_plugin
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_plugin=self.use_plugin,
        )
        self.config = config
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config)
        if hasattr(config, "layer_types") is False:
            config.layer_types = [
                "full_attention" for i in range(config.num_hidden_layers)
            ]
        self.layers = nn.ModuleList(
            [
                Qwen2_5_VLDecoderLayer(config, layer_idx, self.use_plugin)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        if self.use_plugin:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = DynamicQuantLinear(
                config.hidden_size, config.vocab_size, bias=False
            )
        pseudo_input_ids = (
            torch.arange(self.config.max_lm_tokens).view(1, 1, -1).float()
        )
        cache_cos, cache_sin = self.rotary_emb(pseudo_input_ids, pseudo_input_ids)
        mrope_section = self.config.rope_scaling["mrope_section"]
        self.mrope_section = mrope_section * 2
        if self.use_plugin:
            self.quant_input_embeds = QuantStub()
            self.quant_cos = QuantStub()
            self.quant_sin = QuantStub()
            self.quant_attention_mask = QuantStub()
        self.dequant = DeQuantStub()
        self.register_buffer("cache_cos", cache_cos, persistent=True)
        self.register_buffer("cache_sin", cache_sin, persistent=True)

    def get_rotary_emb(self):
        return self.rotary_emb

    def get_input_embeddings(self):
        return self.embed_tokens

    def build(self, inputs_embeds, position_ids, attention_mask, *caches):
        hidden_states = inputs_embeds
        new_keys = []
        new_values = []

        bs, dim, num_tokens = position_ids.type.shape
        if dim > 1:
            split_cache_cos = self.cache_cos.split(self.mrope_section, dim=-1)
            split_cache_sin = self.cache_sin.split(self.mrope_section, dim=-1)

            split_position_ids = []

            for i in range(3):
                if bs > 1:
                    slice_position_ids = leap.slice(
                        position_ids,
                        [0, i, 0],
                        [bs, i + 1, num_tokens],
                        [1, 1, 1],
                    )
                    slice_position_ids = leap.reshape(
                        slice_position_ids, (bs, num_tokens, 1)
                    )
                else:
                    slice_position_ids = leap.slice(
                        position_ids, [0, i, 0], [1, i + 1, num_tokens], [1, 1, 1]
                    )
                    slice_position_ids = leap.reshape(slice_position_ids, (num_tokens, 1))
                split_position_ids.append(slice_position_ids)
            used_cos = []
            used_sin = []

            for (i, cos), sin in zip(enumerate(split_cache_cos), split_cache_sin):
                cur_position_ids = split_position_ids[i % 3]
                cos = cos.contiguous()
                cur_cos = leap.gather_nd(cos, cur_position_ids, 0)
                cur_cos = leap.reshape(cur_cos, (bs, 1, num_tokens, -1))
                used_cos.append(cur_cos)
                sin = sin.contiguous()
                cur_sin = leap.gather_nd(sin, cur_position_ids, 0)
                cur_sin = leap.reshape(cur_sin, (bs, 1, num_tokens, -1))
                used_sin.append(cur_sin)

            cos = leap.concat(used_cos, dim=-1)

            sin = leap.concat(used_sin, dim=-1)
        else:
            if bs > 1:
                position_ids = leap.reshape(position_ids, (bs, num_tokens, 1))
                cos = leap.gather_nd(self.cache_cos, position_ids, 0)
                cos = leap.reshape(cos, (bs, 1, num_tokens, -1))
                sin = leap.gather_nd(self.cache_sin, position_ids, 0)
                sin = leap.reshape(sin, (bs, 1, num_tokens, -1))
            else:
                position_ids = leap.reshape(position_ids, (bs, -1))
                position_ids = leap.transpose(position_ids, (1, 0))
                cos = leap.gather_nd(self.cache_cos, position_ids, 0)
                cos = leap.transpose(cos, (1, 0))
                cos = leap.reshape(cos, (bs, 1, num_tokens, -1))
                sin = leap.gather_nd(self.cache_sin, position_ids, 0)
                sin = leap.transpose(sin, (1, 0))
                sin = leap.reshape(sin, (bs, 1, num_tokens, -1))

        position_embeddings = (cos, sin)

        cache_keys = caches[: len(caches) // 2]
        cache_values = caches[len(caches) // 2 :]
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_keys=cache_keys[idx] if len(cache_keys) else None,
                cache_values=cache_values[idx] if len(cache_values) else None,
            )
            new_keys.append(new_key)
            new_values.append(new_value)
        _, seq_len, hidden_size = hidden_states.type.shape
        hidden_states = self.norm(hidden_states)

        token_logits = self.lm_head(hidden_states)

        return token_logits, *new_keys, *new_values

    def forward(
        self,
        inputs_embeds,
        position_ids,
        attention_mask,
        caches=None,
    ):
        if caches is None:
            caches = []
        new_keys = []
        new_values = []
        dim = position_ids.shape[1]
        if dim > 1:
            split_cache_cos = self.cache_cos.split(self.mrope_section, dim=-1)
            split_cache_sin = self.cache_sin.split(self.mrope_section, dim=-1)

            split_position_ids = position_ids.split([1, 1, 1], dim=1)

            used_cos = []
            used_sin = []
            for (i, cos), sin in zip(enumerate(split_cache_cos), split_cache_sin):
                cur_position_ids = split_position_ids[i % 3]
                cos = cos.contiguous()
                used_cos.append(cos[cur_position_ids])
                sin = sin.contiguous()
                used_sin.append(sin[cur_position_ids])

            cos = torch.cat(used_cos, dim=-1)

            sin = torch.cat(used_sin, dim=-1)
        else:
            cos = self.cache_cos[position_ids]
            sin = self.cache_sin[position_ids]

        if self.use_plugin:
            cos = self.quant_cos(cos)
            sin = self.quant_sin(sin)
            inputs_embeds = self.quant_input_embeds(inputs_embeds)
            attention_mask = self.quant_attention_mask(attention_mask)
        hidden_states = inputs_embeds
        position_embeddings = (cos, sin)
        cache_keys = caches[: len(caches) // 2]
        cache_values = caches[len(caches) // 2 :]
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_keys=cache_keys[idx] if len(cache_keys) else None,
                cache_values=cache_values[idx] if len(cache_values) else None,
            )
            new_keys.append(new_key)
            new_values.append(new_value)

        _, seq_len, hidden_size = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        token_logits = self.lm_head(hidden_states)
        token_logits = self.dequant(token_logits)
        return token_logits, new_keys, new_values

    def get_leap_input_types_text_model(
        self, num_layers, seq_len, cache_len, batch_size: int = 1
    ) -> List[leap.TensorType]:
        batch_size = 1 if batch_size <= 1 else batch_size
        input_types = []
        inputs_embeds = leap.TensorType(
            [batch_size, seq_len, self.config.hidden_size], leap.float16
        )  # noqa: E501
        inputs = inputs_embeds

        attention_mask = leap.TensorType([batch_size, seq_len, cache_len], leap.float16)
        position_ids = leap.TensorType([batch_size, 3, seq_len], leap.int32)
        input_types.append(inputs)
        input_types.append(position_ids)
        input_types.append(attention_mask)

        cache_keys = []
        cache_values = []
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        for _ in range(num_layers):
            cache_key = leap.TensorType(
                [batch_size, cache_len, self.config.num_key_value_heads, head_dim],
                leap.float32,
            )  # noqa: E501
            cache_keys.append(cache_key)
            cache_value = leap.TensorType(
                [batch_size, cache_len, self.config.num_key_value_heads, head_dim],
                leap.float32,
            )  # noqa: E501
            cache_values.append(cache_value)
        caches = cache_keys + cache_values
        input_types.append(caches)
        return input_types

    def get_leap_input_types_decode_model(
        self, num_layers, seq_len, cache_len, batch_size: int = 1
    ) -> List[leap.TensorType]:
        batch_size = 1 if batch_size <= 1 else batch_size
        input_types = []
        inputs_embeds = leap.TensorType(
            [batch_size, seq_len, self.config.hidden_size], leap.float16
        )  # noqa: E501
        inputs = inputs_embeds

        attention_mask = leap.TensorType([batch_size, seq_len, cache_len], leap.float16)
        position_ids = leap.TensorType([batch_size, 1, seq_len], leap.int32)
        input_types.append(inputs)
        input_types.append(position_ids)
        input_types.append(attention_mask)

        cache_keys = []
        cache_values = []
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        for _ in range(num_layers):
            cache_key = leap.TensorType(
                [batch_size, cache_len, self.config.num_key_value_heads, head_dim],
                leap.float32,
            )  # noqa: E501
            cache_keys.append(cache_key)
            cache_value = leap.TensorType(
                [batch_size, cache_len, self.config.num_key_value_heads, head_dim],
                leap.float32,
            )  # noqa: E501
            cache_values.append(cache_value)
        caches = cache_keys + cache_values
        input_types.append(caches)
        return input_types


class Qwen2_5_VLModel(Model):
    def __init__(self, config: Qwen2_5_VLConfig, use_plugin: False):
        super().__init__()
        self.config = config
        self.visual = Qwen2_5_VLVisionModel(config.vision_config, use_plugin=use_plugin)
        self.language_model = Qwen2_5_VLTextModel(
            config.text_config, use_plugin=use_plugin
        )
        self.rope_deltas = None

    def get_image_feature(self, pixel_values, image_grid_thw):
        return self.visual(pixel_values)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_rotary_emb(self):
        return self.language_model.get_rotary_emb()

    def get_config(self):
        return self.config


class Qwen2_5_VLForConditionalGeneration(Model):
    def __init__(self, config: Qwen2_5_VLConfig, use_plugin):
        super().__init__()
        self.model = Qwen2_5_VLModel(config, use_plugin)

    def get_config(self):
        return self.model.config

    def get_visual_model(self):
        return self.model.visual

    def get_text_model(self):
        return self.model.language_model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_rotary_emb(self):
        return self.model.get_rotary_emb()


def remap_state_dict(state_dict, input_model_format="hf"):
    """
    Remap keys in state_dict depending on whether it is safetensors or not.
    """
    if input_model_format == "hf":
        mapping = {
            "model.visual.merger.mlp.0.weight": "model.visual.merger.mlp.proj0.weight",
            "model.visual.merger.mlp.0.bias": "model.visual.merger.mlp.proj0.bias",
            "model.visual.merger.mlp.2.weight": "model.visual.merger.mlp.proj1.weight",
            "model.visual.merger.mlp.2.bias": "model.visual.merger.mlp.proj1.bias",
            "lm_head.weight": "model.language_model.lm_head.weight",
        }
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = mapping.get(key, key)
            new_state_dict[new_key] = value
        return new_state_dict

    else:
        mapping = {
            "visual.merger.mlp.0.weight": "model.visual.merger.mlp.proj0.weight",
            "visual.merger.mlp.0.bias": "model.visual.merger.mlp.proj0.bias",
            "visual.merger.mlp.2.weight": "model.visual.merger.mlp.proj1.weight",
            "visual.merger.mlp.2.bias": "model.visual.merger.mlp.proj1.bias",
            "lm_head.weight": "model.language_model.lm_head.weight",
            "embed_tokens.weight": "model.language_model.embed_tokens.weight",
            "norm.weight": "model.language_model.norm.weight",
        }
        new_state_dict = {}
        for key, value in state_dict.items():
            key = mapping.get(key, key)
            if key.startswith("model."):
                new_key = key
            elif key.startswith("visual."):
                new_key = f"model.{key}"
            elif key.startswith("layers."):
                new_key = f"model.language_model.{key}"
            else:
                new_key = f"model.{key}"
            new_state_dict[new_key] = value

        # _tied_weights_keys
        # If lm_head.weight is not found in the new_state_dict,
        # tie it to embed_tokens.weight to share input/output
        # embedding weights (weight tying).
        lm_head_key = "model.language_model.lm_head.weight"
        if lm_head_key not in new_state_dict:
            new_state_dict[lm_head_key] = new_state_dict[
                "model.language_model.embed_tokens.weight"
            ]  # noqa: E501
        return new_state_dict


class Qwen2_5_VL:
    def __init__(
        self, model: Qwen2_5_VLForConditionalGeneration, model_args: Qwen2_5_VLConfig
    ):
        self.model = model
        self.model_args = model_args

    @staticmethod
    @timeit
    def build(
        model_dir: str,
        chunk_size: int = 256,
        batch_size: int = 1,
        cache_len: int = 4096,
        use_plugin: bool = False,
        w_bits: int = 8,
        mask_value: float = -32768,
        input_model_format: str = "hf",
        image_width: int = 448,
        image_height: int = 448,
        decode_seq_len: int = 1,
    ) -> "Qwen2_5_VL":
        assert os.path.isdir(
            model_dir
        ), f"Checkpoint directory '{model_dir}' does not exist."
        if input_model_format == "hf":
            checkpoints = sorted(Path(model_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {model_dir}"
            ckpt_path = checkpoints[0]  # No parallel
            print(f"loading {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
        config_path = os.path.join(model_dir, "config.json")
        assert os.path.exists(config_path), f"config.json not found in {model_dir}"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        model_args = Qwen2_5_VLConfig()
        vision_config = config.get("vision_config", config)
        text_config = config.get("text_config", config)
        model_args.vision_config = dataclass_from_dict(
            Qwen2_5_VLVisionConfig, vision_config
        )
        model_args.text_config = dataclass_from_dict(Qwen2_5_VLTextConfig, text_config)

        model_args.text_config.prefill_seq_len = chunk_size
        model_args.text_config.cache_len = cache_len
        model_args.text_config.decode_seq_len = decode_seq_len
        model_args.text_config.batch_size = batch_size
        model_args.vision_config.image_height = image_height
        model_args.vision_config.image_width = image_width
        model_args.vision_config.mask_min_value = mask_value
        model_args.text_config.w_bits = w_bits

        if input_model_format == "llmc":
            state_dict = load_safetensors_state_dict(model_dir)
            has_scale = any(".scales" in k for k in state_dict)
            model_args.text_config.has_scale = has_scale

        model = Qwen2_5_VLForConditionalGeneration(model_args, use_plugin)

        new_state_dict = remap_state_dict(
            state_dict, input_model_format=input_model_format
        )
        miss_key, unexpected_key = model.load_state_dict(new_state_dict, False)
        print(f"miss_key: {miss_key}")
        print(f"unexpected_key: {unexpected_key}")
        return Qwen2_5_VL(model, model_args)

    def get_visual_model(self):
        return self.model.get_visual_model()

    def get_text_model(self):
        return self.model.get_text_model()

    def compile(
        self,
        output_lm_model_path: str,
        output_vit_model_path: str,
        enable_vpu=True,
        vit_core_num: list[int] = None,
        prefill_core_num: list[int] = None,
        decode_core_num: list[int] = None,
        vit_kwargs=None,
        llm_kwargs=None,
    ):
        if decode_core_num is None:
            decode_core_num = [1]
        if prefill_core_num is None:
            prefill_core_num = [1]
        if vit_core_num is None:
            vit_core_num = [1]
        assert self.model.is_compiled, "Model must be compiled before compiling."

        def _validate_single_value_list(name: str, values: list[int]):
            if not isinstance(values, list):
                raise ValueError(f"{name} must be a list of int, got {type(values)}")
            if len(values) != 1:
                raise ValueError(
                    f"{name} must be a list of length 1, got {len(values)}: {values}"
                )

        _validate_single_value_list("vit_core_num", vit_core_num)
        _validate_single_value_list("prefill_core_num", prefill_core_num)
        _validate_single_value_list("decode_core_num", decode_core_num)

        stage_core_map = {
            "visual": vit_core_num[0],
            "prefill": prefill_core_num[0],
            "decode": decode_core_num[0],
        }

        model_list = []
        stages = ["visual", "prefill", "decode"]
        print(f"stages: {stages}")
        config = self.model.get_config()
        num_layers = config.text_config.num_hidden_layers

        chunk_size = config.text_config.prefill_seq_len
        cache_len = config.text_config.cache_len
        batch_size = max(getattr(config.text_config, "batch_size", 1), 1)

        for stage_name in stages:
            if stage_name == "visual":
                sub_model = self.get_visual_model()
                high_precision_qpp = True
                inputs = sub_model.get_leap_input_types()
            if stage_name == "prefill":
                sub_model = self.get_text_model()
                high_precision_qpp = True
                inputs = sub_model.get_leap_input_types_text_model(
                    num_layers,
                    chunk_size,
                    cache_len,
                    batch_size=batch_size,
                )
            if stage_name == "decode":
                sub_model = self.get_text_model()
                high_precision_qpp = True
                inputs = sub_model.get_leap_input_types_decode_model(
                    num_layers,
                    config.text_config.decode_seq_len,
                    cache_len,
                    batch_size=batch_size,
                )
            if stage_name == "visual":
                bc_path = str(
                    Path(output_vit_model_path).with_suffix(f".{stage_name}.bc")
                )
            else:
                bc_path = str(
                    Path(output_lm_model_path).with_suffix(f".{stage_name}.bc")
                )
            bc_module = sub_model.export_module(
                inputs, stage_name, bc_path, high_precision_qpp=high_precision_qpp
            )
            model_list.append(bc_module)

        lm_hbos = []
        vit_hbos = []
        for bc_module in model_list:
            func_name = bc_module.functions[0].name
            if func_name == "visual":
                convert_bc_path = str(
                    Path(output_vit_model_path).with_suffix(f".{func_name}_convert.bc")
                )
                kwargs=vit_kwargs
            else:
                convert_bc_path = str(
                    Path(output_lm_model_path).with_suffix(f".{func_name}_convert.bc")
                )
                kwargs=llm_kwargs
            mlir_module = self.model.convert_mlir(
                bc_module,
                convert_bc_path,
                enable_vpu=enable_vpu,
                march=kwargs["march"],
                dynamic_quant=True,
            )
            func = mlir_module.functions[0]
            func.remove_io_op(["Dequantize", "Quantize"])
            if func_name == "visual":
                hbo_path = str(
                    Path(output_vit_model_path).with_suffix(f".{func_name}.hbo")
                )
            else:
                hbo_path = str(
                    Path(output_lm_model_path).with_suffix(f".{func_name}.hbo")
                )

            core_num = stage_core_map[func_name]
            kwargs["core_num"] = core_num
            if kwargs["core_num"] > 1:
                kwargs["max_l2m_size"] = 25165824
                print(
                    f"{func_name}, core_num: {core_num}, set max_l2m_size"
                )
            else:
                kwargs.pop("max_l2m_size", None)
                print(
                    f"{func_name}, core_num: {core_num}, del max_l2m_size"
                )
            print(f"kwargs: {kwargs}")
            hbo_model = self.model.compile_hbo(
                mlir_module, save_path=hbo_path, **kwargs
            )
            if func_name == "visual":
                vit_hbos.append(hbo_model)
            else:
                lm_hbos.append(hbo_model)
        if len(lm_hbos) > 0:
            self.model.link_models(lm_hbos, save_path=output_lm_model_path)
        if len(vit_hbos) > 0:
            self.model.link_models(vit_hbos, save_path=output_vit_model_path)


def dequantize_weight(mod):
    num_itr = mod.g_idx.shape[0] // mod.in_features
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(mod.qzeros, 2).expand(-1, -1, mod.pack_factor),
        mod.wf_unsqueeze_zero,
    ).to(mod.dequant_dtype)
    zeros = torch.bitwise_and(zeros, mod.maxq).reshape(mod.scales.shape)

    weight = torch.bitwise_and(
        torch.bitwise_right_shift(
            torch.unsqueeze(mod.qweight, 1).expand(-1, mod.pack_factor, -1),
            mod.wf_unsqueeze_neg_one,
        ).to(mod.dequant_dtype),
        mod.maxq,
    )
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    if num_itr == 1:
        weights = mod.scales[mod.g_idx.long()] * (weight - zeros[mod.g_idx.long()])
    else:
        num_dim = mod.g_idx.shape[0] // num_itr
        weights = []
        for i in range(num_itr):
            scale_i = mod.scales[:, i * num_dim : (i + 1) * num_dim]
            weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
            zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
            g_idx_i = mod.g_idx[i * num_dim : (i + 1) * num_dim].long()
            weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
        weights = torch.cat(weights, dim=1)

    return weights


def replace_by_nn_layers(model):
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    for name, module in list(model.named_modules()):
        if isinstance(module, TorchQuantLinear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            weight = dequantize_weight(module).to(torch.float16).t()
            new_linear = nn.Linear(
                module.in_features, module.out_features, bias=(module.bias is not None)
            )
            new_linear.weight.data.copy_(weight)
            if module.bias is not None:
                new_linear.bias.data.copy_(module.bias.data)
            setattr(parent, child_name, new_linear)
    return model


def save_model_checkpoint(
    model_dir, output_model_path, load_int8_ckpt=False, dtype=torch.float32
):
    print(model_dir, output_model_path)
    ckpt_dir = os.path.join(output_model_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    dtype = torch.float16 if load_int8_ckpt else dtype
    ckpt_path = os.path.join(ckpt_dir, "model_checkpoint.pth")
    print(f"loading {ckpt_path} with {dtype}.")
    if not os.path.exists(ckpt_path):
        device = "cpu"
        model = hf_Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=device, attn_implementation="eager"
        )
        if load_int8_ckpt:
            start_time = time.time()
            model = replace_by_nn_layers(model)
            end_time = time.time()
            print(
                "Successfully replace Quant_linear by nn.Linear, costs {} seconds.".format(  # noqa
                    end_time - start_time
                )
            )
        torch.save(model.state_dict(), ckpt_path)
        print(f"Save checkpoint path: {ckpt_path}")
    config_json_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_json_path):
        shutil.copyfile(os.path.join(model_dir, "config.json"), config_json_path)

    return ckpt_dir
