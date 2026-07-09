import json
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import List

import torch
from hbdk4.compiler import leap, save
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from leap_llm.nn.modules import (
    DynamicQuantLinear,
    RMSNorm,
)
from leap_llm.nn.modules.conv import Conv2d
from leap_llm.nn.utils import Model, Module, load_safetensors_state_dict, timeit

from .blocks.mlp import InternProjcetMLP
from .blocks.transformer_block import InternVisionEncoderLayer, Qwen3DecoderLayer
from .configuration import (
    InternVL3_5Config,
    InternVL3_5LLMConfig,
    InternVL3_5VisionConfig,
)


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


class InternVisionEmbeddings(Module):
    def __init__(self, config: InternVL3_5VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim)
        )

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(
                1,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [*, channel, width, height]

        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        side = self.image_size // self.patch_size
        if height == side and width == side:
            position_embedding = self.position_embedding
        else:
            position_embedding = torch.cat(
                [
                    self.position_embedding[:, :1, :],
                    self._get_pos_embed(
                        self.position_embedding[:, 1:, :], height, width
                    ),
                ],
                dim=1,
            )
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings

    def build(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        hwc_img_pixel = leap.transpose(pixel_values, [0, 2, 3, 1])
        self.patch_embedding.to("cpu", dtype=torch.float32)
        dtype = hwc_img_pixel.type.element_type

        hwc_img_pixel = leap.cast_type(hwc_img_pixel, output_type=leap.float32)
        patch_embeds = self.patch_embedding(
            hwc_img_pixel
        )  # shape = [*, channel, width, height]
        patch_embeds = leap.cast_type(patch_embeds, output_type=dtype)

        batch_size, height, width, channel = patch_embeds.type.shape
        patch_embeds = leap.reshape(patch_embeds, (batch_size, height * width, -1))

        class_embeds = self.class_embedding.data
        embeddings = leap.concat([class_embeds, patch_embeds], dim=1)

        side = self.image_size // self.patch_size
        if height == side and width == side:
            position_embedding = self.position_embedding.data
        else:
            raise ValueError("Can not process here!")
        embeddings = leap.add(embeddings, position_embedding)
        return embeddings


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"

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
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            cos = cos.squeeze()
            sin = sin.squeeze()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class InternVisionEncoder(Module):
    def __init__(self, config: InternVL3_5VisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        self.layers = nn.ModuleList(
            [
                InternVisionEncoderLayer(config)
                for idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds,
    ):
        hidden_states = inputs_embeds
        for _idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
            )
            hidden_states = layer_outputs
        return hidden_states

    def build(
        self,
        inputs_embeds,
    ):
        hidden_states = inputs_embeds
        for _idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
            )
            hidden_states = layer_outputs
        return hidden_states


class InternVisionModel(Model):
    def __init__(
        self,
        config: InternVL3_5Config,
    ):
        super().__init__()
        self.config = config.vision_config

        self.embeddings = InternVisionEmbeddings(self.config)
        self.encoder = InternVisionEncoder(self.config)

        self.downsample_ratio = config.downsample_ratio

        llm_hidden_size = config.llm_config.hidden_size

        vit_hidden_size = (
            config.vision_config.hidden_size * int(1 / self.downsample_ratio) ** 2
        )
        self.mlp1 = InternProjcetMLP(vit_hidden_size, llm_hidden_size)

    def build(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)

        B, N, C = hidden_states.type.shape
        vit_embeds = leap.slice(hidden_states, [0, 1, 0], [B, N, C], [1, 1, 1])

        h = w = int(vit_embeds.type.shape[1] ** 0.5)
        vit_embeds = leap.reshape(vit_embeds, (B, h, w, -1))
        n, w, h, c = vit_embeds.type.shape
        scale_factor = self.downsample_ratio
        vit_embeds = leap.reshape(
            vit_embeds,
            (
                n,
                int(w * scale_factor),
                -1,
                int(h * scale_factor),
                int(c / scale_factor),
            ),
        )

        vit_embeds = leap.transpose(vit_embeds, (0, 1, 3, 2, 4))
        vit_embeds = leap.reshape(
            vit_embeds, (n, -1, int(c / (scale_factor * scale_factor)))
        )
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def get_leap_input_types(self, image_size) -> List[leap.TensorType]:
        dtype = leap.float16
        vision_input_types = [
            leap.TensorType(
                [
                    1,
                    3,
                    image_size,
                    image_size,
                ],
                dtype,
            ),
        ]
        return vision_input_types

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)

        vit_embeds = hidden_states[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        n, w, h, c = vit_embeds.size()
        scale_factor = self.downsample_ratio
        vit_embeds = vit_embeds.view(
            n, int(w * scale_factor), -1, int(h * scale_factor), int(c / scale_factor)
        )
        vit_embeds = vit_embeds.permute(0, 1, 3, 2, 4).contiguous()
        vit_embeds = vit_embeds.reshape(
            vit_embeds.shape[0], -1, int(c / (scale_factor * scale_factor))
        )
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


class Qwen3Model(Model):
    def __init__(self, config: InternVL3_5LLMConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.use_fastv = config.use_fastv
        self.fastv_k = config.fastv_k
        self.fastv_r = config.fastv_r
        self.image_token_start_index = config.image_token_start_index
        self.image_token_length = config.image_token_length
        self.max_prefill_tokens = config.max_prefill_tokens
        self.min_value = config.min_value
        self.fastv_max_cache_tokens = config.fastv_max_cache_tokens
        self.max_cache_tokens = config.max_cache_tokens

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, num_layer) for num_layer in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        pseudo_input_ids = torch.arange(config.max_cache_tokens).view(1, -1).float()
        cache_cos, cache_sin = self.rotary_emb(pseudo_input_ids, pseudo_input_ids)
        self.register_buffer("cache_cos", cache_cos, persistent=True)
        self.register_buffer("cache_sin", cache_sin, persistent=True)

    def get_input_embeddings(self):
        return self.embed_tokens

    def build(
        self, input_embeds, position_ids, attention_mask, cache_keys=None, cache_values=None
    ):
        if cache_values is None:
            cache_values = []
        if cache_keys is None:
            cache_keys = []
        hidden_states = input_embeds

        position_ids = leap.transpose(position_ids, (1, 0))
        cos = leap.gather_nd(self.cache_cos, position_ids, 0)
        sin = leap.gather_nd(self.cache_sin, position_ids, 0)
        position_embeddings = (cos, sin)

        new_keys = []
        new_values = []
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, new_key, new_value = decoder_layer(
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                hidden_states=hidden_states,
                cache_keys=cache_keys[idx] if len(cache_keys) else None,
                cache_values=cache_values[idx] if len(cache_values) else None,
            )
            new_keys.append(new_key)
            new_values.append(new_value)
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_keys, new_values

    def forward(
        self,
        input_embeds,
        position_ids,
        attention_mask,
        cache_keys=None,
        cache_values=None,
    ):
        if cache_values is None:
            cache_values = []
        if cache_keys is None:
            cache_keys = []
        hidden_states = input_embeds

        cos = self.cache_cos[position_ids]
        sin = self.cache_sin[position_ids]
        position_embeddings = (cos, sin)

        new_keys = []
        new_values = []
        for idx, decoder_layer in enumerate(self.layers):
            seq_len = hidden_states.shape[1]
            if self.use_fastv is True and idx == self.fastv_k:
                if seq_len != 1:
                    seq_length_with_past = hidden_states.shape[1]
                    img_start = -self.max_prefill_tokens + self.image_token_start_index
                    img_end = -self.max_prefill_tokens + (
                        self.image_token_start_index + self.image_token_length
                    )
                    attn_weights = attn_weights[:, :, :, img_start:img_end]  # noqa
                    attn_weights = torch.mean(attn_weights, dim=1)
                    max_idxes = position_ids.argmax()
                    attn_weights_topk = attn_weights[0, max_idxes]

                    topk = round(self.image_token_length * (1 - self.fastv_r))
                    top_attention_rank_index = attn_weights_topk.topk(topk).indices
                    top_attention_rank_index = (
                        top_attention_rank_index + self.image_token_start_index
                    )

                    token_before_img = torch.arange(
                        self.image_token_start_index, device=attn_weights.device
                    )
                    token_after_img = torch.arange(
                        self.image_token_start_index + self.image_token_length,
                        seq_length_with_past,
                        device=attn_weights.device,
                    )

                    keep_index = torch.cat(
                        [token_before_img, top_attention_rank_index, token_after_img]
                    )

                    hidden_states = hidden_states[:, keep_index, :]
                    position_embeddings = (
                        position_embeddings[0][:, keep_index, :],
                        position_embeddings[1][:, keep_index, :],
                    )

                    attention_mask = attention_mask[:, :, keep_index, :]

                    mask_start = self.max_cache_tokens - self.fastv_max_cache_tokens
                    chunk_attention_mask = attention_mask[
                        :, :, :, -self.max_prefill_tokens :
                    ]
                    before_chunk__mask = attention_mask[
                        :, :, :, mask_start : -self.max_prefill_tokens
                    ]
                    chunk_attention_mask = chunk_attention_mask[:, :, :, keep_index]

                    pad_mask_tokens = self.image_token_length - round(
                        self.image_token_length * (1 - self.fastv_r)
                    )

                    n, b, q, _ = attention_mask.shape
                    pad_mask = torch.zeros([n, b, q, pad_mask_tokens]).to(
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat(
                        [pad_mask, before_chunk__mask, chunk_attention_mask], dim=3
                    )
                else:
                    remove_tokens = self.image_token_length - round(
                        self.image_token_length * (1 - self.fastv_r)
                    )
                    n, b, q, _ = attention_mask.shape
                    pad_mask = (
                        torch.ones((n, b, q, remove_tokens)).to(
                            device=attention_mask.device
                        )
                        * self.min_value
                    )
                    mask_start = self.max_cache_tokens - self.fastv_max_cache_tokens
                    attention_mask = attention_mask[:, :, :, mask_start:-remove_tokens]
                    attention_mask = torch.cat([pad_mask, attention_mask], dim=3)

            hidden_states, attn_weights, new_key, new_value = decoder_layer(
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                hidden_states=hidden_states,
                cache_keys=cache_keys[idx] if len(cache_keys) else None,
                cache_values=cache_values[idx] if len(cache_values) else None,
            )
            new_keys.append(new_key)
            new_values.append(new_value)
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_keys, new_values


class Qwen3ForCausalLM(Model):
    def __init__(self, config: InternVL3_5LLMConfig):
        super().__init__()
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = DynamicQuantLinear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.config = config
        self.ppl_mode = os.getenv("INTERNVL_PPL_ENV", "").lower() == "true"
        if self.ppl_mode:
            print("ppl mode set true")
        else:
            print("ppl mode set false")

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def build(
        self,
        input_embeds,
        position_ids,
        attention_mask,
        *caches,
    ):
        cache_keys = caches[: len(caches) // 2]
        cache_values = caches[len(caches) // 2 :]
        hidden_states, new_keys, new_values = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )

        if not self.ppl_mode:
            seqlen = hidden_states.type.shape[1]
            position_ids = leap.reshape(position_ids, [seqlen])
            if seqlen > 1:
                vaild_lens = leap.reduce_argmax(position_ids, dims=[0], keepDim=True)
                hidden_states = leap.index(hidden_states, vaild_lens, dim=1)
        logits = self.lm_head(hidden_states)
        print("outputs shape ", logits.type)
        caches = new_keys + new_values
        return logits, *caches

    def forward(
        self,
        input_embeds,
        position_ids,
        attention_mask,
        cache_keys=None,
        cache_values=None,
    ):
        if cache_values is None:
            cache_values = []
        if cache_keys is None:
            cache_keys = []
        hidden_states, new_keys, new_values = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        if self.lm_head.weight.device != hidden_states.device:
            self.lm_head.to(hidden_states.device)
        logits = self.lm_head(hidden_states)
        return logits, new_keys, new_values

    def get_leap_input_types(self, seq_len, cache_len) -> List[leap.TensorType]:
        batch_size = 1
        hidden_size = self.config.hidden_size
        input_types = []
        inputs_embeds = leap.TensorType(
            [batch_size, seq_len, hidden_size], leap.float16
        )
        inputs = inputs_embeds

        attention_mask = leap.TensorType(
            [batch_size, 1, seq_len, cache_len], leap.float16
        )
        position_ids = leap.TensorType([batch_size, seq_len], leap.int32)
        input_types.append(inputs)
        input_types.append(position_ids)
        input_types.append(attention_mask)

        cache_keys = []
        cache_values = []

        num_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        for _ in range(num_layers):
            cache_key = leap.TensorType(
                [batch_size, cache_len, num_key_value_heads, head_dim], leap.float32
            )
            cache_keys.append(cache_key)
            cache_value = leap.TensorType(
                [batch_size, cache_len, num_key_value_heads, head_dim], leap.float32
            )
            cache_values.append(cache_value)
        input_types.append(cache_keys)
        input_types.append(cache_values)
        return input_types


class InterVL3_5Model(Model):
    def __init__(self, config: InternVL3_5Config):
        super().__init__()
        self.config = config
        self.vision_model = InternVisionModel(config)
        self.language_model = Qwen3ForCausalLM(config.llm_config)

    def get_vision_model(self):
        return self.vision_model

    def get_language_model(self):
        return self.language_model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_rotary_emb(self):
        return self.language_model.get_rotary_emb()

    def get_config(self):
        return self.config


class InterVL3_5:
    def __init__(self, model: InterVL3_5Model, model_args: InternVL3_5Config):
        self.model = model
        self.model_args = model_args

    @staticmethod
    @timeit
    def build(
        model_dir: str,
        chunk_size: int = 256,
        cache_len: int = 4096,
    ) -> "InterVL3_5":
        assert os.path.isdir(
            model_dir
        ), f"Checkpoint directory '{model_dir}' does not exist."

        device = "cpu"
        hf_model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            # use_flash_attn=True,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        checkpoint = hf_model.state_dict()

        scales_dict = load_safetensors_state_dict(
            model_dir,
            include_substrings=["buf_scales"],
            content_change_map={".buf_scales": ".scales"},
            prefix_remove_list=[],
        )
        for key, v in scales_dict.items():
            checkpoint[key] = v

        config_path = os.path.join(model_dir, "config.json")
        assert os.path.exists(config_path), f"config.json not found in {model_dir}"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        model_args = InternVL3_5Config()
        vision_config = config.get("vision_config", config)
        llm_config = config.get("llm_config", config)

        # 设置从外部传入的 chunk_size 和 cache_len
        llm_config["prefill_seq_len"] = chunk_size
        llm_config["decode_seq_len"] = 1
        llm_config["max_cache_tokens"] = cache_len

        model_args.vision_config = dataclass_from_dict(
            InternVL3_5VisionConfig, vision_config
        )
        model_args.llm_config = dataclass_from_dict(InternVL3_5LLMConfig, llm_config)

        if not scales_dict:
            model_args.llm_config.has_scale = False
        else:
            model_args.llm_config.has_scale = True
            print("float model has scale")

        model = InterVL3_5Model(model_args)
        mapping = {
            "mlp1.0.weight": "vision_model.mlp1.norm.weight",
            "mlp1.0.bias": "vision_model.mlp1.norm.bias",
            "mlp1.1.weight": "vision_model.mlp1.fc1.weight",
            "mlp1.1.bias": "vision_model.mlp1.fc1.bias",
            "mlp1.3.weight": "vision_model.mlp1.fc2.weight",
            "mlp1.3.bias": "vision_model.mlp1.fc2.bias",
        }
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = mapping.get(key, key)
            new_state_dict[new_key] = value

        miss_key, unexpected_key = model.load_state_dict(new_state_dict, False)
        print(f"miss_key: {miss_key}")
        print(f"unexpected_key: {unexpected_key}")
        return InterVL3_5(model, model_args)

    def get_vit_model(self):
        return self.model.get_vision_model()

    def get_language_model(self):
        return self.model.get_language_model()

    def set_calibration_mode(self):
        for _n, v in self.model.language_model.named_modules():
            if hasattr(v, "quantized"):
                v.quantized = True

    def set_float_mode(self):
        for _n, v in self.model.named_modules():
            if hasattr(v, "quantized"):
                v.quantized = False

    def compile(
        self,
        stage: str,
        output_model_path: str,
        enable_vpu=True,
        vit_core_num: list[int] = None,
        prefill_core_num: list[int] = None,
        decode_core_num: list[int] = None,
        **kwargs,
    ):
        if decode_core_num is None:
            decode_core_num = [1]
        if prefill_core_num is None:
            prefill_core_num = [1]
        if vit_core_num is None:
            vit_core_num = [1]
        assert self.model.is_compiled, "Model must be compiled before compiling."

        stages = []
        hbos = []
        if stage == "all":
            stages = ["vit", "prefill", "decode"]
        elif stage == "vit":
            stages = ["vit"]
        elif stage == "llm":
            stages = ["prefill", "decode"]
        else:
            raise ValueError(f"Invalid stage: {stage} in compile")
        for stage_name in stages:
            seq_len = None
            if stage_name == "vit":
                sub_model = self.get_vit_model()
                high_precision_qpp = True
                image_size = 448
                inputs = sub_model.get_leap_input_types(image_size)
                core_num_list = vit_core_num
            if stage_name == "prefill":
                sub_model = self.get_language_model()
                high_precision_qpp = True
                seq_len = self.model_args.llm_config.prefill_seq_len
                cache_len = self.model_args.llm_config.max_cache_tokens
                inputs = sub_model.get_leap_input_types(seq_len, cache_len)
                core_num_list = prefill_core_num
            if stage_name == "decode":
                sub_model = self.get_language_model()
                high_precision_qpp = True
                seq_len = self.model_args.llm_config.decode_seq_len
                cache_len = self.model_args.llm_config.max_cache_tokens
                inputs = sub_model.get_leap_input_types(seq_len, cache_len)
                core_num_list = decode_core_num

            for core_num in core_num_list:
                new_stage_name = stage_name + f"_core_{core_num}"
                bc_path = str(
                    Path(output_model_path).with_suffix(f".{new_stage_name}.bc")
                )
                print("bc_path:", bc_path)
                bc_module = sub_model.export_module(
                    inputs,
                    new_stage_name,
                    bc_path,
                    high_precision_qpp=high_precision_qpp,
                )

                convert_bc_path = str(
                    Path(output_model_path).with_suffix(f".{new_stage_name}_convert.bc")
                )
                mlir_module = self.model.convert_mlir(
                    bc_module,
                    convert_bc_path,
                    enable_vpu=enable_vpu,
                    march=kwargs["march"],
                    dynamic_quant=True,
                )
                func = mlir_module.functions[0]
                func.remove_io_op(["Dequantize", "Quantize"])
                convert_removed_bc_path = str(
                    Path(output_model_path).with_suffix(
                        f".{new_stage_name}_convert_rm.bc"
                    )
                )
                save(mlir_module, convert_removed_bc_path)
                hbo_path = str(
                    Path(output_model_path).with_suffix(f".{new_stage_name}.hbo")
                )
                kwargs["core_num"] = core_num
                if kwargs["core_num"] > 1:
                    kwargs["max_l2m_size"] = 25165824
                    print(f"{new_stage_name}, core_num: {core_num}, set max_l2m_size")
                else:
                    kwargs.pop("max_l2m_size", None)
                    print(f"{new_stage_name}, core_num: {core_num}, del max_l2m_size")
                print(f"kwargs : {kwargs}")
                hbo_model = self.model.compile_hbo(
                    mlir_module, save_path=hbo_path, **kwargs
                )
                hbos.append(hbo_model)
        self.model.link_models(hbos, save_path=output_model_path)
