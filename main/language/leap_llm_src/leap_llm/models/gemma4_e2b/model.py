import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.blocks.audio_conv import Gemma4AudioConvProjection
from leap_llm.models.gemma4_e2b.blocks.audio_decoder_layer import Gemma4AudioLayer
from leap_llm.models.gemma4_e2b.blocks.audio_pos_embedding import Gemma4AudioRelPositionalEncoding
from leap_llm.models.gemma4_e2b.blocks.mm_embedder import Gemma4MultimodalEmbedder
from leap_llm.models.gemma4_e2b.blocks.rmsnorm import Gemma4RMSNorm
from leap_llm.models.gemma4_e2b.blocks.text_decoder_layer import Gemma4TextDecoderLayer
from leap_llm.models.gemma4_e2b.blocks.text_rotary_embedding import Gemma4TextRotaryEmbedding
from leap_llm.models.gemma4_e2b.blocks.text_scaled_embedding import Gemma4TextScaledWordEmbedding
from leap_llm.models.gemma4_e2b.blocks.vision_encoder import Gemma4VisionEncoder
from leap_llm.models.gemma4_e2b.blocks.vision_patch_embedding import Gemma4VisionPatchEmbedder
from leap_llm.models.gemma4_e2b.blocks.vision_pooler import Gemma4VisionPooler
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Model, load_safetensors_state_dict, timeit

DUMP_DIR = "/tmp/gemma4_dump"


def dump_tensor(name, tensor, step=None):
    """Dump a tensor to a .npy file and print stats."""
    if tensor is None:
        return
    arr = tensor.detach().cpu().float().numpy()
    prefix = f"layer{step}_" if step is not None else ""
    filepath = os.path.join(DUMP_DIR, f"{prefix}{name}.npy")
    os.makedirs(DUMP_DIR, exist_ok=True)
    np.save(filepath, arr)
    print(
        f"  [DUMP] {prefix}{name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}"
    )


def create_logger(name, log_lvl=logging.INFO):
    logger = logging.getLogger(name)
    handle = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s:%(lineno)d %(message)s")
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    logger.setLevel(log_lvl)
    return logger


logger = create_logger(__name__)


class Gemma4AudioModel(Model):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config
        self.subsample_conv_projection = Gemma4AudioConvProjection(config)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [Gemma4AudioLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.output_proj = DynamicQuantLinear(config.hidden_size, config.output_proj_dims, bias=True)
        self.embed_audio = Gemma4MultimodalEmbedder(config)

    def forward(
        self,
        input_features: torch.Tensor,
        mask_conv_layer_0: torch.Tensor,
        mask_conv_layer_1: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        hidden_states = self.subsample_conv_projection(input_features, mask_conv_layer_0, mask_conv_layer_1)
        logger.debug(f"[after conv_subsample] hidden_states shape: {hidden_states.shape}")
        position_embedding = self.rel_pos_enc.audio_pe.to(input_features.device)
        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                position_embedding,
            )
        output = self.output_proj(hidden_states)
        output = self.embed_audio(output)
        return output

    def build(
        self,
        input_features,
        mask_conv_layer_0,
        mask_conv_layer_1,
        attention_mask,
    ):
        hidden_states = self.subsample_conv_projection(input_features, mask_conv_layer_0, mask_conv_layer_1)
        position_embedding = self.rel_pos_enc.audio_pe.to(torch.float16).cpu()
        if hidden_states.type.element_type == "f16":
            logger.debug("converting position_embedding to f16")
            position_embedding = position_embedding.to(torch.float16)
        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                position_embedding,
            )
        output = self.output_proj(hidden_states)
        return output

    def get_leap_input_types(self):
        bs = 1
        seq_len = 3000
        feature_dim = 128

        audio_feature = leap.TensorType([bs, 1, seq_len, feature_dim], leap.float16)

        mask_conv_0 = leap.TensorType([bs, 1, seq_len, 1], leap.float16)

        mask_conv_1 = leap.TensorType([bs, 1, seq_len // 2, 1], leap.float16)

        attention_mask = leap.TensorType([bs, 63, 12, 24], leap.float16)  # NOTE: 63 should be inferred from seq_len

        input_types = [
            audio_feature,
            mask_conv_0,
            mask_conv_1,
            attention_mask,
        ]

        return input_types

    def rename_graph_io(self, graph):
        pass

    def compile(
        self,
        output_model_path: str,
        audio_core_num: int,
        **kwargs,
    ):
        stage_name = "audio_tower"
        assert self.is_compiled, "Model must in compile mode"
        hbo_list = []
        inputs = self.get_leap_input_types()
        compile_cfg = kwargs.copy()
        compile_cfg["core_num"] = audio_core_num
        bc_path = str(Path(output_model_path).with_suffix(f".{stage_name}.bc"))
        mlir_path = str(Path(output_model_path).with_suffix(f".{stage_name}.mlir.bc"))
        hbo_path = str(Path(output_model_path).with_suffix(f".{stage_name}.hbo"))
        # 1. export module
        bc_module = self.export_module(inputs, stage_name, bc_path)

        # 2. mlir conversion
        mlir_module = self.convert_mlir(
            bc_module,
            mlir_path,
            dynamic_quant=True,
            enable_spu=False,
            enable_vpu=compile_cfg["enable_vpu"],
            march=compile_cfg["march"],
        )
        func = mlir_module.functions[0]
        func.remove_io_op(["Dequantize", "Quantize"])

        # graph = mlir_module.graphs[0]
        # self.rename_graph_io(graph, stage_name=stage_name)
        # enable hpc by default
        compile_cfg["enable_hpc"] = True
        if compile_cfg["core_num"] > 1:
            compile_cfg["max_l2m_size"] = 25165824

        logger.info(f"compile config:\n{compile_cfg}")

        hbo = self.compile_hbo(mlir_module, hbo_path, **compile_cfg)
        hbo_list.append(hbo)

        return self.link_models(hbo_list, output_model_path)


class Gemma4VisionModel(Model):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        text_hidden_size: int = 1536,
    ):
        super().__init__()
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)
        self.embed_vision = Gemma4MultimodalEmbedder(config, text_hidden_size)

    def forward(self, pixel_values):
        inputs_embeds = self.patch_embedder(pixel_values)
        hidden_states = self.encoder(inputs_embeds)
        output = self.pooler(hidden_states)
        output = self.embed_vision(output)
        return output

    def build(self, pixel_values):
        inputs_embeds = self.patch_embedder(pixel_values)
        hidden_states = self.encoder(inputs_embeds)
        output = self.pooler(hidden_states)
        output = self.embed_vision(output)
        return output

    def get_leap_input_types(self, num_patches):
        input_types = [leap.TensorType([1, num_patches, 768], leap.float16)]
        return input_types

    def rename_graph_io(self, graph):
        pass

    def compile(
        self,
        output_model_path: str,
        vision_core_num: int,
        num_patches: int,
        **kwargs,
    ):
        stage_name = "vision"
        assert self.is_compiled, "Model must in compile mode"
        hbo_list = []
        inputs = self.get_leap_input_types(num_patches)
        compile_cfg = kwargs.copy()
        compile_cfg["core_num"] = vision_core_num
        bc_path = str(Path(output_model_path).with_suffix(f".{stage_name}.bc"))
        mlir_path = str(Path(output_model_path).with_suffix(f".{stage_name}.mlir.bc"))
        hbo_path = str(Path(output_model_path).with_suffix(f".{stage_name}.hbo"))
        # 1. export module
        bc_module = self.export_module(inputs, stage_name, bc_path)

        # 2. mlir conversion
        mlir_module = self.convert_mlir(
            bc_module,
            mlir_path,
            dynamic_quant=True,
            enable_spu=False,
            enable_vpu=compile_cfg["enable_vpu"],
            march=compile_cfg["march"],
        )
        func = mlir_module.functions[0]
        func.remove_io_op(["Dequantize", "Quantize"])

        # graph = mlir_module.graphs[0]
        # self.rename_graph_io(graph, stage_name=stage_name)
        # enable hpc by default
        compile_cfg["enable_hpc"] = True
        if compile_cfg["core_num"] > 1:
            compile_cfg["max_l2m_size"] = 25165824

        logger.info(f"compile config:\n{compile_cfg}")

        hbo = self.compile_hbo(mlir_module, hbo_path, **compile_cfg)
        hbo_list.append(hbo)

        return self.link_models(hbo_list, output_model_path)


class Gemma4TextModel(Model):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, layer_idx, logger) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.unique_layer_types = set(self.config.layer_types)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = DynamicQuantLinear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
                w_bits=config.w_bits,
                has_scale=config.has_scale,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        self.lm_head = DynamicQuantLinear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_per_layer_input_embeddings(self):
        return self.embed_tokens_per_layer if self.embed_tokens_per_layer else None

    def forward(
        self,
        input_ids,
        inputs_embeds,  # For multi-modality inference
        per_layer_inputs,  # For multi-modality inference
        fa_mask,
        sa_mask,
        position_ids,
        *past_key_values,
    ):
        """Forward pass. For compilation, past_key_values are passed as *args."""

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Convert tuple to list for indexing
        past_key_values = list(past_key_values)

        if input_ids is not None:
            # TextModel Only inference branch
            inputs_embeds = self.embed_tokens(input_ids)
            if self.hidden_size_per_layer_input:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
                # project_per_layer_inputs is the up-projection (Linear hidden→num_layers*ple_dim),
                # reshape, and RMSNorm. The decoder_layer does the down-projection
                # (gate, multiply, project back to hidden, norm).
                per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if (inputs_embeds is not None) and (per_layer_inputs is not None) and (input_ids is None):
            # Multi-modality inference branch: API passes identity-only PLE,
            # we still need to apply the up-projection before passing to the layer.
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        hidden_states = inputs_embeds

        # NOTE: only prev_layers has Key / Value caching
        num_prev_layers = self.config.num_hidden_layers - self.config.num_kv_shared_layers

        past_keys = past_key_values[:num_prev_layers]
        past_values = past_key_values[num_prev_layers:]

        shared_kv_states = {}  # dict of tuple, {layer_idx: (key_states, value_states)}
        new_keys, new_values = [], []

        sa_cos = self.rotary_emb.sliding_attention_cos  # [1, cache_len, 256]
        sa_sin = self.rotary_emb.sliding_attention_sin
        fa_cos = self.rotary_emb.full_attention_cos  # [1, cache_len, 512]
        fa_sin = self.rotary_emb.full_attention_sin

        # position_ids: [bs, seq_len] → index into cos/sin dim=1
        pos_idx = position_ids.long().unsqueeze(-1)  # [bs, seq_len, 1]
        fa_pe = (
            torch.gather(fa_cos, 1, pos_idx.expand(-1, -1, fa_cos.size(-1))).to(dtype=inputs_embeds.dtype),
            torch.gather(fa_sin, 1, pos_idx.expand(-1, -1, fa_sin.size(-1))).to(dtype=inputs_embeds.dtype),
        )
        sa_pe = (
            torch.gather(sa_cos, 1, pos_idx.expand(-1, -1, sa_cos.size(-1))).to(dtype=inputs_embeds.dtype),
            torch.gather(sa_sin, 1, pos_idx.expand(-1, -1, sa_sin.size(-1))).to(dtype=inputs_embeds.dtype),
        )

        per_layer_input = None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            logger.debug(f"Running layer[{i}]")
            if per_layer_inputs is not None:
                per_layer_input = per_layer_inputs[:, :, i, :]

            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                per_layer_input,
                fa_mask=fa_mask,
                sa_mask=sa_mask,
                fa_position_embeddings=fa_pe,
                sa_position_embeddings=sa_pe,
                past_key=past_keys[i] if i < num_prev_layers else None,
                past_value=past_values[i] if i < num_prev_layers else None,
                shared_kv_states=shared_kv_states,
            )

            if new_key is not None and new_value is not None:
                logger.debug(f"new Key / Value states as output in layer [{i}]")
                new_keys.append(new_key)
                new_values.append(new_value)

        hidden_states = self.norm(hidden_states)
        # dump_tensor("final_norm_output", hidden_states)
        logits = self.lm_head(hidden_states)
        # dump_tensor("logits", logits)

        return logits, *new_keys, *new_values

    def build(
        self,
        inputs_embeds,
        per_layer_input_embeds,
        fa_mask,
        sa_mask,
        position_ids,
        *past_key_values,
    ):
        """Gemma4TextModel leap.build() flow
        inputs_embeds, per_layer_input_embeds are both scaled.
        """
        # Convert tuple to list for indexing
        past_key_values = list(past_key_values) if past_key_values else None

        bs, seq_len, _ = inputs_embeds.type.shape

        per_layer_input_embeds = leap.reshape(
            per_layer_input_embeds,
            [
                bs,
                seq_len,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            ],
        )
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = leap.mul(per_layer_projection, self.per_layer_model_projection_scale)
        per_layer_projection = leap.reshape(
            per_layer_projection,
            [
                bs,
                seq_len,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            ],
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        per_layer_inputs = leap.add(per_layer_projection, per_layer_input_embeds)
        per_layer_inputs = leap.mul(per_layer_inputs, self.per_layer_input_scale)

        pos_idx = leap.reshape(position_ids, [bs, seq_len, 1])

        fa_cos = self.rotary_emb.full_attention_cos
        fa_sin = self.rotary_emb.full_attention_sin
        sa_cos = self.rotary_emb.sliding_attention_cos
        sa_sin = self.rotary_emb.sliding_attention_sin

        # Gather RoPE embeddings based on position_ids
        fa_pe_cos = leap.gather_nd(fa_cos, pos_idx, batchDim=1)
        fa_pe_cos = leap.reshape(fa_pe_cos, [bs, seq_len, 1, fa_cos.size(-1)])
        fa_pe_sin = leap.gather_nd(fa_sin, pos_idx, batchDim=1)
        fa_pe_sin = leap.reshape(fa_pe_sin, [bs, seq_len, 1, fa_sin.size(-1)])
        sa_pe_cos = leap.gather_nd(sa_cos, pos_idx, batchDim=1)
        sa_pe_cos = leap.reshape(sa_pe_cos, [bs, seq_len, 1, sa_cos.size(-1)])
        sa_pe_sin = leap.gather_nd(sa_sin, pos_idx, batchDim=1)
        sa_pe_sin = leap.reshape(sa_pe_sin, [bs, seq_len, 1, sa_sin.size(-1)])

        fa_position_embeddings = (fa_pe_cos, fa_pe_sin)
        sa_position_embeddings = (sa_pe_cos, sa_pe_sin)

        # Prepare KV cache
        num_prev_layers = self.config.num_hidden_layers - self.config.num_kv_shared_layers
        past_keys = past_key_values[:num_prev_layers] if past_key_values else None
        past_values = past_key_values[num_prev_layers:] if past_key_values else None

        # KV sharing dict for storing full-length K/V
        shared_kv_states = {}

        new_keys, new_values = [], []

        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = leap.slice(
                per_layer_inputs,
                [0, 0, i, 0],
                [bs, seq_len, i + 1, self.config.hidden_size_per_layer_input],
                [1, 1, 1, 1],
            )
            per_layer_input = leap.reshape(
                per_layer_input,
                [bs, seq_len, self.config.hidden_size_per_layer_input],
            )

            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                per_layer_input,
                fa_mask,
                sa_mask,
                fa_position_embeddings,
                sa_position_embeddings,
                past_keys[i] if past_keys is not None and i < len(past_keys) else None,
                past_values[i] if past_values is not None and i < len(past_values) else None,
                shared_kv_states,
            )

            if new_key is not None and new_value is not None:
                new_keys.append(new_key)
                new_values.append(new_value)

        # Final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, *new_keys, *new_values

    def get_per_layer_inputs(self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> torch.Tensor:
        """Compute the token-identity component of Per-Layer Embeddings (PLE).

        Looks up `input_ids` in `embed_tokens_per_layer` (a scaled embedding that multiplies
        by `sqrt(hidden_size_per_layer_input)`) and reshapes the packed output from
        `[batch, seq, num_hidden_layers * hidden_size_per_layer_input]` to
        `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`.

        If only `inputs_embeds` is provided (no `input_ids`), reverses the main embedding
        to recover `input_ids` for the PLE lookup.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call get_per_layer_inputs() from a model initialized with a config that does not support"
                f" per-layer embeddings. {self.config}"
            )

        # If only inputs_embeds are provided, reverse main embedding to find the input_ids - this allows to `generate`
        # from `inputs_embeds` only as other models (otherwise it would need the value from both embeddings)
        if input_ids is None:
            with torch.no_grad():
                input_ids = (
                    (
                        inputs_embeds[:, :, None, :]
                        == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5
                    )
                    .all(dim=3)
                    .nonzero()[:, 2]
                )
                try:
                    input_ids = input_ids.view(inputs_embeds.shape[:2])
                except RuntimeError as err:
                    raise RuntimeError(
                        "It seems like you tried to call `forward` from `inputs_embeds`"
                        "without providing `input_ids`, and that "
                        "the `inputs_embeds` you provided do not exactly match the embedding weights. "
                        "Since Gemma4 needs to reverse "
                        "the embedding to compute another embedding, make sure you provide exact `inputs_embeds`"
                    ) from err

        ple = self.embed_tokens_per_layer(input_ids)

        ple = ple.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return ple

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the context-aware component of PLE and combine with token-identity.

        Projects `inputs_embeds` through `per_layer_model_projection` (Linear), scales by
        `1/sqrt(hidden_size)`, reshapes to `[batch, seq, num_layers, ple_dim]`, and normalizes
        with `per_layer_projection_norm` (RMSNorm).

        If `per_layer_inputs` (the token-identity component from `get_per_layer_inputs()`)
        is provided, combines both: `(context_projection + token_identity) * (1/sqrt(2))`.
        If `per_layer_inputs` is None (e.g. for multimodal inputs where input_ids are not
        available), returns just the context projection.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call project_per_layer_inputs() from a model initialized with a config that does not"
                f" support per-layer embeddings. {self.config}"
            )

        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def get_leap_input_types(self, seq_len: int):
        bs = 1
        # when prefill, seq_len = chunk_size,
        # when decode, seq_len = 1
        # inputs_embeds: [1, seq_len, hidden_size_per_layer_input] float32
        input_embeds = leap.TensorType([bs, seq_len, self.config.hidden_size], leap.float16)

        per_layer_input_embeds = leap.TensorType(
            [bs, seq_len, self.config.num_hidden_layers * self.config.hidden_size_per_layer_input],
            leap.float16,
        )

        # fa_mask: [1, 1, seq_len, cache_len] float16
        fa_mask_type = leap.TensorType([1, 1, seq_len, self.config.cache_len], leap.float16)

        # sa_mask: [1, 1, seq_len, 2 * sliding_window] float16
        sa_mask_type = leap.TensorType([1, 1, seq_len, 2 * self.config.sliding_window], leap.float16)

        # position_ids: [1, seq_len] int32
        position_ids_type = leap.TensorType([1, seq_len], leap.int32)

        input_types = [
            input_embeds,
            per_layer_input_embeds,
            fa_mask_type,
            sa_mask_type,
            position_ids_type,
        ]

        # Add past_key_values types
        num_prev_layers = self.config.num_hidden_layers - self.config.num_kv_shared_layers
        k_types = []
        v_types = []
        for i in range(num_prev_layers):
            if self.config.layer_types[i] == "sliding_attention":
                k_type = leap.TensorType(
                    [
                        bs,
                        2 * self.config.sliding_window,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ],
                    leap.float32,
                )
                v_type = leap.TensorType(
                    [
                        bs,
                        2 * self.config.sliding_window,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ],
                    leap.float32,
                )
            else:  # full_attention
                k_type = leap.TensorType(
                    [
                        bs,
                        self.config.cache_len,
                        self.config.num_key_value_heads,
                        self.config.global_head_dim,
                    ],
                    leap.float32,
                )
                v_type = leap.TensorType(
                    [
                        bs,
                        self.config.cache_len,
                        self.config.num_key_value_heads,
                        self.config.global_head_dim,
                    ],
                    leap.float32,
                )
            k_types.append(k_type)
            v_types.append(v_type)

        input_types.extend(k_types)
        input_types.extend(v_types)

        return input_types

    def rename_graph_io(self, graph):
        pass

    def compile(
        self,
        output_model_path: str,
        prefill_core_num: int,
        decode_core_num: int,
        **kwargs,
    ):
        assert self.is_compiled, "Model must in compile mode"
        hbo_list = []

        # compile several stages for the TextModel
        for stage_name in ["prefill", "decode"]:
            if stage_name == "prefill":
                logger.info(f"prefill stage: chunk_size={self.config.chunk_size} " f", ctx_len={self.config.cache_len}")
                inputs = self.get_leap_input_types(seq_len=self.config.chunk_size)
                compile_cfg = kwargs.copy()
                compile_cfg["core_num"] = prefill_core_num
            elif stage_name == "decode":
                logger.info(f"decode stage: chunk_size=1 " f", ctx_len={self.config.cache_len}")
                inputs = self.get_leap_input_types(seq_len=1)
                compile_cfg = kwargs.copy()
                compile_cfg["core_num"] = decode_core_num

            bc_path = str(Path(output_model_path).with_suffix(f".{stage_name}.bc"))
            mlir_path = str(Path(output_model_path).with_suffix(f".{stage_name}.mlir.bc"))
            hbo_path = str(Path(output_model_path).with_suffix(f".{stage_name}.hbo"))
            # 1. export module
            bc_module = self.export_module(inputs, stage_name, bc_path)
            # 2. mlir conversion
            mlir_module = self.convert_mlir(
                bc_module,
                mlir_path,
                dynamic_quant=True,
                enable_spu=False,
                enable_vpu=compile_cfg["enable_vpu"],
                march=compile_cfg["march"],
            )
            func = mlir_module.functions[0]
            func.remove_io_op(["Dequantize", "Quantize"])
            # graph = mlir_module.graphs[0]
            # self.rename_graph_io(graph, stage_name=stage_name)
            # enable hpc by default
            compile_cfg["enable_hpc"] = True
            if compile_cfg["core_num"] > 1:
                compile_cfg["max_l2m_size"] = 25165824
            logger.info(f"compile config:\n{compile_cfg}")

            hbo = self.compile_hbo(mlir_module, hbo_path, **compile_cfg)
            hbo_list.append(hbo)

        return self.link_models(hbo_list, output_model_path)


class Gemma4Model(Model):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.vision_tower = Gemma4VisionModel(config.vision_config)
        self.audio_tower = Gemma4AudioModel(config.audio_config)
        self.language_model = Gemma4TextModel(config.text_config)

    def get_image_features(self, pixel_values, image_position_ids):
        hidden_states = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
        )
        return hidden_states

    def get_audio_features(self, input_features, attention_mask=None):
        raise NotImplementedError("get_audio_features is not implemented")

    # NOTE: property name shall not be language_model, to avoid recursive calling
    @property
    def language(self):
        return self.language_model

    @property
    def visual(self):
        return self.vision_tower

    @property
    def audio(self):
        return self.audio_tower


class Gemma4ModelWrapper:
    def __init__(self, model, model_args):
        self.model: Gemma4Model = model
        self.model_args: Gemma4Config = model_args

    @staticmethod
    @timeit
    def build(
        model_dir,
        model_type,
        chunk_size,
        cache_len,
        w_bits,
        mask_value,
        output_model_dir,
        image_thw,
        march="nash-m",
        **kwargs,
    ):
        model, model_args = None, None
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_dir)
        model_args = Gemma4Config(**hf_config.to_dict())
        print(type(model_args))

        new_state_dict = load_safetensors_state_dict(
            model_dir,
            include_substrings=[
                # Gemma4TextModel
                "weight",
                "bias",
                "buf_scales",
                "layer_scalar",
                # Gemma4ClippableLinear weight fields
                "input_max",
                "input_min",
                "output_max",
                "output_min",
                "per_dim_scale",
                # Gemma4VisionModel
                "position_embedding_table",
                "residual_weight",
            ],
        )

        has_scale = any(".scales" in k for k in new_state_dict)

        if has_scale:
            warnings.warn(f"Current checkpoint contains quantization info, w_bits = {w_bits} bits", stacklevel=2)

        # -- vision config --
        # The HF checkpoint stores the multimodal embedder as a top-level
        # `model.embed_vision.*` module, but our model nests it inside
        # `vision_tower.embed_vision.*` (and similarly for audio). Rename the
        # keys so the module hierarchy matches.
        embed_vision_keys = [k for k in list(new_state_dict.keys()) if k.startswith("embed_vision.")]
        for k in embed_vision_keys:
            new_key = "vision_tower." + k
            new_state_dict[new_key] = new_state_dict.pop(k)

        # -- audio config --
        embed_audio_keys = [k for k in list(new_state_dict.keys()) if k.startswith("embed_audio.")]
        for k in embed_audio_keys:
            new_key = "audio_tower." + k
            new_state_dict[new_key] = new_state_dict.pop(k)

        # -- text config --
        if model_args.text_config.tie_word_embeddings:
            print("Tie word embeddings and lm head weights")
            new_state_dict["language_model.lm_head.weight"] = new_state_dict[
                "language_model.embed_tokens.weight"
            ].clone()

        embed_tokens_filename = os.path.join(output_model_dir, "Gemma-4-E2B-it_embed_tokens_fp16.bin")
        embed_tokens_per_layer_filename = os.path.join(
            output_model_dir, "Gemma-4-E2B-it_embed_tokens_per_layer_fp16.bin"
        )

        print(embed_tokens_filename)
        print(embed_tokens_per_layer_filename)

        # add HBM specific arguments
        model_args.text_config.chunk_size = chunk_size
        model_args.text_config.cache_len = cache_len
        model_args.vision_config.image_thw = image_thw

        model = Gemma4Model(model_args)
        model.load_state_dict(new_state_dict, strict=False)

        # NOTE: merge the `scale` in ScaledTextEmbedding into the dumping weight files.
        # get the scale after the model is loaded
        wt = new_state_dict["language_model.embed_tokens.weight"]
        embed_tokens_scale = model.language_model.embed_tokens.embed_scale
        logger.info(f"embed_tokens.embed_scale: {embed_tokens_scale}")
        wt = wt.data.float().detach().cpu() * embed_tokens_scale.float().detach().cpu()
        wt.to(torch.float16).numpy().tofile(embed_tokens_filename)

        wt = new_state_dict["language_model.embed_tokens_per_layer.weight"]
        embed_tokens_per_layer_scale = model.language_model.embed_tokens_per_layer.embed_scale
        logger.info(f"embed_tokens_per_layer.embed_scale: {embed_tokens_per_layer_scale}")
        wt = wt.data.float().detach().cpu() * embed_tokens_per_layer_scale.float().detach().cpu()
        wt.to(torch.float16).numpy().tofile(embed_tokens_per_layer_filename)

        return Gemma4ModelWrapper(model, model_args)

    def get_model(self):
        return self.model

    def get_model_args(self):
        return self.model_args
