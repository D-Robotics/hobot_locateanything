import importlib
import json
import logging
import os
import warnings
from logging import Logger
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from hbdk4.compiler import leap

from leap_llm.models.qwen3_vl.blocks.text_block import Qwen3VLTextDecoderLayer
from leap_llm.models.qwen3_vl.blocks.text_rotary_emb import Qwen3VLTextRotaryEmbedding
from leap_llm.models.qwen3_vl.blocks.vision_block import Qwen3VLVisionBlock
from leap_llm.models.qwen3_vl.blocks.vision_patch import (
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
)
from leap_llm.models.qwen3_vl.utils.visual_utils import (
    prepare_patch_pos_emb,
    vision_rotary_pos_emb,
)
from leap_llm.nn.modules.embedding import FakeQuantEmbedding
from leap_llm.nn.modules.linear import DynamicQuantLinear
from leap_llm.nn.modules.rms_norm import RMSNorm
from leap_llm.nn.utils import Model, load_safetensors_state_dict, timeit


class Qwen3VLVisionModel(Model):
    def __init__(self, config, logger: Logger = None):
        super().__init__()
        self.logger = logger
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.patch_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = FakeQuantEmbedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.rotary_dim = config.head_dim // 2
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config=config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )
        # TODO: hardcode for now, shall take grid_thw as input ?
        self.grid_thw = torch.tensor([[1, 28, 28]], device=self.merger.norm.weight.device)

        # NOTE: the device below needs sanitizing, since the
        # Qwen3VLVisionModel is not created yet.
        # to.(device) during forward() since calibration is faster using cuda
        # during build(), the device is cpu anyway, so
        # `on the same device` won't happen
        self._init_grid_pos_emb()
        self._init_rot_pos_emb()

    def _init_grid_pos_emb(self):
        (self.grid_h, self.grid_w, self.grid_idx, self.grid_wt) = prepare_patch_pos_emb(
            grid_thw=self.grid_thw,
            device=self.pos_embed.weight.device,
            wt_dtype=self.pos_embed.weight.dtype,
        )
        self.grid_wt = self.grid_wt[:, :, None]

    def _init_rot_pos_emb(self):
        rotary_pos_emb = vision_rotary_pos_emb(dim=self.rotary_dim, grid_thw=self.grid_thw, device=self.grid_thw.device)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        emb = emb.unsqueeze(-2).unsqueeze(0)
        self.position_embeddings = (emb.cos(), emb.sin())

    def gen_patch_pe_leap(self):
        """
        visual patch position embedding function for leap.
        NOTE: This function shall be converted to hbdk.constant after export.
        """
        self.logger.debug(f"grid_idx.shape = {self.grid_idx.shape}")
        # make grid indices broadcastable and continguous for numpy conversion
        gathered_feat = self.pos_embed(self.grid_idx.unsqueeze(-1).contiguous())
        self.logger.debug(f"gathered_feat.shape = {gathered_feat.type.shape}")
        pos_embeds = leap.mul(gathered_feat, self.grid_wt.contiguous())
        self.logger.debug(f"pos_embeds.shape = {pos_embeds.type.shape}")
        # (1, 784, 1024)
        patch_pos_embeds = leap.reduce_sum(pos_embeds, dims=[0])  # keepdim = True
        self.logger.debug(f"patch_pos_embeds.shape = {patch_pos_embeds.type.shape}")
        patch_pos_embeds = leap.reshape(
            patch_pos_embeds,
            [
                int(self.grid_h // self.spatial_merge_size),
                self.spatial_merge_size,
                int(self.grid_w // self.spatial_merge_size),
                -1,
            ],
        )
        patch_pos_embeds = leap.transpose(patch_pos_embeds, [0, 2, 1, 3])
        patch_pos_embeds = leap.reshape(patch_pos_embeds, [int(self.grid_h * self.grid_w), -1])
        # FIXME: convert to dtype according to param `dtype`
        patch_pos_embeds = leap.cast_type(patch_pos_embeds, output_type=leap.float16)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor):
        assert hidden_states.size(0) == 1, "pixel_value batch size shall be 1"

        hidden_states = hidden_states.squeeze(0)
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.pos_embed(self.grid_idx.to(device=hidden_states.device)) * self.grid_wt.to(
            device=hidden_states.device
        )
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.reshape(
            self.grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.grid_w // self.spatial_merge_size,
            -1,
        )
        patch_pos_embeds = patch_pos_embeds.transpose(1, 2)
        patch_pos_embeds = patch_pos_embeds.reshape(self.grid_h * self.grid_w, -1)
        hidden_states = hidden_states + patch_pos_embeds

        self.position_embeddings = tuple(e.to(hidden_states.device) for e in self.position_embeddings)

        deepstack_feature_lists = []

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, position_embeddings=self.position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature.unsqueeze(0))

        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states.unsqueeze(0)

        return hidden_states, deepstack_feature_lists

    def build(self, hidden_states):
        bs, seqlen, hs = hidden_states.type.shape
        assert bs == 1, "pixel_value batch size shall be 1"
        hidden_states = leap.reshape(hidden_states, [seqlen, hs])
        hidden_states = self.patch_embed(hidden_states)
        patch_pos_embeds = self.gen_patch_pe_leap()
        hidden_states = leap.add(hidden_states, patch_pos_embeds)
        hidden_states = leap.reshape(hidden_states, [bs, seqlen, -1])

        self.position_embeddings = tuple(e.to(device="cpu", dtype=torch.float16) for e in self.position_embeddings)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, position_embeddings=self.position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, *deepstack_feature_lists

    def get_leap_input_types(self):
        input_types = [leap.TensorType([1, 784, 1536], leap.float16)]
        return input_types

    def rename_graph_io(self, graph, num_deepstack_embeds: int = 3):
        """
        Rename graph flatten inputs and outputs for LLM KV-cache style models.
        """
        # -------- inputs --------
        input_names = ["pixel_values_image"]

        assert len(graph.flatten_inputs) >= len(input_names), "flatten_inputs size mismatch"

        for tensor, name in zip(graph.flatten_inputs, input_names):
            tensor.name = name

        # -------- outputs --------
        output_names = ["image_embed"]

        for i in range(num_deepstack_embeds):
            output_names.append(f"deepstack_embed_{i}")

        assert len(graph.flatten_outputs) >= len(output_names), "flatten_outputs size mismatch"

        for tensor, name in zip(graph.flatten_outputs, output_names):
            tensor.name = name

    def compile(self, output_model_path, core_num: int, **kwargs):
        assert self.is_compiled, "Model must in compile mode"
        inputs = self.get_leap_input_types()
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        mlir_path = str(Path(output_model_path).with_suffix(".mlir.bc"))
        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        # 1. export module
        bc_module = self.export_module(inputs, "vision", bc_path)
        # 2. mlir conversion
        mlir_module = self.convert_mlir(
            bc_module,
            mlir_path,
            dynamic_quant=True,
            enable_spu=False,
            enable_vpu=True,
            march=kwargs["march"],
        )
        func = mlir_module.functions[0]
        func.remove_io_op(["Dequantize", "Quantize"])
        graph = mlir_module.graphs[0]
        self.rename_graph_io(graph)
        hbo_list = []
        kwargs["core_num"] = core_num
        kwargs["enable_hpc"] = True
        if core_num > 1:
            kwargs["max_l2m_size"] = 25165824
        self.logger.info(f"compile config:\n{kwargs}")
        hbo = self.compile_hbo(mlir_module, hbo_path, **kwargs)
        hbo_list.append(hbo)
        # link model.hbo to model.hbm
        return self.link_models(hbo_list, output_model_path)


class Qwen3VLTextModel(Model):
    def __init__(self, config, logger: logging.Logger = None):
        super().__init__()
        self.config = config
        self.logger = logger
        # self.padding_idx = config.pad_token_id # not needed,
        # it is for embedding padding
        self.vocab_size = config.vocab_size
        self.embed_tokens = FakeQuantEmbedding(self.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(config, logger) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # move lm_heah from GenerateClass to TextModel
        self.lm_head = DynamicQuantLinear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config, device=self.embed_tokens.weight.device)

    def get_input_embeddings(self):
        return self.embed_tokens

    def _deepstack_masked_scatter_add_leap(self, hidden_states, visual_mask, visual_embeds):
        raise NotImplementedError("masked_scatter_add not implemented, " "only if add is I/O critical")

    def _deepstack_add_leap(self, hidden_states: np.ndarray, visual_embeds: np.ndarray):
        # assert hidden_states.ndim == visual_embeds.ndim,
        # dimension shall be consistant"
        hidden_states = leap.add(hidden_states, visual_embeds)
        return hidden_states

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_embeds: torch.Tensor):
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states + visual_embeds
        return hidden_states

    def forward(
        self,
        input_embeds,
        position_ids,
        attention_mask,
        cache_keys,
        cache_values,
        deepstack_visual_embeds,
    ):
        hidden_states = input_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        new_keys, new_values = [], []

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings,
                cache_keys[layer_idx],
                cache_values[layer_idx],
            )
            new_keys.append(new_key)
            new_values.append(new_value)
            # only first 3 layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(hidden_states, deepstack_visual_embeds[layer_idx])

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, *new_keys, *new_values

    def build(
        self,
        input_embeds: np.ndarray,
        position_ids: np.ndarray,
        attention_mask: np.ndarray,
        *input_list: List[np.ndarray],
        # *past_keys: List[np.ndarray],
        # *past_values: List[np.ndarray],
        # *visual_embeds: List[np.ndarray],
    ):
        """leap forward function for Qwen3VLTextModel

        Args:
            input_embeds    (float16)   [bs, seq_len, hidden_size]:
                input embeddings with text embeddings and ViT results embedded
            position_ids    (int32)     [bs, seq_len]:
                multi-modal position ids
            attention_mask  (float16)   [bs, seq_len, ctx_len]:
                causal attention mask
            past_keys       (float32)   [bs, ctx_len, #kv_head, head_dim]:
                unquanted key cache
            past_values     (float32)   [bs, ctx_len, #kv_head, head_dim]:
                unquanted value cache
            visual_embeds   (float16)   [bs, vis_len, hidden_size]:
                deepstack visual embeddings from selected ViT layers
            NOTE: switching deepstack embeddings to the last to comply with runtime
                    port regulation
        Returns:
            text logits
        """
        hidden_states = input_embeds
        # TODO: in decode stage, visual_embeds shall be None
        num_inputs = len(input_list)
        if num_inputs > self.config.num_hidden_layers * 2:
            num_deepstack_embeds = 3
            # visual_embeds = input_list[:num_deepstack_embeds]
            visual_embeds = input_list[-num_deepstack_embeds:]
            self.logger.info(f"#deepstack_embed = {len(visual_embeds)}")
        else:
            num_deepstack_embeds = 0
            visual_embeds = None
            self.logger.info("decode stage has no deepstack embeddings")

        kv_cache_start_idx = 0

        past_keys = input_list[kv_cache_start_idx : kv_cache_start_idx + self.config.num_hidden_layers]

        past_values = input_list[self.config.num_hidden_layers : 2 * self.config.num_hidden_layers]

        self.logger.info(f"#key_cache = {len(past_keys)}")
        self.logger.info(f"#value_cache = {len(past_values)}")

        position_embeddings = self.rotary_emb(position_ids)

        new_keys, new_values = [], []

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings,
                past_keys[layer_idx],
                past_values[layer_idx],
            )
            new_keys.append(new_key)
            new_values.append(new_value)

            # self.logger.info(f"hidden_states.shape={hidden_states.type.shape}")
            # self.logger.info(f"new_key.shape={new_key.type.shape}")
            # self.logger.info(f"new_value.shape={new_value.type.shape}")

            # deepstack embeddings are only applied to the first 3 layers
            if visual_embeds is not None and layer_idx in range(len(visual_embeds)):
                self.logger.info(f"deepstack embed.shape = {visual_embeds[0].type.shape}")
                hidden_states = self._deepstack_add_leap(hidden_states, visual_embeds[layer_idx])

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, *new_keys, *new_values

    def get_leap_input_types(self, seq_len, ctx_len: int = 4096, num_deepstack_embeds: int = 3):
        # input_embeds, position_ids, attention_mask
        input_types = [
            leap.TensorType([self.config.bs, seq_len, self.config.hidden_size], leap.float16),
            leap.TensorType([3, self.config.bs, seq_len], leap.int32),
            leap.TensorType([self.config.bs, seq_len, ctx_len], leap.float16),
        ]
        # past_keys, past_values
        for _ in range(self.config.num_hidden_layers * 2):
            input_types.append(
                leap.TensorType(
                    [
                        self.config.bs,
                        ctx_len,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ],
                    leap.float32,
                )
            )
        if seq_len != 1:
            # visual_embeds list
            for _ in range(num_deepstack_embeds):
                input_types.append(leap.TensorType([self.config.bs, seq_len, self.config.hidden_size], leap.float16))
        return input_types

    def rename_graph_io(self, graph, stage_name: str, num_deepstack_embeds: int = 3):
        """
        Rename graph flatten inputs and outputs for LLM KV-cache style models.
        """
        # -------- inputs --------
        input_names = [
            "input_embeds",
            "position_ids",
            "attention_mask",
        ]
        for i in range(self.config.num_hidden_layers):
            input_names.append(f"layer_{i}_cache_key")
        for i in range(self.config.num_hidden_layers):
            input_names.append(f"layer_{i}_cache_value")
        if stage_name == "prefill":
            for i in range(num_deepstack_embeds):
                input_names.append(f"deepstack_embed_{i}")

        assert len(graph.flatten_inputs) >= len(input_names), "flatten_inputs size mismatch"

        for tensor, name in zip(graph.flatten_inputs, input_names):
            tensor.name = name

        # -------- outputs --------
        output_names = ["logits"]

        for i in range(self.config.num_hidden_layers):
            output_names.append(f"layer_{i}_new_key")
        for i in range(self.config.num_hidden_layers):
            output_names.append(f"layer_{i}_new_value")

        assert len(graph.flatten_outputs) >= len(output_names), "flatten_outputs size mismatch"

        for tensor, name in zip(graph.flatten_outputs, output_names):
            tensor.name = name

    def compile(
        self,
        output_model_path: str,
        prefill_core_num: int,
        decode_core_num: int,
        **kwargs,
    ):
        assert self.is_compiled, "Model must in compile mode"
        hbo_list = []

        # compile several stages for qwen3VLTextModel
        for stage_name in ["prefill", "decode"]:
            if stage_name == "prefill":
                self.logger.info(
                    f"prefill stage: chunk_size={self.config.chunk_size} " f", ctx_len={self.config.cache_len}"
                )
                inputs = self.get_leap_input_types(
                    seq_len=self.config.chunk_size,
                    ctx_len=self.config.cache_len,
                )
                compile_cfg = kwargs.copy()
                compile_cfg["core_num"] = prefill_core_num
            elif stage_name == "decode":
                self.logger.info(f"decode stage: chunk_size=1 " f", ctx_len={self.config.cache_len}")
                inputs = self.get_leap_input_types(
                    seq_len=1,
                    ctx_len=self.config.cache_len,
                )
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
            graph = mlir_module.graphs[0]
            self.rename_graph_io(graph, stage_name=stage_name)
            # enable hpc by default
            compile_cfg["enable_hpc"] = True
            if compile_cfg["core_num"] > 1:
                compile_cfg["max_l2m_size"] = 25165824
            self.logger.info(f"compile config:\n{compile_cfg}")

            hbo = self.compile_hbo(mlir_module, hbo_path, **compile_cfg)
            hbo_list.append(hbo)

        return self.link_models(hbo_list, output_model_path)


class Qwen3VLModel(Model):
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config, logger=logger)
        self.language_model = Qwen3VLTextModel(config.text_config, logger=logger)
        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        raise NotImplementedError("This method is not supported.")

    def get_decoder(self):
        return self.language_model

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)


class Qwen3VLForConditionalGeneration(Model):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, model_config, logger=None):
        super().__init__()
        self.model = Qwen3VLModel(model_config, logger=logger)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_decoder(self):
        return self.model.get_decoder()

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual


class Qwen3VL_Wrapper:
    def __init__(self, model, model_args):
        self.model: Qwen3VLForConditionalGeneration = model
        self.model_args = model_args

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
        logger=None,
    ):
        def _get_config(model_type: str):
            module_name = model_type.replace("-", "_") + "_instruct"
            try:
                module = importlib.import_module(f"leap_llm.models.qwen3_vl.config.{module_name}")
                return (
                    module.Qwen3VLConfig,
                    module.Qwen3VLTextConfig,
                    module.Qwen3VLVisionConfig,
                )
            except ImportError:
                raise NotImplementedError(f"Model type {model_type} not supported") from None

        model, model_args = None, None
        Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig = _get_config(model_type)

        model_args = Qwen3VLConfig(vision_config=Qwen3VLVisionConfig(), text_config=Qwen3VLTextConfig())

        print(type(model_args))

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, encoding="utf-8") as f:
            usr_config = json.load(f)

        if usr_config["tie_word_embeddings"] is not None and usr_config["tie_word_embeddings"] is False:
            model_args.tie_word_embeddings = False
            warnings.warn("config.json with tie_word_embeddings = False.", stacklevel=2)

        # huggingface input format, safetensors
        new_state_dict = load_safetensors_state_dict(model_dir)

        # if contains scaled info, keys' names have been transformed
        has_scale = any(".scales" in k for k in new_state_dict)
        if has_scale:
            warnings.warn(f"Current checkpoint contains quantization info, w_bits = {w_bits} bits", stacklevel=2)

        new_state_dict = {f"model.{k}": v for k, v in new_state_dict.items()}

        # the model weight shard start as visual.xxx and language_model.xxx
        # make prefix as `model.lm_head.weight`
        # NOTE: llmc format may over-write the lm_head weight.
        if model_args.tie_word_embeddings:
            new_state_dict["model.language_model.lm_head.weight"] = new_state_dict[
                "model.language_model.embed_tokens.weight"
            ]
        else:
            # move model.lm_head.weight to model.language_model.lm_head.weight
            new_state_dict["model.language_model.lm_head.weight"] = new_state_dict["model.lm_head.weight"]
            del new_state_dict["model.lm_head.weight"]

        # dump the embedding for the runtime
        if model_type == "qwen3-vl-2b":
            embed_token_filename = "Qwen3-VL-2B_Instruct_embed_tokens_fp16.bin"
        elif model_type == "qwen3-vl-4b":
            embed_token_filename = "Qwen3-VL-4B_Instruct_embed_tokens_fp16.bin"
        elif model_type == "qwen3-vl-8b":
            embed_token_filename = "Qwen3-VL-8B_Instruct_embed_tokens_fp16.bin"
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        embed_token_path = os.path.join(output_model_dir, embed_token_filename)
        embed_weight = new_state_dict["model.language_model.embed_tokens.weight"]
        embed_data = embed_weight.data.to(torch.float16).detach().cpu().numpy()
        embed_data.tofile(embed_token_path)

        # convert patch_embed Conv3d weight to Linear
        conv3d_wt = new_state_dict["model.visual.patch_embed.proj.weight"]
        # print(conv3d_wt.shape) # torch.Size([1024, 3, 2, 16, 16])
        conv3d_wt = conv3d_wt.reshape(model_args.vision_config.hidden_size, -1).contiguous()
        # print(conv3d_wt.shape)
        new_state_dict["model.visual.patch_embed.proj.weight"] = conv3d_wt

        model_args.text_config.has_scale = has_scale
        model_args.text_config.w_bits = w_bits
        model_args.text_config.mask_value = mask_value
        model_args.text_config.chunk_size = chunk_size
        model_args.text_config.cache_len = cache_len

        model = Qwen3VLForConditionalGeneration(model_args, logger=logger)
        model.load_state_dict(new_state_dict, strict=True)

        return Qwen3VL_Wrapper(model, model_args)

    def get_image_feature(self, pixel_values, image_thw_grid):
        visual = self.get_visual_model()
        return visual(pixel_values)

    def get_model(self):
        return self.model

    def get_model_args(self):
        return self.model_args

    def get_visual_model(self):
        return self.model.visual

    def get_lang_model(self):
        return self.model.language_model

    def compile():
        pass
