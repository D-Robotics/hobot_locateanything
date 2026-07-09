from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hbdk4.compiler import leap
from safetensors.torch import load_file as safe_load_file

from leap_llm.models.Spirit_v1_5.blocks.text_block import Qwen3VLTextDecoderLayer
from leap_llm.models.Spirit_v1_5.blocks.text_rotary_emb import Qwen3VLTextRotaryEmbedding
from leap_llm.models.Spirit_v1_5.blocks.vision_block import Qwen3VLVisionBlock
from leap_llm.models.Spirit_v1_5.blocks.vision_patch import Qwen3VLVisionPatchEmbed, Qwen3VLVisionPatchMerger
from leap_llm.models.Spirit_v1_5.utils.visual_utils import prepare_patch_pos_emb, vision_rotary_pos_emb
from leap_llm.nn.modules.embedding import Embedding
from leap_llm.nn.modules.linear import DynamicQuantLinear
from leap_llm.nn.modules.rms_norm import RMSNorm
from leap_llm.nn.utils import Model, timeit

LANGUAGE_PREFIX = "qwen.model.language_model."
VISION_PREFIX = "qwen.model.visual."
from transformers import AutoConfig


class Qwen3VLVisionModel(Model):
    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.patch_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = Embedding(config.num_position_embeddings, config.hidden_size)
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
        # Spirit V1.5: 3 cameras × 320 patches each = 960 total patches
        # Each camera: grid [1, 16, 20], per-image coordinate like HF
        self.num_images = 3
        self.grid_thw_per_image = torch.tensor([[1, 16, 20]], device=self.merger.norm.weight.device)
        self.grid_thw = self.grid_thw_per_image.repeat(self.num_images, 1)

        self._init_grid_pos_emb()
        self._init_rot_pos_emb()

    def _init_grid_pos_emb(self):
        (self.grid_h, self.grid_w, self.grid_idx, self.grid_wt) = prepare_patch_pos_emb(
            grid_thw=self.grid_thw_per_image,
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
        # make grid indices broadcastable and continguous for numpy conversion
        gathered_feat = self.pos_embed(self.grid_idx.unsqueeze(-1).contiguous())
        pos_embeds = leap.mul(gathered_feat, self.grid_wt.contiguous())
        # (1, 784, 1024)
        patch_pos_embeds = leap.reduce_sum(pos_embeds, dims=[0])  # keepdim = True
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
        ).to(hidden_states.dtype)
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        merge = self.spatial_merge_size
        patch_pos_embeds = patch_pos_embeds.reshape(
            self.grid_h // merge, merge, self.grid_w // merge, merge, -1
        )
        patch_pos_embeds = patch_pos_embeds.permute(0, 2, 1, 3, 4).flatten(0, 3)
        patch_pos_embeds = patch_pos_embeds.repeat(self.num_images, 1)
        hidden_states = hidden_states + patch_pos_embeds

        self.position_embeddings = tuple(e.to(hidden_states.device, hidden_states.dtype) for e in self.position_embeddings)

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
        patch_pos_embeds_list = []
        for i in range(self.num_images):
            patch_pos_embeds_list.append(patch_pos_embeds)
        patch_pos_embeds = leap.concat(patch_pos_embeds_list, dim=0)

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

class Qwen3VLTextModel(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.padding_idx = config.pad_token_id # not needed,
        # it is for embedding padding
        self.vocab_size = config.vocab_size
        self.embed_tokens = Embedding(self.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(config) for _ in range(config.num_hidden_layers)])
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

    def _deepstack_process(
        self, hidden_states: torch.Tensor, packed_visual_embeds: torch.Tensor
    ):
        hidden_states = hidden_states + packed_visual_embeds
        return hidden_states

    def forward(
        self,
        input_embeds,
        position_ids,
        attention_mask,
        packed_visual_embeds,
    ):
        hidden_states = input_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings,
            )
            # only first 3 layers
            if packed_visual_embeds is not None and layer_idx in range(len(packed_visual_embeds)):
                hidden_states = self._deepstack_process(hidden_states, packed_visual_embeds[layer_idx])

        # hidden_states = self.norm(hidden_states)
        # logits = self.lm_head(hidden_states)
        return hidden_states

    def build(
        self,
        input_embeds,
        position_ids,
        attention_mask,
        *packed_visual_embeds,
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

        position_embeddings = self.rotary_emb(position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings,
            )
            if packed_visual_embeds is not None and layer_idx in range(len(packed_visual_embeds)):
                hidden_states = self._deepstack_add_leap(hidden_states, packed_visual_embeds[layer_idx])

        # hidden_states = self.norm(hidden_states)
        # logits = self.lm_head(hidden_states)
        return hidden_states


class SpiritLLMModel:
    @staticmethod
    @timeit
    def build(config_path: str, model_dir: str) -> "SpiritLLMModel":
        def extract_text_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            text_state_dict: dict[str, torch.Tensor] = {}

            for key, value in state_dict.items():
                if key.startswith(LANGUAGE_PREFIX):
                    text_state_dict[key[len(LANGUAGE_PREFIX) :]] = value

            # Spirit checkpoint stores lm_head outside language_model.
            if "qwen.lm_head.weight" in state_dict:
                text_state_dict["lm_head.weight"] = state_dict["qwen.lm_head.weight"]
            elif "qwen.model.language_model.lm_head.weight" in state_dict:
                text_state_dict["lm_head.weight"] = state_dict["qwen.model.language_model.lm_head.weight"]
            else:
                raise KeyError("Cannot find lm_head weight in Spirit checkpoint")

            return text_state_dict
        config = AutoConfig.from_pretrained(str(config_path))
        config.text_config.w_bits = 8
        config.text_config.has_scale = False
        model = Qwen3VLTextModel(config.text_config)
        state_dict = safe_load_file(str(model_dir))
        text_state_dict = extract_text_state_dict(state_dict)

        model.load_state_dict(text_state_dict, strict=True)
        return SpiritLLMModel(model, config)

    def __init__(self, model: Qwen3VLTextModel, model_args: AutoConfig):
        self.model = model
        self.model_args = model_args


    def get_leap_input_types(
        self, seqlen, images_num
    ) -> list[leap.TensorType]:
        input_types = [
            leap.TensorType([1, seqlen, 2560], leap.float16),
            leap.TensorType([images_num, 1, seqlen], leap.int32),
            leap.TensorType([1, 1, seqlen, seqlen], leap.float16),
        ]
        for _ in range(images_num):
            input_types.append(leap.TensorType([1, seqlen, 2560], leap.float16))
        return input_types

    def compile(
        self,
        output_model_path: str,
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        inputs = self.get_leap_input_types(320, 3)
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "spirit_llm", bc_path)
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



class SpiritVisionModel:
    @staticmethod
    @timeit
    def build(config_path: str, model_dir: str) -> "SpiritVisionModel":
        def extract_vision_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            vision_state_dict: dict[str, torch.Tensor] = {}

            for key, value in state_dict.items():
                if key.startswith(VISION_PREFIX):
                    vision_state_dict[key[len(VISION_PREFIX) :]] = value
            conv3d_wt = vision_state_dict["patch_embed.proj.weight"]
            # print(conv3d_wt.shape) # torch.Size([1024, 3, 2, 16, 16])
            conv3d_wt = conv3d_wt.reshape(config.vision_config.hidden_size, -1).contiguous()
            # print(conv3d_wt.shape)
            vision_state_dict["patch_embed.proj.weight"] = conv3d_wt

            return vision_state_dict
        config = AutoConfig.from_pretrained(str(config_path))
        config.vision_config.w_bits = 8
        config.vision_config.has_scale = False
        config.vision_config.head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        model = Qwen3VLVisionModel(config.vision_config)

        state_dict = safe_load_file(str(model_dir))
        vision_state_dict = extract_vision_state_dict(state_dict)

        model.load_state_dict(vision_state_dict, strict=True)
        return SpiritVisionModel(model, config)

    def __init__(self, model: Qwen3VLVisionModel, model_args: AutoConfig):
        self.model = model
        self.model_args = model_args

    def get_leap_input_types(
        self, vision_tokens_num
    ) -> list[leap.TensorType]:
        input_types = [
            leap.TensorType([1, vision_tokens_num, 1536], leap.float16),
        ]
        return input_types

    def compile(
        self,
        output_model_path: str,
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        inputs = self.get_leap_input_types(960)
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "spirit_vision", bc_path)
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
