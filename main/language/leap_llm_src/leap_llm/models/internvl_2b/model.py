import json
import os
from dataclasses import asdict, dataclass, fields
from inspect import signature
from pathlib import Path
from typing import List

import torch
from hbdk4.compiler import leap, save
from torch import nn

from leap_llm.models.internvl_2b.blocks import DecoderLayer, EncoderLayer
from leap_llm.models.internvl_2b.projector import Projector
from leap_llm.nn.modules import (  # noqa: E402
    DynamicQuantLinear,
    Embedding,
    RMSNorm,
    VisionEmbeddings,
)
from leap_llm.nn.utils import Model, load_safetensors_state_dict, timeit  # noqa: E402


@dataclass
class LlmConfig:
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_attention_heads: int = 12
    num_hidden_layers: int = 28
    num_key_value_heads: int = 2
    vocab_size: int = 92553
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000
    max_position_embeddings: int = 32768
    max_batch_size: int = 32
    head_dim: int = int(hidden_size / num_attention_heads)
    prefill_seq_len: int = 256
    decode_seq_len: int = 1
    input_embedding: bool = False
    w_bits: int = 8
    has_scale: bool = False


@dataclass
class VisionConfig:
    hidden_size: int = 1024
    image_size: int = 448
    initializer_factor: float = 1.0
    initializer_range: float = 0.02
    intermediate_size: int = 4096
    layer_norm_eps: float = 1e-06
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 24
    patch_size: int = 14


class LLM(Model):
    def __init__(self, params: LlmConfig, cache_len: int):
        super().__init__()

        self.tok_embeddings = Embedding(params.vocab_size, params.hidden_size)
        self.layers = torch.nn.ModuleList()

        DecoderLayerConfig = {
            k: v
            for k, v in asdict(params).items()
            if k in signature(DecoderLayer.__init__).parameters
        }
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(DecoderLayer(layer_id=layer_id, **DecoderLayerConfig))

        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        self.output = DynamicQuantLinear(
            params.hidden_size, params.vocab_size, bias=False
        )
        self.params = params
        self.cache_len = cache_len
        cos, sin = self._set_cos_sin_cache(
            params.max_position_embeddings,
            params.head_dim,
            base=params.rope_theta,
        )

        self.cos = cos[:cache_len, :]
        self.sin = sin[:cache_len, :]
        self.ppl_mode = os.getenv("INTERNVL_PPL_ENV", "").lower() == "true"
        if self.ppl_mode:
            print("ppl mode set true")
        else:
            print("ppl mode set false")

    def _set_cos_sin_cache(self, max_seq_len_cached, head_dim, base=1000000.0):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        return cos_cached, sin_cached

    def build(self, inputs, position_ids, mask, *caches):
        _bsz = 1
        seqlen = 1
        if self.params.input_embedding:
            hidden_states = inputs
            seqlen = inputs.type.shape[0]
        else:
            _bsz, seqlen = inputs.type.shape
            tokens = leap.reshape(inputs, [seqlen, _bsz])
            hidden_states = self.tok_embeddings(tokens)
        new_keys = []
        new_values = []

        caches_k = caches[: len(caches) // 2]
        caches_v = caches[len(caches) // 2 :]
        if seqlen > 1:
            vaild_lens = leap.reduce_argmax(position_ids, dims=[0], keepDim=True)
        position_ids = leap.reshape(position_ids, [seqlen, _bsz])
        cos = leap.gather_nd(self.cos, position_ids, 0)
        sin = leap.gather_nd(self.sin, position_ids, 0)
        cos = leap.cast_type(cos, output_type=hidden_states.type.element_type)
        sin = leap.cast_type(sin, output_type=hidden_states.type.element_type)

        for layer, cache_k, cache_v in zip(self.layers, caches_k, caches_v):
            cache_k = leap.transpose(cache_k, [1, 0, 2])
            cache_v = leap.transpose(cache_v, [1, 0, 2])

            hidden_states, new_k, new_v = layer(
                hidden_states, cos, sin, cache_k, cache_v, mask
            )

            new_k = leap.transpose(new_k, [1, 0, 2])
            new_v = leap.transpose(new_v, [1, 0, 2])

            new_keys.append(new_k)
            new_values.append(new_v)

        hidden_states = self.norm(hidden_states)
        hidden_states = leap.reshape(
            hidden_states, [1, seqlen, self.params.hidden_size]
        )
        if not self.ppl_mode and seqlen > 1:
            hidden_states = leap.index(hidden_states, vaild_lens, dim=1)
        logits = self.output(hidden_states)

        print("outputs shape ", logits.type)
        return logits, *new_keys, *new_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        caches: List[torch.Tensor],
    ):
        new_keys = []
        new_values = []

        caches_k = caches[: len(caches) // 2]
        caches_v = caches[len(caches) // 2 :]

        mask_type = mask

        cos = self.cos.to(position_ids.device)[position_ids]
        sin = self.sin.to(position_ids.device)[position_ids]

        for layer, cache_k, cache_v in zip(self.layers, caches_k, caches_v):
            cache_k_type = cache_k.transpose(1, 0)
            cache_v_type = cache_v.transpose(1, 0)

            hidden_states, new_k, new_v = layer(
                hidden_states, cos, sin, cache_k_type, cache_v_type, mask_type
            )

            new_k = new_k.transpose(1, 0)
            new_v = new_v.transpose(1, 0)

            new_keys.append(new_k)
            new_values.append(new_v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)
        return logits, *new_keys, *new_values

    def get_sin_cos(self, position_ids: torch.Tensor):
        cos = self.cos.to(position_ids.device)[position_ids]
        sin = self.sin.to(position_ids.device)[position_ids]
        return sin, cos

    def get_embedding(self, token_ids: torch.Tensor):
        return self.tok_embeddings(token_ids)


class VIT(Model):
    def __init__(
        self, params: VisionConfig, llm_hidden_size: int, downsample_ratio: float
    ):
        super().__init__()

        self.embeddings = VisionEmbeddings(
            params.hidden_size,
            params.num_channels,
            params.patch_size,
            params.image_size,
        )
        self.layers = nn.ModuleList()

        EncoderLayerConfig = {
            k: v
            for k, v in asdict(params).items()
            if k in signature(EncoderLayer.__init__).parameters
        }
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(EncoderLayer(layer_id=layer_id, **EncoderLayerConfig))

        self.mlp1 = Projector(params.hidden_size, llm_hidden_size, downsample_ratio)
        self.downsample_ratio = downsample_ratio

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.type.shape
        x = leap.reshape(x, [n, w, int(h * scale_factor), int(c / scale_factor)])
        x = leap.transpose(x, [0, 2, 1, 3])
        x = leap.reshape(
            x,
            [
                n,
                int(h * scale_factor),
                int(w * scale_factor),
                int(c / (scale_factor * scale_factor)),
            ],
        )
        x = leap.transpose(x, [0, 2, 1, 3])
        return x

    def build(self, img_pixel):
        hidden_states = self.embeddings(img_pixel)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        batch, n, c = hidden_states.type.shape
        vit_embeds = leap.slice(
            hidden_states, [0, 1, 0], [batch + 1, n + 1, c + 1], [1, 1, 1]
        )
        print("vit_embs shape: ", vit_embeds.type)
        h = w = int(n**0.5)
        vit_embeds = leap.reshape(vit_embeds, [batch, h, w, c])
        vit_embeds = self.pixel_shuffle(vit_embeds, self.downsample_ratio)
        batch, h, w, c = vit_embeds.type.shape
        vit_embeds = leap.reshape(vit_embeds, [batch, h * w, c])
        vit_embeds = self.mlp1(vit_embeds)
        print("vit projector shape: ", vit_embeds.type)
        return vit_embeds

    def pixel_shuffle_torch(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(
        self,
        img_pixel: torch.Tensor,
    ):
        hidden_states = self.embeddings(img_pixel)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        vit_embeds = hidden_states[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle_torch(
            vit_embeds, scale_factor=self.downsample_ratio
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        print("vit_embeds ", vit_embeds.shape)
        return vit_embeds


class Internlm2b:
    @staticmethod
    def split_wqkv(wqkv: torch.Tensor, heads: int, kv_heads: int):
        size, hidden_dim = wqkv.shape
        all_heads_num = heads + 2 * kv_heads
        head_dim = hidden_dim // heads
        assert head_dim * all_heads_num == size, "wqkv output size invalid"
        wqkv = wqkv.view(all_heads_num, head_dim, hidden_dim)
        groups = wqkv.chunk(kv_heads, dim=0)

        # 提取出q, k, v
        q_tensors = []
        k_tensors = []
        v_tensors = []
        wq_in_per_group: int = heads // kv_heads
        for group in groups:
            q_tensors.append(group[:wq_in_per_group])
            k_tensors.append(group[wq_in_per_group : wq_in_per_group + 1])
            v_tensors.append(group[-1])
        # 将各个组内的q, k, v合并
        wq = torch.cat(q_tensors, dim=0)
        wk = torch.cat(k_tensors, dim=0)
        wv = torch.cat(v_tensors, dim=0)
        wq = wq.view(heads * head_dim, hidden_dim)
        wv = wv.view(kv_heads * head_dim, hidden_dim)
        wk = wk.view(kv_heads * head_dim, hidden_dim)
        return wq, wk, wv

    @staticmethod
    def split_wqkv_scales(wqkv_scales: torch.Tensor, heads: int, kv_heads: int):
        size, one_dim = wqkv_scales.shape
        assert one_dim == 1, "wqkv.scales second dim must be 1"

        all_heads_num = heads + 2 * kv_heads
        head_dim = size // all_heads_num
        assert (
            head_dim * all_heads_num == size
        ), "wqkv.scales size is not divisible by all_heads_num"

        # [all_heads_num * head_dim, 1] -> [all_heads_num, head_dim, 1]
        wqkv_scales = wqkv_scales.view(all_heads_num, head_dim, 1)
        groups = wqkv_scales.chunk(kv_heads, dim=0)

        q_tensors = []
        k_tensors = []
        v_tensors = []
        wq_in_per_group: int = heads // kv_heads
        for group in groups:
            q_tensors.append(group[:wq_in_per_group])
            k_tensors.append(group[wq_in_per_group : wq_in_per_group + 1])
            v_tensors.append(group[-1])

        wq_scales = torch.cat(q_tensors, dim=0)
        wk_scales = torch.cat(k_tensors, dim=0)
        wv_scales = torch.cat(v_tensors, dim=0)

        # [heads, head_dim, 1] -> [heads * head_dim, 1]
        wq_scales = wq_scales.view(heads * head_dim, 1)
        wk_scales = wk_scales.view(kv_heads * head_dim, 1)
        wv_scales = wv_scales.view(kv_heads * head_dim, 1)
        return wq_scales, wk_scales, wv_scales

    @staticmethod
    @timeit
    def load_model(
        input_model_path: str,
        checkpoint: {},
        chunk_size=512,
        cache_len=4096,
        w_bits=8,
        weight_scales_file=None,
        kept_tokens_file=None,
    ) -> "Internlm2b":
        config_path = os.path.join(input_model_path, "config.json")
        assert os.path.exists(
            config_path
        ), f"config.json not found in {input_model_path}"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_args_dict = {
            field.name: config["llm_config"].get(field.name, field.default)
            for field in fields(LlmConfig)
        }

        if kept_tokens_file is not None:
            print("This model is not support compress vocab now!")

        # TODO: 不支持 batch, 代码内部是没有额外添加及处理 batch 维度的
        model_args_dict["max_batch_size"] = 1
        model_args_dict["prefill_seq_len"] = chunk_size
        model_args_dict["input_embedding"] = True
        model_args_dict["w_bits"] = w_bits

        state_dict = load_safetensors_state_dict(
            input_model_path,
            include_substrings=["buf_scales"],
            content_change_map={".buf_scales": ".scales"},
            prefix_remove_list=[],
        )
        for key, v in state_dict.items():
            checkpoint[key] = v

        if weight_scales_file:
            has_scale = True
        else:
            has_scale = any(".scales" in k for k in state_dict.keys())
        model_args_dict["has_scale"] = has_scale
        print(f"has_scale flag : {has_scale}")

        model_args = LlmConfig(**model_args_dict)
        # model_args.num_hidden_layers = 1  # for temp test
        model = LLM(model_args, cache_len=cache_len)

        prefix = "language_model.model.layers."
        for layer_id in range(model_args.num_hidden_layers):
            name_wqkv = prefix + str(layer_id) + ".attention.wqkv.weight"
            wq, wk, wv = Internlm2b.split_wqkv(
                checkpoint[name_wqkv],
                model_args.num_attention_heads,
                model_args.num_key_value_heads,
            )
            checkpoint[prefix + str(layer_id) + ".attention.q_proj.weight"] = wq
            checkpoint[prefix + str(layer_id) + ".attention.k_proj.weight"] = wk
            checkpoint[prefix + str(layer_id) + ".attention.v_proj.weight"] = wv
            checkpoint.pop(name_wqkv, "can not pop wqkv element")

            # 处理量化 scale：将 wqkv.scales 拆成 q/k/v.scales
            name_wqkv_scales = prefix + str(layer_id) + ".attention.wqkv.scales"
            if name_wqkv_scales in checkpoint:
                (
                    wq_scales,
                    wk_scales,
                    wv_scales,
                ) = Internlm2b.split_wqkv_scales(
                    checkpoint[name_wqkv_scales],
                    model_args.num_attention_heads,
                    model_args.num_key_value_heads,
                )
                checkpoint[prefix + str(layer_id) + ".attention.q_proj.scales"] = (
                    wq_scales
                )
                checkpoint[prefix + str(layer_id) + ".attention.k_proj.scales"] = (
                    wk_scales
                )
                checkpoint[prefix + str(layer_id) + ".attention.v_proj.scales"] = (
                    wv_scales
                )
                checkpoint.pop(name_wqkv_scales, "can not pop wqkv.scales element")

        if weight_scales_file:
            weight_scales = torch.load(weight_scales_file)
            for key, value in weight_scales.items():
                key = key.replace(".weight_quantizer", "")
                checkpoint["language_model." + key] = value

        new_state_dict = {}
        for key, value in checkpoint.items():
            if not key.startswith("language_model."):
                continue
            key = key[len("language_model.") :]
            new_key = key
            if key.startswith("model."):
                new_key = key[len("model.") :]
            new_state_dict[new_key] = value
            # for temp test
            # if new_key.startswith("layers.0."):
            #    new_state_dict[new_key] = value
            # elif not new_key.startswith("layers."):
            #    new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)

        return Internlm2b(model, model_args)

    def __init__(self, model: LLM, model_args: LlmConfig):
        self.model = model
        self.model_args = model_args

    def get_leap_input_types(self, seq_len, dtype) -> List[leap.TensorType]:
        input_types = []
        if self.model_args.input_embedding:
            input_types.append(
                leap.TensorType([seq_len, self.model_args.hidden_size], dtype)
            )
        else:
            input_types.append(leap.TensorType([1, seq_len], leap.int64))
        # prepare position ids and mask inputs
        input_types.append(leap.TensorType([seq_len], leap.int32))
        input_types.append(leap.TensorType([seq_len, self.model.cache_len], dtype))
        # prepare cache k v inputs
        for _ in range(self.model_args.num_hidden_layers * 2):
            input_types.append(
                leap.TensorType(
                    [
                        self.model.cache_len,
                        self.model_args.num_key_value_heads,
                        self.model_args.head_dim,
                    ],
                    leap.float32,
                )
            )
        return input_types

    def compile(
        self,
        dtype,
        stage: str,
        output_model_path: str,
        prefill_core_num: list[int] = [1],
        decode_core_num: list[int] = [1],
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        stages = []
        if stage in {"prefill", "all"}:
            stages.append("prefill")
        if stage in {"decode", "all"}:
            stages.append("decode")
        hbos = []
        for stage_name in stages:
            if stage_name == "prefill":
                seq_len, core_num_list = (
                    self.model_args.prefill_seq_len,
                    prefill_core_num,
                )
            else:
                seq_len, core_num_list = self.model_args.decode_seq_len, decode_core_num

            inputs = self.get_leap_input_types(seq_len, dtype)
            for core_num in core_num_list:
                new_stage_name = stage_name + f"_core_{core_num}"

                bc_path = str(
                    Path(output_model_path).with_suffix(f".{new_stage_name}.bc")
                )
                print(f"bc_path : {bc_path}")
                bc_module = self.model.export_module(inputs, new_stage_name, bc_path)

                convert_bc_path = str(
                    Path(output_model_path).with_suffix(f".{new_stage_name}_convert.bc")
                )
                print(f"bc_convert_path : {convert_bc_path}")
                mlir_module = self.model.convert_mlir(
                    bc_module,
                    save_path=convert_bc_path,
                    march=kwargs["march"],
                    dynamic_quant=True,
                )
                # return
                print("convert_mlir done")

                # input delete: "Quantize" ; output delete: "Dequantize"
                inputs_len = len(mlir_module[0].flatten_inputs)
                outputs_len = len(mlir_module[0].flatten_outputs)
                print(f"inputs:{inputs_len} , outputs:{outputs_len}")
                for i in range(3, inputs_len):
                    print(f"delete flatten_inputs[{i}] Quantize]")
                    mlir_module[0].flatten_inputs[i].remove_attached_op()
                print("input del quantize done")

                for i in range(1, outputs_len):
                    print(f"delete flatten_outputs[{i}] Dequantize]")
                    mlir_module[0].flatten_outputs[i].remove_attached_op()
                print("output del dequantize done")

                convert_removed_bc_path = str(
                    Path(output_model_path).with_suffix(
                        f".{new_stage_name}_convert_rm.bc"
                    )
                )
                print(f"convert_removed_bc_path : {convert_removed_bc_path}")
                save(mlir_module, convert_removed_bc_path)
                print("remove done")

                hbo_path = str(
                    Path(output_model_path).with_suffix(f".{new_stage_name}.hbo")
                )
                print(f"hbo_path : {hbo_path}")

                kwargs["core_num"] = core_num
                if kwargs["core_num"] > 1:
                    kwargs["max_l2m_size"] = 25165824
                    print(
                        f"{new_stage_name}, core_num: {core_num}, set max_l2m_size"
                    )
                else:
                    kwargs.pop("max_l2m_size", None)
                    print(
                        f"{new_stage_name}, core_num: {core_num}, del max_l2m_size"
                    )
                print(f"kwargs : {kwargs}")
                hbo_model = self.model.compile_hbo(
                    mlir_module,
                    hbo_path,
                    **kwargs,
                )
                print("compile_hbo done")
                hbos.append(hbo_model)

        return self.model.link_models(hbos, output_model_path)

    def get_embedding(self, token_ids: torch.Tensor):
        return self.model.get_embedding(token_ids)

    def get_model_args(self):
        return self.model_args

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        caches: List[torch.Tensor],
    ):
        hidden_states = self.get_embedding(tokens)
        return self.model(hidden_states, position_ids, mask, caches)

    def set_compile_mode(self, mode: bool):
        self.model.compile_mode(mode)

    def set_model_device(self, device, dtype):
        self.model.to(device, dtype=dtype)


class Internvl2bVision:
    @staticmethod
    def split_wqkv(wqkv: torch.Tensor, heads: int, kv_heads: int):
        size, hidden_dim = wqkv.shape
        all_heads_num = heads + 2 * kv_heads
        head_dim = hidden_dim // heads
        assert head_dim * all_heads_num == size, "wqkv output size invalid"

        wqkv = wqkv.view(all_heads_num, head_dim, hidden_dim)
        sub_tensors = torch.split(tensor=wqkv, split_size_or_sections=16, dim=0)
        wq = sub_tensors[0]
        wk = sub_tensors[1]
        wv = sub_tensors[2]

        wq = wq.view(heads * head_dim, hidden_dim)
        wk = wk.view(kv_heads * head_dim, hidden_dim)
        wv = wv.view(kv_heads * head_dim, hidden_dim)
        return wq, wk, wv

    def split_bqkv(bqkv, heads: int, kv_heads: int):
        size = bqkv.shape[0]
        all_heads_num = heads + 2 * kv_heads
        head_dim = size // all_heads_num
        assert head_dim * all_heads_num == size, "bqkv output size invalid"
        bqkv = bqkv.view(all_heads_num, head_dim)

        sub_tensors = torch.split(tensor=bqkv, split_size_or_sections=16, dim=0)
        bq = sub_tensors[0]
        bk = sub_tensors[1]
        bv = sub_tensors[2]

        bq = bq.view(heads * head_dim)
        bk = bk.view(kv_heads * head_dim)
        bv = bv.view(kv_heads * head_dim)

        return bq, bk, bv

    @staticmethod
    @timeit
    def load_model(input_model_path: str, checkpoint: {}) -> "Internvl2bVision":
        config_path = os.path.join(input_model_path, "config.json")
        assert os.path.exists(
            config_path
        ), f"config.json not found in {input_model_path}"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_args_dict = {
            field.name: config["vision_config"].get(field.name, field.default)
            for field in fields(VisionConfig)
        }

        model_args = VisionConfig(**model_args_dict)
        # model_args.num_hidden_layers = 1  # for temp test
        model = VIT(
            model_args, config["llm_config"]["hidden_size"], config["downsample_ratio"]
        )

        prefix = "vision_model.encoder.layers."
        for layer_id in range(model_args.num_hidden_layers):
            # split qkv.weight
            name_wqkv = prefix + str(layer_id) + ".attn.qkv.weight"
            wq, wk, wv = Internvl2bVision.split_wqkv(
                checkpoint[name_wqkv],
                model_args.num_attention_heads,
                model_args.num_attention_heads,
            )
            checkpoint[prefix + str(layer_id) + ".attn.wq.weight"] = wq
            checkpoint[prefix + str(layer_id) + ".attn.wk.weight"] = wk
            checkpoint[prefix + str(layer_id) + ".attn.wv.weight"] = wv
            checkpoint.pop(name_wqkv, "can not pop wqkv element")
            # split qkv.bias
            name_bqkv = prefix + str(layer_id) + ".attn.qkv.bias"
            bq, bk, bv = Internvl2bVision.split_bqkv(
                checkpoint[name_bqkv],
                model_args.num_attention_heads,
                model_args.num_attention_heads,
            )
            checkpoint[prefix + str(layer_id) + ".attn.wq.bias"] = bq
            checkpoint[prefix + str(layer_id) + ".attn.wk.bias"] = bk
            checkpoint[prefix + str(layer_id) + ".attn.wv.bias"] = bv
            checkpoint.pop(name_bqkv, "can not pop bqkv element")

        new_state_dict = {}
        for key, value in checkpoint.items():
            if not key.startswith("vision_model."):
                continue
            key = key[len("vision_model.") :]
            new_key = key
            if key.startswith("encoder."):
                new_key = key[len("encoder.") :]
            new_state_dict[new_key] = value
            # if new_key.startswith("layers.0."):
            #    new_state_dict[new_key] = value
            # elif not new_key.startswith("layers."):
            #    new_state_dict[new_key] = value

        # process projector params
        new_state_dict["mlp1.norm.weight"] = checkpoint["mlp1.0.weight"]
        new_state_dict["mlp1.norm.bias"] = checkpoint["mlp1.0.bias"]
        new_state_dict["mlp1.linear1.weight"] = checkpoint["mlp1.1.weight"]
        new_state_dict["mlp1.linear1.bias"] = checkpoint["mlp1.1.bias"]
        new_state_dict["mlp1.linear3.weight"] = checkpoint["mlp1.3.weight"]
        new_state_dict["mlp1.linear3.bias"] = checkpoint["mlp1.3.bias"]
        # for conv2d compile
        weight_name = "embeddings.patch_embedding.weight"
        new_state_dict[weight_name] = (
            new_state_dict[weight_name].permute(0, 2, 3, 1).contiguous()
        )
        bias_name = "embeddings.patch_embedding.bias"
        new_state_dict[bias_name] = new_state_dict[bias_name]

        model.load_state_dict(new_state_dict)

        return Internvl2bVision(model, model_args)

    def __init__(self, model: VIT, model_args: VisionConfig):
        self.model = model
        self.model_args = model_args

    def get_leap_inputs(self, dtype) -> List[leap.TensorType]:
        PIXEL_SHAPE = (
            1,
            self.model_args.num_channels,
            self.model_args.image_size,
            self.model_args.image_size,
        )
        inputs = [leap.TensorType(PIXEL_SHAPE, dtype)]

        return inputs

    def compile(
        self,
        dtype,
        output_model_path: str,
        vit_core_num: list[int] = [1],
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."
        inputs = self.get_leap_inputs(dtype)
        hbos = []
        for core_num in vit_core_num:
            new_stage_name = f"vit_core_{core_num}"
            bc_path = str(Path(output_model_path).with_suffix(f".{new_stage_name}.bc"))
            bc_module = self.model.export_module(inputs, new_stage_name, bc_path)
            print(f"bc_path : {bc_path}")

            covert_bc_path = str(
                Path(output_model_path).with_suffix(f".{new_stage_name}_convert.bc")
            )
            mlir_module = self.model.convert_mlir(
                bc_module,
                save_path=covert_bc_path,
                march=kwargs["march"],
                dynamic_quant=True,
            )

            kwargs["core_num"] = core_num
            if kwargs["core_num"] > 1:
                kwargs["max_l2m_size"] = 25165824
                print(
                    f"{new_stage_name}, core_num: {core_num}, set max_l2m_size"
                )
            else:
                kwargs.pop("max_l2m_size", None)
                print(
                    f"{new_stage_name}, core_num: {core_num}, delete max_l2m_size"
                )
            print(f"kwargs : {kwargs}")
            hbo_path = str(
                Path(output_model_path).with_suffix(f".{new_stage_name}.hbo")
            )
            hbo_model = self.model.compile_hbo(
                mlir_module,
                hbo_path,
                **kwargs,
            )
            hbos.append(hbo_model)

        return self.model.link_models(hbos, output_model_path)

    def forward(self, image_pixel: torch.Tensor):
        return self.model(image_pixel)

    def set_compile_mode(self, mode: bool):
        self.model.compile_mode(mode)

    def set_model_device(self, device, dtype):
        self.model.to(device, dtype=dtype)
