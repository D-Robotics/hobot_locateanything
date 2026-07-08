import json
import os
import shutil
from dataclasses import asdict, dataclass, fields
from inspect import signature
from pathlib import Path
from typing import List

import torch
from hbdk4.compiler import leap, save
from transformers import AutoModelForCausalLM

from leap_llm.models.internlm2.blocks import InternLM2DecoderLayer  # noqa: E402
from leap_llm.nn.modules import (
    ConstFakeQuant,
    FakeQuantEmbedding,  # noqa: E402
    FakeQuantLinear,
    FakeQuantRMSNorm,
)
from leap_llm.nn.utils import Model, timeit  # noqa: E402


@dataclass
class ModelArgs:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    num_key_value_heads: int = 8
    vocab_size: int = 92544
    rms_norm_eps: float = 1e-05
    rope_theta: float = 1000000
    max_position_embeddings: int = 32768
    max_batch_size: int = 1
    head_dim: int = int(hidden_size / num_attention_heads)
    prefill_seq_len: int = 256
    decode_seq_len: int = 1


class LLM(Model):
    def __init__(
        self, params: ModelArgs, cache_len: int, preserve_precision: bool = False
    ):
        super().__init__()

        self.tok_embeddings = FakeQuantEmbedding(params.vocab_size, params.hidden_size)
        self.layers = torch.nn.ModuleList()

        DecoderLayerConfig = {
            k: v
            for k, v in asdict(params).items()
            if k in signature(InternLM2DecoderLayer.__init__).parameters
        }
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(
                InternLM2DecoderLayer(
                    layer_id=layer_id,
                    preserve_precision=preserve_precision,
                    **DecoderLayerConfig,
                )
            )

        self.norm = FakeQuantRMSNorm(
            params.hidden_size,
            eps=params.rms_norm_eps,
            preserve_precision=preserve_precision,
        )
        self.output = FakeQuantLinear(params.hidden_size, params.vocab_size, bias=False)
        self.params = params
        self.cache_len = cache_len
        cos, sin = self._set_cos_sin_cache(
            params.max_position_embeddings,
            params.head_dim,
            base=params.rope_theta,
        )

        self.cos = cos[:cache_len, :]
        self.sin = sin[:cache_len, :]

        self.mask_fq = ConstFakeQuant(16)
        self.cos_fq = ConstFakeQuant(16)
        self.sin_fq = ConstFakeQuant(16)

    def _set_cos_sin_cache(self, max_seq_len_cached, head_dim, base=1000000.0):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(torch.float32)
        sin_cached = emb.sin().to(torch.float32)
        return cos_cached, sin_cached

    def build(self, tokens, position_ids, mask, *caches):
        _bsz, seqlen = tokens.type.shape
        tokens = leap.reshape(tokens, [seqlen, _bsz])
        hidden_states = self.tok_embeddings(tokens)

        new_keys = []
        new_values = []

        caches_k = caches[: len(caches) // 2]
        caches_v = caches[len(caches) // 2 :]

        position_ids = leap.reshape(position_ids, [seqlen, _bsz])
        cos = leap.gather_nd(self.cos, position_ids, 0)
        sin = leap.gather_nd(self.sin, position_ids, 0)

        cos = self.cos_fq(cos)
        sin = self.sin_fq(sin)
        mask = self.mask_fq(mask)

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
        logits = self.output(hidden_states)
        return logits, *new_keys, *new_values

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        caches: List[torch.Tensor],
    ):
        hidden_states = self.tok_embeddings(tokens)

        new_keys = []
        new_values = []

        caches_k = caches[: len(caches) // 2]
        caches_v = caches[len(caches) // 2 :]

        cos = self.cos.to(position_ids.device)[position_ids]
        sin = self.sin.to(position_ids.device)[position_ids]

        cos = self.cos_fq(cos)
        sin = self.sin_fq(sin)
        mask = self.mask_fq(mask)

        for layer, cache_k, cache_v in zip(self.layers, caches_k, caches_v):
            cache_k = cache_k.transpose(1, 0)
            cache_v = cache_v.transpose(1, 0)

            hidden_states, new_k, new_v = layer(
                hidden_states, cos, sin, cache_k, cache_v, mask
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


class Internlm2:
    @staticmethod
    def split_wqkv(wqkv: torch.Tensor, heads: int, kv_heads: int):
        size, hidden_dim = wqkv.shape
        all_heads_num = heads + 2 * kv_heads
        head_dim = hidden_dim // heads
        assert head_dim * all_heads_num == size, "wqkv output size invalid"
        wqkv = wqkv.view(all_heads_num, head_dim, hidden_dim)
        groups = wqkv.chunk(kv_heads, dim=0)

        q_tensors = []
        k_tensors = []
        v_tensors = []
        wq_in_per_group: int = heads // kv_heads
        for group in groups:
            q_tensors.append(group[:wq_in_per_group])
            k_tensors.append(group[wq_in_per_group : wq_in_per_group + 1])
            v_tensors.append(group[-1])

        wq = torch.cat(q_tensors, dim=0)
        wk = torch.cat(k_tensors, dim=0)
        wv = torch.cat(v_tensors, dim=0)
        wq = wq.view(heads * head_dim, hidden_dim)
        wv = wv.view(kv_heads * head_dim, hidden_dim)
        wk = wk.view(kv_heads * head_dim, hidden_dim)
        return wq, wk, wv

    @staticmethod
    @timeit
    def build(
        model_dir: str,
        chunk_size=256,
        cache_len=1024,
        preserve_precision=False,
    ) -> "Internlm2":
        assert os.path.isdir(
            model_dir
        ), f"Config directory '{model_dir}' does not exist."

        checkpoints = sorted(Path(model_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {model_dir}"
        ckpt_path = checkpoints[0]  # No parallel
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        config_path = os.path.join(model_dir, "config.json")
        assert os.path.exists(config_path), f"config.json not found in {model_dir}"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_args_dict = {
            field.name: config.get(field.name, field.default)
            for field in fields(ModelArgs)
        }

        model_args_dict["max_batch_size"] = 1
        model_args_dict["prefill_seq_len"] = chunk_size

        model_args = ModelArgs(**model_args_dict)
        # model_args.num_hidden_layers = 1  # for temp test
        model = LLM(
            model_args, cache_len=cache_len, preserve_precision=preserve_precision
        )

        prefix = "model.layers."
        for layer_id in range(model_args.num_hidden_layers):
            name_wqkv = prefix + str(layer_id) + ".attention.wqkv.weight"
            wq, wk, wv = Internlm2.split_wqkv(
                checkpoint[name_wqkv],
                model_args.num_attention_heads,
                model_args.num_key_value_heads,
            )
            checkpoint[prefix + str(layer_id) + ".attention.wq.weight"] = wq
            checkpoint[prefix + str(layer_id) + ".attention.wk.weight"] = wk
            checkpoint[prefix + str(layer_id) + ".attention.wv.weight"] = wv
            checkpoint.pop(name_wqkv, "can not pop wqkv element")

        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key
            if key.startswith("model."):
                new_key = key[len("model.") :]
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)

        return Internlm2(model, model_args)

    def __init__(self, model: LLM, model_args: ModelArgs):
        self.model = model
        self.model_args = model_args

    def get_leap_input_types(self, seq_len) -> List[leap.TensorType]:
        input_types = [
            leap.TensorType([1, seq_len], leap.int32),
            leap.TensorType([seq_len], leap.int32),
            leap.TensorType([seq_len, self.model.cache_len], leap.float32),
        ]
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
        stage: str,
        output_model_path: str,
        enable_vpu=True,
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        model_list = []
        stages = []
        if stage in {"prefill", "all"}:
            stages.append("prefill")
        if stage in {"decode", "all"}:
            stages.append("decode")

        for stage_name in stages:
            seq_len = (
                self.model_args.prefill_seq_len
                if stage_name == "prefill"
                else self.model_args.decode_seq_len
            )
            inputs = self.get_leap_input_types(seq_len)
            bc_path = str(Path(output_model_path).with_suffix(f".{stage_name}.bc"))
            bc_module = self.model.export_module(
                inputs, stage_name, bc_path, high_precision_qpp=False
            )
            model_list.append(bc_module)

        hbos = []
        for bc_module in model_list:
            func_name = bc_module.functions[0].name
            convert_bc_path = str(
                Path(output_model_path).with_suffix(f".{func_name}_convert.bc")
            )
            mlir_module = self.model.convert_mlir(
                bc_module,
                convert_bc_path,
                enable_vpu=enable_vpu,
                march=kwargs["march"],
            )

            func = mlir_module.functions[0]
            # "Transpose" "Reshape",
            func.remove_io_op(["Dequantize", "Quantize"])
            convert_removed_bc_path = str(
                Path(output_model_path).with_suffix(f".{func_name}_convert_removed.bc")
            )
            save(mlir_module, convert_removed_bc_path)
            print(f"kwargs : {kwargs}")
            hbo_path = str(Path(output_model_path).with_suffix(f".{func_name}.hbo"))
            hbo_model = self.model.compile_hbo(
                mlir_module,
                hbo_path,
                **kwargs,
            )
            hbos.append(hbo_model)

        return self.model.link_models(hbos, output_model_path)

    def get_model_args(self):
        return self.model_args

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        caches: List[torch.Tensor],
    ):
        return self.model(tokens, position_ids, mask, caches)

    def set_compile_mode(self, mode: bool):
        self.model.compile_mode(mode)

    def set_model_device(self, device, dtype):
        self.model.to(device, dtype=dtype)


def save_model_checkpoint(model_dir, output_model_path):
    dir_path = os.path.dirname(output_model_path)

    ckpt_dir = os.path.join(dir_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model_checkpoint.pth")

    if not os.path.exists(ckpt_path):
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Save checkpoint path: {ckpt_path}")

    config_json_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_json_path):
        shutil.copyfile(os.path.join(model_dir, "config.json"), config_json_path)

    return ckpt_dir
