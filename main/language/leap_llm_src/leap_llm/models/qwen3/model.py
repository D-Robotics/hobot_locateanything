import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List

import torch
from hbdk4.compiler import leap, save, statistics

from leap_llm.models.qwen3.blocks import DecoderLayer
from leap_llm.nn.modules import DynamicQuantLinear, FakeQuantEmbedding, RMSNorm
from leap_llm.nn.utils import Model, load_safetensors_state_dict, timeit

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Qwen3Config:
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_size: int = 2048
    head_dim: int = 128
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000
    attention_bias: bool = False
    max_position_embeddings: int = 40960
    batch_size: int = 1
    prefill_seq_len: int = 256
    decode_seq_len: int = 1
    context_len: int = 4096
    w_bits: int = 8
    has_scale: bool = False
    enable_eagle3: bool = False
    num_draft_tokens: int = 32


class Qwen3Model(Model):
    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = FakeQuantEmbedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(
                DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                )
            )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.lm_head = DynamicQuantLinear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.context_len = config.context_len

        cos, sin = self._set_cos_sin_cache(
            config.max_position_embeddings,
            config.head_dim,
            base=config.rope_theta,
        )
        self.cos = cos[:, : self.context_len, :]
        self.sin = sin[:, : self.context_len, :]

    def get_input_embeddings(self):
        return self.embed_tokens

    def _set_cos_sin_cache(self, max_seq_len_cached, head_dim, base=1000000.0):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(torch.float32).unsqueeze(0)
        sin_cached = emb.sin().to(torch.float32).unsqueeze(0)
        return cos_cached, sin_cached

    def build(self, tokens, position_ids, attention_mask, *caches):
        bs, num_tokens = tokens.type.shape

        tokens = leap.transpose(tokens, (1, 0))
        hidden_states = self.embed_tokens(tokens)
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
        hidden_states = leap.reshape(hidden_states, [bs, num_tokens, -1])

        new_keys = []
        new_values = []
        self.cos = self.cos.to(device="cpu", dtype=torch.float16)
        self.sin = self.sin.to(device="cpu", dtype=torch.float16)

        position_ids = leap.reshape(position_ids, (bs, num_tokens, 1))
        cos = leap.gather_nd(self.cos, position_ids, batchDim=1)
        sin = leap.gather_nd(self.sin, position_ids, batchDim=1)

        cos = leap.reshape(cos, (bs, num_tokens, -1))
        sin = leap.reshape(sin, (bs, num_tokens, -1))

        position_embeddings = (cos, sin)

        cache_keys = caches[: len(caches) // 2]
        cache_values = caches[len(caches) // 2 :]

        all_hidden_states = []

        for idx, decoder_layer in enumerate(self.layers):
            if self.config.enable_eagle3:
                num_layers = self.config.num_hidden_layers
                if idx == 2 or idx == num_layers // 2 or idx == num_layers - 3:
                    all_hidden_states.append(hidden_states)

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

        if self.config.enable_eagle3:
            fused_hidden_states = leap.concat(all_hidden_states, -1)
            return token_logits, *new_keys, *new_values, fused_hidden_states

        return token_logits, *new_keys, *new_values

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        caches: List[torch.Tensor],
    ):
        hidden_states = self.embed_tokens(tokens)

        new_keys = []
        new_values = []
        self.cos = self.cos.to(position_ids.device).to(hidden_states.dtype)
        self.sin = self.sin.to(position_ids.device).to(hidden_states.dtype)

        position_ids = position_ids.unsqueeze(-1).expand(-1, -1, self.cos.size(-1))
        cos = torch.gather(self.cos, 1, position_ids)
        sin = torch.gather(self.sin, 1, position_ids)

        position_embeddings = (cos, sin)
        cache_keys = caches[: len(caches) // 2]
        cache_values = caches[len(caches) // 2 :]

        all_hidden_states = []

        for idx, decoder_layer in enumerate(self.layers):
            if self.config.enable_eagle3:
                num_layers = self.config.num_hidden_layers
                if idx == 2 or idx == num_layers // 2 or idx == num_layers - 3:
                    all_hidden_states.append(hidden_states)

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
        logits = self.lm_head(hidden_states)

        if self.config.enable_eagle3:
            fused_hidden_states = torch.cat(all_hidden_states, dim=-1)
            return logits, *new_keys, *new_values, fused_hidden_states

        return logits, *new_keys, *new_values


class Qwen3:
    @staticmethod
    @timeit
    def load_model(
        input_model_path: str,
        chunk_size=256,
        cache_len=4096,
        w_bits=8,
        enable_eagle3=False,
        num_draft_tokens=32,
        decode_seq_len=None,
    ) -> "Qwen3":
        config_path = os.path.join(input_model_path, "config.json")
        assert os.path.exists(
            config_path
        ), f"config.json not found in {input_model_path}"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # attention_bias = config.get("attention_bias", True)
        config_dict = {
            field.name: config.get(field.name, field.default)
            for field in fields(Qwen3Config)
        }

        config_dict["batch_size"] = 1
        config_dict["prefill_seq_len"] = chunk_size
        config_dict["context_len"] = cache_len
        # config_dict["num_hidden_layers"] = 1
        config_dict["w_bits"] = w_bits
        config_dict["enable_eagle3"] = enable_eagle3
        config_dict["num_draft_tokens"] = num_draft_tokens
        if enable_eagle3:
            config_dict["decode_seq_len"] = num_draft_tokens
        elif decode_seq_len is not None:
            config_dict["decode_seq_len"] = decode_seq_len

        config = Qwen3Config(**config_dict)
        new_state_dict = load_safetensors_state_dict(input_model_path)
        has_scale = any(".scales" in k for k in new_state_dict)
        config.has_scale = has_scale

        model = Qwen3Model(config)

        # _tied_weights_keys
        # If lm_head.weight is not found in the new_state_dict,
        # tie it to embed_tokens.weight to share input/output
        # embedding weights (weight tying).
        if "lm_head.weight" not in new_state_dict:
            new_state_dict["lm_head.weight"] = new_state_dict["embed_tokens.weight"]

        model.load_state_dict(new_state_dict, strict=True)
        print("Model load state_dict success.")

        return Qwen3(model, config)

    def __init__(self, model: Qwen3Model, config: Qwen3Config):
        self.model = model
        self.config = config

    def get_config(self):
        return self.config

    def get_leap_input_types(self, seq_len) -> List[leap.TensorType]:
        input_types = [
            leap.TensorType([self.config.batch_size, seq_len], leap.int32),
            leap.TensorType([self.config.batch_size, seq_len], leap.int32),
            leap.TensorType(
                [self.config.batch_size, seq_len, self.model.config.context_len],
                leap.float16,
            ),
        ]

        for _ in range(self.config.num_hidden_layers * 2):
            input_types.append(
                leap.TensorType(
                    [
                        self.config.batch_size,
                        self.config.context_len,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ],
                    leap.float32,
                )
            )
        return input_types

    def rename_graph_io(self, graph, num_hidden_layers):
        """
        Rename graph flatten inputs and outputs for LLM KV-cache style models.

        Inputs:
            0: input_ids
            1: position_ids
            2: attention_mask
            3 ~ 3+L-1: layer_i_cache_key
            3+L ~ 3+2L-1: layer_i_cache_value

        Outputs:
            0: logits
            1 ~ L: layer_i_new_key
            1+L ~ 1+2L-1: layer_i_new_value
        """
        # -------- inputs --------
        input_names = [
            "input_ids",
            "position_ids",
            "attention_mask",
        ]

        for i in range(num_hidden_layers):
            input_names.append(f"layer_{i}_cache_key")
        for i in range(num_hidden_layers):
            input_names.append(f"layer_{i}_cache_value")

        assert len(graph.flatten_inputs) >= len(
            input_names
        ), "flatten_inputs size mismatch"

        for tensor, name in zip(graph.flatten_inputs, input_names):
            tensor.name = name

        # -------- outputs --------
        output_names = ["logits"]

        for i in range(num_hidden_layers):
            output_names.append(f"layer_{i}_new_key")
        for i in range(num_hidden_layers):
            output_names.append(f"layer_{i}_new_value")

        if self.config.enable_eagle3:
            output_names.append("fused_hidden_states")

        assert len(graph.flatten_outputs) >= len(
            output_names
        ), "flatten_outputs size mismatch"

        for tensor, name in zip(graph.flatten_outputs, output_names):
            tensor.name = name

    def compile(
        self,
        stage: str,
        output_model_path: str,
        enable_vpu=True,
        prefill_core_num: list[int] = None,
        decode_core_num: list[int] = None,
        **kwargs,
    ):
        if decode_core_num is None:
            decode_core_num = [1]
        if prefill_core_num is None:
            prefill_core_num = [1]
        assert self.model.is_compiled, "Model must be compiled before compiling."

        def _validate_single_value_list(name: str, values: list[int]):
            if not isinstance(values, list):
                raise ValueError(f"{name} must be a list of int, got {type(values)}")
            if len(values) != 1:
                raise ValueError(
                    f"{name} must be a list of length 1, got {len(values)}: {values}"
                )

        _validate_single_value_list("prefill_core_num", prefill_core_num)
        _validate_single_value_list("decode_core_num", decode_core_num)

        stage_core_map = {
            "prefill": prefill_core_num[0],
            "decode": decode_core_num[0],
        }

        model_list = []
        stages = []
        if stage in {"prefill", "all"}:
            stages.append("prefill")
        if stage in {"decode", "all"}:
            stages.append("decode")

        for stage_name in stages:
            seq_len = (
                self.config.prefill_seq_len
                if stage_name == "prefill"
                else self.config.decode_seq_len
            )
            inputs = self.get_leap_input_types(seq_len)
            bc_path = str(Path(output_model_path).with_suffix(f".{stage_name}.bc"))
            bc_module = self.model.export_module(inputs, stage_name, bc_path)
            model_list.append(bc_module)

        hbos = []
        for bc_module in model_list:
            bc_module._skip_move_cpu_ops_pass = True
            func_name = bc_module.functions[0].name
            convert_bc_path = str(
                Path(output_model_path).with_suffix(f".{func_name}_convert.bc")
            )
            mlir_module = self.model.convert_mlir(
                bc_module,
                convert_bc_path,
                enable_vpu=enable_vpu,
                march=kwargs["march"],
                dynamic_quant=True,
                softmax_version="skip",
            )
            func = mlir_module.functions[0]
            func.remove_io_op(["Dequantize", "Quantize"])
            statistics(mlir_module)

            graph = mlir_module.graphs[0]
            self.rename_graph_io(graph, self.config.num_hidden_layers)

            convert_removed_bc_path = str(Path(output_model_path).with_suffix(f".{func_name}_convert_removed.bc"))
            save(mlir_module, convert_removed_bc_path)

            hbo_path = str(Path(output_model_path).with_suffix(f".{func_name}.hbo"))

            core_num = stage_core_map[func_name]
            kwargs["core_num"] = core_num
            kwargs["enable_hpc"] = True
            kwargs["input_no_padding"] = False
            kwargs["output_no_padding"] = False

            # 根据阶段设置参数
            if func_name == "prefill":
                if kwargs["core_num"] > 1:
                    kwargs["max_l2m_size"] = 25165824
                else:
                    kwargs.pop("max_l2m_size", None)
            elif func_name == "decode":
                kwargs["enable_hpc"] = False
                if kwargs["core_num"] > 1:
                    kwargs["max_l2m_size"] = 25165824
                else:
                    kwargs.pop("max_l2m_size", None)

            hbo_model = self.model.compile_hbo(
                mlir_module,
                hbo_path,
                **kwargs,
            )
            hbos.append(hbo_model)

        return self.model.link_models(hbos, output_model_path)
