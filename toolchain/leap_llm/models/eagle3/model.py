import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List

import torch
from hbdk4.compiler import leap, statistics

from leap_llm.models.eagle3.blocks import Eagle3DecoderLayer
from leap_llm.nn.modules import DynamicQuantLinear, FakeQuantEmbedding, RMSNorm
from leap_llm.nn.utils import Model, timeit

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Eagle3DraftConfig:
    vocab_size: int = 151936
    draft_vocab_size: int = 32000
    hidden_size: int = 4096
    head_dim: int = 128
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_hidden_layers: int = 1
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000
    max_position_embeddings: int = 40960
    hidden_act: str = "silu"
    target_hidden_size: int = 0
    batch_size: int = 1
    prefill_seq_len: int = 256
    decode_seq_len: int = 10 # topk eagle3 的参数
    speculative_num_steps: int = 7 # depth eagle3 的参数
    context_len: int = 4096
    w_bits: int = 8
    has_scale: bool = False


class Eagle3DraftModel(Model):
    def __init__(self, config: Eagle3DraftConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = FakeQuantEmbedding(
            config.vocab_size, config.hidden_size
        )
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        self.fc = DynamicQuantLinear(
            config.target_hidden_size,
            config.hidden_size,
            bias=False,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )

        self.midlayer = Eagle3DecoderLayer(config)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = DynamicQuantLinear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        self.context_len = config.context_len

        cos, sin = self._set_cos_sin_cache(
            config.max_position_embeddings,
            config.head_dim,
            base=config.rope_theta,
        )
        self.cos = cos[:, : self.context_len, :]
        self.sin = sin[:, : self.context_len, :]

        d2t = torch.zeros(config.draft_vocab_size, dtype=torch.long)
        t2d = torch.zeros(config.vocab_size, dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

    def _set_cos_sin_cache(self, max_seq_len_cached, head_dim, base=1000000.0):
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(torch.float32).unsqueeze(0)
        sin_cached = emb.sin().to(torch.float32).unsqueeze(0)
        return cos_cached, sin_cached

    def build(self, input_ids, position_ids, attention_mask, cache_keys, cache_values, hidden_states):
        bs, num_tokens, input_dim = hidden_states.type.shape

        # Embedding lookup (same transpose pattern as Qwen3)
        input_ids_t = leap.transpose(input_ids, (1, 0))
        inputs_embeds = self.embed_tokens(input_ids_t)
        inputs_embeds = leap.cast_type(inputs_embeds, output_type=leap.float16)
        inputs_embeds = leap.reshape(inputs_embeds, [bs, num_tokens, -1])

        # Conditional fc: only when input comes from target model (dim=hidden*3)
        if input_dim != self.config.hidden_size:
            hidden_states = self.fc(hidden_states)

        # Position embeddings (RoPE)
        self.cos = self.cos.to(device="cpu", dtype=torch.float16)
        self.sin = self.sin.to(device="cpu", dtype=torch.float16)

        position_ids = leap.reshape(position_ids, (bs, num_tokens, 1))
        cos = leap.gather_nd(self.cos, position_ids, batchDim=1)
        sin = leap.gather_nd(self.sin, position_ids, batchDim=1)
        cos = leap.reshape(cos, (bs, num_tokens, -1))
        sin = leap.reshape(sin, (bs, num_tokens, -1))
        position_embeddings = (cos, sin)

        # ===== cache_keys: tensor<1x1024x8x128xf32>
        hidden_states, new_key, new_value = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        output_hidden = hidden_states
        hidden_states = self.norm(hidden_states)
        token_logits = self.lm_head(hidden_states)

        return token_logits, new_key, new_value, output_hidden

    def forward(
        self,    
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        caches: List[torch.Tensor],
        hidden_states: torch.Tensor,
    ):
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        if hidden_states.shape[-1] != self.config.hidden_size:
            hidden_states = self.fc(hidden_states)

        # Position embeddings (RoPE)
        self.cos = self.cos.to(position_ids.device).to(hidden_states.dtype)
        self.sin = self.sin.to(position_ids.device).to(hidden_states.dtype)

        position_ids_exp = position_ids.unsqueeze(-1).expand(
            -1, -1, self.cos.size(-1)
        )
        cos = torch.gather(self.cos, 1, position_ids_exp)
        sin = torch.gather(self.sin, 1, position_ids_exp)
        position_embeddings = (cos, sin)

        cache_keys = caches[:1]
        cache_values = caches[1:]

        hidden_states, new_key, new_value = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys[0] if len(cache_keys) else None,
            cache_values=cache_values[0] if len(cache_values) else None,
        )

        output_hidden = hidden_states
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_key, new_value, output_hidden


class Eagle3Draft:
    @staticmethod
    @timeit
    def load_model(
        draft_model_path: str,
        target_model_path: str,
        chunk_size: int = 256,
        cache_len: int = 4096,
        eagle_topk: int = 6,
        speculative_num_steps: int = 7, # depth
        w_bits: int = 8,
        torch_dtype: torch.dtype = torch.float32,
    ) -> "Eagle3Draft":
        config_path = os.path.join(draft_model_path, "config.json")
        assert os.path.exists(config_path), (
            f"config.json not found in {draft_model_path}"
        )
        with open(config_path, encoding="utf-8") as f:
            raw_config = json.load(f)

        config_dict = {
            field.name: raw_config.get(field.name, field.default)
            for field in fields(Eagle3DraftConfig)
        }

        base_hidden_size = raw_config.get("hidden_size", 4096)
        config_dict["target_hidden_size"] = base_hidden_size * 3
        config_dict["batch_size"] = 1
        config_dict["prefill_seq_len"] = chunk_size
        config_dict["decode_seq_len"] = eagle_topk
        config_dict["speculative_num_steps"] = speculative_num_steps
        config_dict["context_len"] = cache_len
        config_dict["w_bits"] = w_bits

        config = Eagle3DraftConfig(**config_dict)

        model = Eagle3DraftModel(config)

        # Load draft-specific weights from pytorch_model.bin
        draft_weights_path = os.path.join(draft_model_path, "pytorch_model.bin")
        assert os.path.exists(draft_weights_path), (
            f"pytorch_model.bin not found in {draft_model_path}"
        )
        draft_state = torch.load(draft_weights_path, map_location="cpu")

        # Assemble state dict
        state_dict = {}
        for k, v in draft_state.items():
            if k in ("d2t", "t2d"):
                continue
            state_dict[k] = v

        # NOTE:
        # TODO: junjun.zhao modify this code 
        """
        如果 target model 经过 quarot 变化，则需要给 draft model 的 
        部分权重也要进行变换（现在这部分代码是在外部写脚本处理的）
        目前临时代码这么写，如果 bin 中包含 embed_tokens 则使用 自带的
        """
        if "embed_tokens.weight" in draft_state:
            print("Using embed_tokens.weight from draft model bin (e.g. quarot-transformed).")
        else:
            print("embed_tokens.weight not found in draft model bin, loading from target model.")
            state_dict["embed_tokens.weight"] = _load_target_embed_tokens(target_model_path)

        has_scale = any(".scales" in k for k in state_dict)
        config.has_scale = has_scale

        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"Eagle3Draft load_state_dict: {load_result}")

        # Load buffers
        if "d2t" in draft_state:
            model.d2t.copy_(draft_state["d2t"])
        if "t2d" in draft_state:
            model.t2d.copy_(draft_state["t2d"])

        # Convert model to specified dtype
        model = model.to(torch_dtype)

        print(f"Eagle3Draft model loaded successfully (dtype={torch_dtype}).")
        return Eagle3Draft(model, config)

    def __init__(self, model: Eagle3DraftModel, config: Eagle3DraftConfig):
        self.model = model
        self.config = config

    def get_config(self):
        return self.config

    def get_leap_input_types(self, seq_len, is_prefill=False) -> List[leap.TensorType]:
        hidden_dim = (
            self.config.target_hidden_size
            if is_prefill
            else self.config.hidden_size
        )
        input_types = [            
            # input_ids
            leap.TensorType([self.config.batch_size, seq_len], leap.int32),
            # position_ids
            leap.TensorType([self.config.batch_size, seq_len], leap.int32),
            # attention_mask
            leap.TensorType(
                [self.config.batch_size, seq_len, self.config.context_len],
                leap.float16,
            ),
        ]

        # 1 layer KV cache: cache_key + cache_value
        for _ in range(self.config.num_hidden_layers*2):
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

        # hidden_states
        input_types.append(
            leap.TensorType(
                [self.config.batch_size, seq_len, hidden_dim], leap.float16
            )
        )
        return input_types

    def rename_graph_io(self, graph):
        """Rename graph I/O for the EAGLE3 draft model.

        Inputs:            
            0: input_ids
            1: position_ids
            2: attention_mask
            3: layer_0_cache_key
            4: layer_0_cache_value
            5: hidden_states

        Outputs:
            0: logits
            1: layer_0_new_key
            2: layer_0_new_value
            3: output_hidden_states
        """
        input_names = [
            "input_ids",
            "position_ids",
            "attention_mask",
            "layer_0_cache_key",
            "layer_0_cache_value",
            "hidden_states",
        ]

        assert len(graph.flatten_inputs) >= len(input_names), (
            f"flatten_inputs size mismatch: {len(graph.flatten_inputs)} < {len(input_names)}"
        )
        for tensor, name in zip(graph.flatten_inputs, input_names):
            tensor.name = name

        output_names = [
            "logits",
            "layer_0_new_key",
            "layer_0_new_value",
            "output_hidden_states",
        ]

        assert len(graph.flatten_outputs) >= len(output_names), (
            f"flatten_outputs size mismatch: {len(graph.flatten_outputs)} < {len(output_names)}"
        )
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
        assert self.model.is_compiled, "Model must be in compiled mode."

        darft_ext_name = "extend"
        prefill_name = "prefill"
        decode_name = "decode"

        stage_core_map = {
            prefill_name: prefill_core_num[0],
            decode_name: decode_core_num[0],
            darft_ext_name: decode_core_num[0],
        }

        model_list = []
        stages = []
        if stage in {prefill_name, "all"}:
            stages.append(prefill_name)
        if stage in {decode_name, "all"}:
            stages.append(decode_name)
        if stage in {darft_ext_name, "all"}:
            stages.append(darft_ext_name)            

        for stage_name in stages:
            is_prefill = stage_name == prefill_name
            seq_len = (
                self.config.prefill_seq_len if is_prefill else self.config.decode_seq_len
            )
            if stage_name == darft_ext_name:
                # depth eagle3 的参数 + 1 就是包含 draft model prefill 部分 再加上 base model 
                seq_len = self.config.speculative_num_steps + 1 + 1 
                is_prefill = True

            inputs = self.get_leap_input_types(seq_len, is_prefill=is_prefill)
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
            graph = mlir_module.graphs[0]
            self.rename_graph_io(graph)

            statistics(mlir_module)
            hbo_path = str(
                Path(output_model_path).with_suffix(f".{func_name}.hbo")
            )

            core_num = stage_core_map[func_name]
            kwargs["core_num"] = core_num
            kwargs["enable_hpc"] = True
            kwargs["input_no_padding"] = False
            kwargs["output_no_padding"] = False

            if func_name in [prefill_name, darft_ext_name]:
                if kwargs["core_num"] > 1:
                    kwargs["max_l2m_size"] = 25165824
                else:
                    kwargs.pop("max_l2m_size", None)
            elif func_name in [decode_name]:
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


def _load_target_embed_tokens(target_model_path: str) -> torch.Tensor:
    """Load embed_tokens weight from the target (base) model safetensors."""
    from safetensors import safe_open

    index_json_path = os.path.join(
        target_model_path, "model.safetensors.index.json"
    )
    if os.path.exists(index_json_path):
        with open(index_json_path) as f:
            index_json = json.load(f)
        emb_file = index_json["weight_map"]["model.embed_tokens.weight"]
        emb_path = os.path.join(target_model_path, emb_file)
        with safe_open(emb_path, framework="pt", device="cpu") as f:
            tensor_slice = f.get_slice("model.embed_tokens.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
        return tensor

    # Fallback: single safetensors file
    single_path = os.path.join(target_model_path, "model.safetensors")
    if os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            return f.get_tensor("model.embed_tokens.weight").float()

    # Fallback: pytorch_model.bin
    bin_index_path = os.path.join(
        target_model_path, "pytorch_model.bin.index.json"
    )
    if os.path.exists(bin_index_path):
        with open(bin_index_path) as f:
            index_json = json.load(f)
        emb_file = index_json["weight_map"]["model.embed_tokens.weight"]
        emb_path = os.path.join(target_model_path, emb_file)
        weights = torch.load(emb_path, map_location="cpu")
        return weights["model.embed_tokens.weight"].float()

    raise FileNotFoundError(
        f"Cannot find embed_tokens weight in {target_model_path}. "
        "Expected model.safetensors.index.json, model.safetensors, "
        "or pytorch_model.bin.index.json."
    )
