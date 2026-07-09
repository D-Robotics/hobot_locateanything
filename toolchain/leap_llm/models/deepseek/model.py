import json
import os
import warnings
from dataclasses import asdict, dataclass, fields
from inspect import signature
from pathlib import Path
from typing import List

import torch
from hbdk4.compiler import leap, save

from leap_llm.models.deepseek.blocks import DecoderLayer  # noqa: E402
from leap_llm.nn.modules import (
    ConstFakeQuant,
    DynamicQuantLinear,
    FakeQuantEmbedding,  # noqa: E402
    FakeQuantLinear,
    FakeQuantRMSNorm,
    RMSNorm,
)
from leap_llm.nn.utils import Model, load_safetensors_state_dict, timeit  # noqa: E402


@dataclass
class ModelArgs:
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_attention_heads: int = 12
    num_hidden_layers: int = 28
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-06
    rope_theta: float = 10000
    max_position_embeddings: int = 131072
    max_batch_size: int = 32
    head_dim: int = int(hidden_size / num_attention_heads)
    prefill_seq_len: int = 256
    decode_seq_len: int = 1
    w_bits: int = 8
    has_scale: bool = False
    fuse_norm: bool = False


@dataclass
class QuantizeArgs:
    pass


class LLM(Model):
    def __init__(
        self,
        params: ModelArgs,
        cache_len: int,
        preserve_precision: bool = False,
        march: str = "nash-e",
    ):
        super().__init__()

        self.embed_tokens = FakeQuantEmbedding(params.vocab_size, params.hidden_size)
        self.layers = torch.nn.ModuleList()
        self.march = march

        DecoderLayerConfig = {
            k: v for k, v in asdict(params).items() if k in signature(DecoderLayer.__init__).parameters
        }
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(
                DecoderLayer(
                    layer_id=layer_id,
                    preserve_precision=preserve_precision,
                    march=self.march,
                    **DecoderLayerConfig,
                )
            )

        if "nash-p" in self.march:
            self.lm_head = DynamicQuantLinear(
                params.hidden_size,
                params.vocab_size,
                bias=False,
            )
        else:
            self.lm_head = FakeQuantLinear(params.hidden_size, params.vocab_size, bias=False)

        if "nash-p" in self.march:
            self.norm = RMSNorm(
                params.hidden_size,
                eps=params.rms_norm_eps,
            )
        else:
            self.norm = FakeQuantRMSNorm(
                params.hidden_size,
                eps=params.rms_norm_eps,
                preserve_precision=preserve_precision,
                fuse_norm=params.fuse_norm,
            )

        self.params = params
        self.cache_len = cache_len
        cos, sin = self._set_cos_sin_cache(
            params.max_position_embeddings,
            params.head_dim,
            base=params.rope_theta,
        )
        if "nash-p" in self.march:
            self.cos = cos.unsqueeze(0)[:, :cache_len, :]
            self.sin = sin.unsqueeze(0)[:, :cache_len, :]
        else:
            self.cos = cos[:cache_len, :]
            self.sin = sin[:cache_len, :]

        self.mask_fq = ConstFakeQuant(16)
        self.cos_fq = ConstFakeQuant(16)
        self.sin_fq = ConstFakeQuant(16)

    def _set_cos_sin_cache(self, max_seq_len_cached, head_dim, base=1000000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(torch.float32)
        sin_cached = emb.sin().to(torch.float32)
        return cos_cached, sin_cached

    def build(self, tokens, position_ids, mask, *caches):
        """Deepseek leap model forward function

        Args:
            tokens (int32):         (bs, seq_len)
            position_ids (int32):   (bs, seq_len)
            mask (fp16):            (bs, seq_len, ctx_len)
            caches (fp32):          (bs, ctx_len, num_kv_heads, head_dim)

        Returns:
            logits:                 (bs, seq_len, vocab_size)
            caches:                 (bs, seq_len, num_kv_heads, head_dim)
        """
        if "nash-p" in self.march:
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

            cache_keys = caches[: len(caches) // 2]
            cache_values = caches[len(caches) // 2 :]

            for idx, decoder_layer in enumerate(self.layers):
                hidden_states, new_key, new_value = decoder_layer(
                    hidden_states, cos, sin, cache_keys[idx], cache_values[idx], mask
                )
                new_keys.append(new_key)
                new_values.append(new_value)

            hidden_states = self.norm(hidden_states)
            token_logits = self.lm_head(hidden_states)

            return token_logits, *new_keys, *new_values
        else:
            _bsz, seqlen = tokens.type.shape
            tokens = leap.reshape(tokens, [seqlen, _bsz])
            hidden_states = self.embed_tokens(tokens)

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

                hidden_states, new_k, new_v = layer(hidden_states, cos, sin, cache_k, cache_v, mask)
                new_k = leap.transpose(new_k, [1, 0, 2])
                new_v = leap.transpose(new_v, [1, 0, 2])

                new_keys.append(new_k)
                new_values.append(new_v)

            hidden_states = self.norm(hidden_states)
            hidden_states = leap.reshape(hidden_states, [1, seqlen, self.params.hidden_size])
            logits = self.lm_head(hidden_states)

            return logits, *new_keys, *new_values

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        caches: List[torch.Tensor],
    ):
        if "nash-p" in self.march:
            hidden_states = self.embed_tokens(tokens)
            new_keys = []
            new_values = []
            self.cos = self.cos.to(position_ids.device).to(hidden_states.dtype)
            self.sin = self.sin.to(position_ids.device).to(hidden_states.dtype)
            position_ids = position_ids.unsqueeze(-1).expand(-1, -1, self.cos.size(-1))
            cos = torch.gather(self.cos, 1, position_ids)
            sin = torch.gather(self.sin, 1, position_ids)

            caches_k = caches[: len(caches) // 2]
            caches_v = caches[len(caches) // 2 :]

            for layer, cache_k, cache_v in zip(self.layers, caches_k, caches_v):
                hidden_states, new_k, new_v = layer(hidden_states, cos, sin, cache_k, cache_v, mask)
                new_keys.append(new_k)
                new_values.append(new_v)

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, *new_keys, *new_values
        else:
            hidden_states = self.embed_tokens(tokens)

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

                hidden_states, new_k, new_v = layer(hidden_states, cos, sin, cache_k, cache_v, mask)

                new_k = new_k.transpose(1, 0)
                new_v = new_v.transpose(1, 0)
                new_keys.append(new_k)
                new_values.append(new_v)

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, *new_keys, *new_values


class DeepSeek:
    @staticmethod
    @timeit
    def build(
        input_model_path: str,
        chunk_size=256,
        cache_len=512,
        preserve_precision=False,
        w_bits=8,
        march: str = "nash-e",
    ) -> "DeepSeek":
        config_path = os.path.join(input_model_path, "config.json")
        assert os.path.exists(config_path), f"config.json not found in {input_model_path}"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        model_args_dict = {field.name: config.get(field.name, field.default) for field in fields(ModelArgs)}

        # TODO: 不支持 batch
        model_args_dict["max_batch_size"] = 1
        model_args_dict["prefill_seq_len"] = chunk_size
        # model_args_dict["num_hidden_layers"] = 1
        model_args_dict["w_bits"] = w_bits

        model_args = ModelArgs(**model_args_dict)
        new_state_dict = load_safetensors_state_dict(input_model_path)
        has_scale = any(".scales" in k for k in new_state_dict)
        if has_scale:
            warnings.warn(f"Current checkpoint contains quantization info, w_bit = {w_bits} bits", stacklevel=2)
        model_args.has_scale = has_scale
        model_args.fuse_norm = has_scale

        model = LLM(
            model_args,
            cache_len=cache_len,
            preserve_precision=preserve_precision,
            march=march,
        )

        # _tied_weights_keys
        # If lm_head.weight is not found in the new_state_dict,
        # tie it to embed_tokens.weight to share input/output
        # embedding weights (weight tying).
        if "lm_head.weight" not in new_state_dict:
            new_state_dict["lm_head.weight"] = new_state_dict["embed_tokens.weight"]

        model.load_state_dict(new_state_dict, strict=True)
        print("Model load state_dict success.")

        return DeepSeek(model, model_args)

    def __init__(self, model: LLM, model_args: ModelArgs):
        self.model = model
        self.model_args = model_args

    def get_model_args(self):
        return self.model_args

    def get_leap_input_types_nash_p(self, seq_len) -> List[leap.TensorType]:
        input_types = [
            leap.TensorType([1, seq_len], leap.int32),
            leap.TensorType([1, seq_len], leap.int32),
            leap.TensorType(
                [
                    1,
                    seq_len,
                    self.model.cache_len,
                ],
                leap.float16,
            ),
        ]

        for _ in range(self.model_args.num_hidden_layers * 2):
            input_types.append(
                leap.TensorType(
                    [
                        1,
                        self.model.cache_len,
                        self.model_args.num_key_value_heads,
                        self.model_args.head_dim,
                    ],
                    leap.float32,
                )
            )
        return input_types

    def get_leap_input_types(self, seq_len) -> List[leap.TensorType]:
        input_types = [
            leap.TensorType([1, seq_len], leap.int32),
            leap.TensorType([seq_len], leap.int32),
            leap.TensorType([seq_len, self.model.cache_len], leap.float32),
        ]

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
        prefill_core_num: int,
        decode_core_num: int,
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
            # seq_len varies in P/D stage
            seq_len = self.model_args.prefill_seq_len if stage_name == "prefill" else self.model_args.decode_seq_len

            if "nash-p" in kwargs["march"]:
                inputs = self.get_leap_input_types_nash_p(seq_len)
            else:
                inputs = self.get_leap_input_types(seq_len)

            bc_path = str(Path(output_model_path).with_suffix(f".{stage_name}.bc"))
            bc_module = self.model.export_module(inputs, stage_name, bc_path)
            model_list.append(bc_module)

        hbos = []
        for bc_module in model_list:
            bc_module._skip_move_cpu_ops_pass = True
            func_name = bc_module.functions[0].name
            # print("=== func_name:", func_name)
            convert_bc_path = str(Path(output_model_path).with_suffix(f".{func_name}_convert.bc"))
            if "nash-p" in kwargs["march"]:
                
                mlir_module = self.model.convert_mlir(
                    bc_module,
                    convert_bc_path,
                    march=kwargs["march"],
                    enable_vpu=enable_vpu,
                    dynamic_quant=True,
                    softmax_version="skip",
                )
            else:
                mlir_module = self.model.convert_mlir(
                    bc_module,
                    convert_bc_path,
                    enable_vpu=enable_vpu,
                    march=kwargs["march"],
                )
            func = mlir_module.functions[0]
            # "Transpose" "Reshape",
            func.remove_io_op(["Dequantize", "Quantize"])
            convert_removed_bc_path = str(Path(output_model_path).with_suffix(f".{func_name}_convert_removed.bc"))
            save(mlir_module, convert_removed_bc_path)
            hbo_path = str(Path(output_model_path).with_suffix(f".{func_name}.hbo"))
            
            kwargs["enable_hpc"] = True
            kwargs["input_no_padding"] = False
            kwargs["output_no_padding"] = False

            if "prefill" in func_name:
                if prefill_core_num == 4:
                    kwargs["core_num"] = 4
                    kwargs["max_l2m_size"] = 25165824  # 24 Mb
                elif prefill_core_num == 2:
                    kwargs["core_num"] = 2
                    kwargs["max_l2m_size"] = 12582912  # 12 Mb
                else:
                    kwargs["core_num"] = prefill_core_num

            if "decode" in func_name:
                kwargs["enable_hpc"] = False
                if decode_core_num == 4:
                    kwargs["core_num"] = 4
                    kwargs["max_l2m_size"] = 25165824
                elif decode_core_num == 2:
                    kwargs["core_num"] = 2
                    kwargs["max_l2m_size"] = 12582912
                else:
                    kwargs["core_num"] = decode_core_num

            print(f"kwargs : {kwargs}")
            hbo_model = self.model.compile_hbo(
                mlir_module,
                hbo_path,
                **kwargs,
            )
            hbos.append(hbo_model)

        return self.model.link_models(hbos, output_model_path)

    def set_compile_mode(self, mode: bool):
        self.model.compile_mode(mode)

    def set_model_device(self, device, dtype):
        self.model.to(device, dtype=dtype)

    @staticmethod
    def verify_difference(result1, result2):
        pass

    @torch.inference_mode()
    def generate(self):
        pass

    def text_completion(self):
        pass
        # self.generate()

    def chat_completion(self):
        pass
        # self.generate()
