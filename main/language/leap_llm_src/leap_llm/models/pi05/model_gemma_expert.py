import math
from pathlib import Path

import torch
from hbdk4.compiler import leap
from safetensors.torch import load_file

from leap_llm.models.pi05.blocks.attention import RotaryPosEmb
from leap_llm.models.pi05.blocks.configuration_gemma import GemmaConfig
from leap_llm.models.pi05.blocks.configuration_paligemma import PaliGemmaConfig
from leap_llm.models.pi05.blocks.mlp import GemmaMLP
from leap_llm.models.pi05.blocks.rmsnorm import GemmaRMSNorm
from leap_llm.nn.modules import DynamicQuantLinear, DynamicQuantMatmul
from leap_llm.nn.utils import Model, Module, timeit  # noqa: E402


class Attention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.q_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = DynamicQuantLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.qk = DynamicQuantMatmul()
        self.sv = DynamicQuantMatmul()

        self.apply_rotary_pos_emb = RotaryPosEmb()

    def build(self, hidden_states, attention_mask, cache_k, cache_v, cos, sin):
        seqlen = hidden_states.type.shape[1]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(
            query_states, [seqlen, self.num_attention_heads, self.head_dim]
        )
        query_states = leap.transpose(query_states, [1, 0, 2])
        key_states = leap.reshape(
            key_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )

        key_states = leap.transpose(key_states, [1, 0, 2])
        value_states = leap.reshape(
            value_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )
        value_states = leap.transpose(value_states, [1, 0, 2])

        # xk, xv
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        key_states = leap.concat([cache_k, key_states], 1)
        value_states = leap.concat([cache_v, value_states], 1)
        _, c_len, _ = key_states.type.shape

        H, W, C = query_states.type.shape

        query_states = leap.reshape(
            query_states,
            [
                self.num_key_value_heads,
                self.num_key_value_groups * W,
                self.head_dim,
            ],
        )
        attn_weights = self.qk(query_states, key_states)
        attn_weights = leap.reshape(attn_weights, [H, seqlen, c_len])
        attn_weights = leap.mul(attn_weights, self.scaling)

        if attention_mask is not None:
            if len(attention_mask.type.shape) == len(attn_weights.type.shape) - 1:
                print("attn_weights:", attn_weights.type)
                attention_mask = leap.reshape(attention_mask, [1, seqlen, seqlen])
                print("attention_mask:", attention_mask.type)
            attn_weights = leap.add(attn_weights, attention_mask)

        # NOTE: test
        attn_weights = leap.softmax(attn_weights, -1)
        # ret = attn_weights
        attn_weights = leap.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, c_len],
        )
        value_states = leap.transpose(value_states, [0, 2, 1])
        attn_output = self.sv(attn_weights, value_states)
        attn_output = leap.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output = leap.transpose(attn_output, [1, 0, 2])
        attn_output = leap.reshape(attn_output, [seqlen, -1])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask, cache_k, cache_v, cos, sin):
        assert hidden_states.ndim == 3
        batch_size, seqlen, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            [seqlen, self.num_attention_heads, self.head_dim]
        ).transpose(1, 0)
        key_states = key_states.reshape(
            [seqlen, self.num_key_value_heads, self.head_dim]
        ).transpose(1, 0)
        value_states = value_states.reshape(
            [seqlen, self.num_key_value_heads, self.head_dim]
        ).transpose(1, 0)

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )


        key_states = torch.cat([cache_k, key_states], -2)
        value_states = torch.cat([cache_v, value_states], -2)
        _, c_len, _ = key_states.shape
        key_states_t = key_states.transpose(2, 1)

        H, W, C = query_states.shape
        query_states = query_states.reshape(
            [
                self.num_key_value_heads,
                self.num_key_value_groups * W,
                self.head_dim,
            ]
        )

        attn_weights = self.qk(query_states, key_states_t)
        attn_weights = attn_weights.reshape([H, seqlen, c_len])
        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.softmax(attn_weights, -1)

        attn_weights = torch.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, c_len],
        )

        attn_output = self.sv(attn_weights, value_states)

        attn_output = torch.reshape(attn_output, [H, seqlen, self.head_dim])


        attn_output = torch.transpose(attn_output, 1, 0)
        attn_output = torch.reshape(attn_output, [batch_size, seqlen, -1])
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DecoderLayer(Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=config.adarms_cond_dim
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=config.adarms_cond_dim
        )

    def build(self, hidden_states, attention_mask, cache_k, cache_v, cos, sin, adarms_cond=None):
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache_k=cache_k,
            cache_v=cache_v,
            cos=cos,
            sin=sin,
        )
        hidden_states = leap.add(leap.mul(hidden_states, gate), residual)

        # Fully Connected
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(leap.mul(hidden_states, gate), residual)

        return hidden_states

    def forward(self, hidden_states, attention_mask, cache_k, cache_v, cos, sin, adarms_cond=None):
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache_k=cache_k,
            cache_v=cache_v,
            cos=cos,
            sin=sin,
        )
        hidden_states = residual + hidden_states * gate

        # Fully Connected
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * gate
        return hidden_states


class GemmaModel(Model):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(
            [
                DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        cond_dim = config.adarms_cond_dim
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

        cos, sin = self._set_cos_sin_cache(
            config.max_position_embeddings,
            config.head_dim,
            base=config.rope_theta,
        )
        self.cos = cos
        self.sin = sin

    def _set_cos_sin_cache(self, max_seq_len_cached, head_dim, base=1000000.0):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # cos_cached = emb.cos().to(torch.float16)
        # sin_cached = emb.sin().to(torch.float16)
        cos_cached = emb.cos().to(torch.float16)
        sin_cached = emb.sin().to(torch.float16)
        return cos_cached, sin_cached

    def build(self, inputs_embeds, attention_mask, position_ids, caches, adarms_cond=None):
        _bsz, seqlen = position_ids.type.shape
        caches_k = caches[: len(caches) // 2]
        caches_v = caches[len(caches) // 2 :]
        position_ids = leap.reshape(position_ids, [seqlen, _bsz])
        cos = leap.gather_nd(self.cos, position_ids, 0)
        sin = leap.gather_nd(self.sin, position_ids, 0)

        hidden_states = inputs_embeds
        for decoder_layer, cache_k, cache_v in zip(
            self.layers[: self.config.num_hidden_layers], caches_k, caches_v
        ):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                cos=cos,
                sin=sin,
                adarms_cond=adarms_cond,
            )

        hidden_states, _ = self.norm(hidden_states, cond=adarms_cond)
        return hidden_states

    def forward(self, inputs_embeds, attention_mask, position_ids, caches, adarms_cond):
        caches_k = caches[: len(caches) // 2]
        caches_v = caches[len(caches) // 2 :]

        cos = self.cos.to(position_ids.device)[position_ids]
        sin = self.sin.to(position_ids.device)[position_ids]
        # embed positions
        hidden_states = inputs_embeds

        # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        for decoder_layer, cache_k, cache_v in zip(
            self.layers[: self.config.num_hidden_layers], caches_k, caches_v
        ):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                cos=cos,
                sin=sin,
                adarms_cond=adarms_cond,
            )

        hidden_states, _ = self.norm(hidden_states, cond=adarms_cond)

        return hidden_states


class GemmaExpert(Model):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.model = GemmaModel(config)
        self.action_in_proj = DynamicQuantLinear(32, config.hidden_size)
        self.action_out_proj = DynamicQuantLinear(config.hidden_size, 32)

        self.time_mlp_in = DynamicQuantLinear(config.hidden_size, config.hidden_size)
        self.time_mlp_out = DynamicQuantLinear(config.hidden_size, config.hidden_size)

        self.lm_head = DynamicQuantLinear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.action_horizon = config.action_horizon
        self.sinusoidal_lookup_table = self._build_sinusoidal_lookup_table(
            1.0, 0.0, 0.1, self.action_in_proj.out_features, 4e-3, 4.0
        )
        self.idx_id = [i + 1 for i in range(self.action_horizon)]

    def _build_sinusoidal_lookup_table(
        self, start, end, step, dimension, min_period, max_period
    ):
        """
        init 阶段：预计算正弦位置编码查表
        返回：list 或 tensor，用于直接查表
        """
        assert dimension % 2 == 0
        times = torch.arange(start, end, -step)  # [1.0, 0.9, ..., 0.1]

        dtype = torch.float64
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 2 * math.pi / period

        sin_input = times[:, None] * scaling_factor[None, :]

        table = torch.cat(
            [torch.sin(sin_input), torch.cos(sin_input)], dim=-1
        )  # [num_times, dim]
        return table

    def build(self, state, x_t, denoise_idx, attention_mask, position_ids, *caches):

        self.sinusoidal_lookup_table = self.sinusoidal_lookup_table.to("cpu")
        time_emb = leap.gather_nd(self.sinusoidal_lookup_table, denoise_idx, 0)
        time_emb = leap.cast_type(time_emb, output_type=leap.float16)
        time_emb = leap.reshape(time_emb, [1, 1024])
        action_emb = self.action_in_proj(x_t)


        x = self.time_mlp_in(time_emb)
        x = leap.swish(x)
        time_emb = self.time_mlp_out(x)
        time_emb = leap.swish(time_emb)

        embs = leap.concat([action_emb], dim=1)

        outputs_embeds = self.model(
            inputs_embeds=embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            caches=caches,
            adarms_cond=time_emb,
        )

        suffix_out = self.action_out_proj(outputs_embeds)
        suffix_out = leap.mul(suffix_out, -0.1)
        v_t = leap.add(x_t, suffix_out)
        return v_t

    def forward(self, state, x_t, denoise_idx, attention_mask, position_ids, caches):
        embs = []
        self.sinusoidal_lookup_table = self.sinusoidal_lookup_table.to(device=state.device)
        time_emb = self.sinusoidal_lookup_table[denoise_idx]

        time_emb = time_emb.type(dtype=state.dtype).to(device=state.device)

        action_emb = self.action_in_proj(x_t)
        x = self.time_mlp_in(time_emb)
        x = torch.nn.functional.silu(x)
        time_emb = self.time_mlp_out(x)
        time_emb = torch.nn.functional.silu(time_emb)
        
        embs.append(action_emb)
        embs = torch.cat(embs, dim=1)
        outputs_embeds = self.model(
            inputs_embeds=embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            caches=caches,
            adarms_cond=time_emb,
        )
        suffix_out = outputs_embeds[:, -self.action_horizon :]
        suffix_out = self.action_out_proj(suffix_out)
        v_t = x_t - 0.1 * suffix_out
        return v_t



class GemmaExpertModel:
    @staticmethod
    @timeit
    def build(model_dir: str, vision_tokens_num, action_horizon=50) -> "GemmaExpertModel":
        state_dict = load_file(model_dir)
        gemma_expert_prefix = "paligemma_with_expert.gemma_expert."

        filtered_state_dict = {}

        # 提取 vision_tower 部分
        filtered_state_dict.update(
            {
                k.replace(gemma_expert_prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(gemma_expert_prefix)
            }
        )
        filtered_state_dict["action_in_proj.weight"] = state_dict[
            "action_in_proj.weight"
        ]
        filtered_state_dict["action_in_proj.bias"] = state_dict["action_in_proj.bias"]
        filtered_state_dict["action_out_proj.weight"] = state_dict[
            "action_out_proj.weight"
        ]
        filtered_state_dict["action_out_proj.bias"] = state_dict["action_out_proj.bias"]
        filtered_state_dict["time_mlp_in.weight"] = state_dict[
            "time_mlp_in.weight"
        ]
        filtered_state_dict["time_mlp_in.bias"] = state_dict["time_mlp_in.bias"]
        filtered_state_dict["time_mlp_out.weight"] = state_dict["time_mlp_out.weight"]
        filtered_state_dict["time_mlp_out.bias"] = state_dict["time_mlp_out.bias"]

        use_adarms = [False, True]
        modelConfig = PaliGemmaConfig()
        modelConfig.text_config.hidden_size = 1024
        modelConfig.text_config.intermediate_size = 4096
        modelConfig.text_config.num_attention_heads = 8
        modelConfig.text_config.head_dim = 256
        modelConfig.text_config.num_hidden_layers = 18
        modelConfig.text_config.num_key_value_heads = 1
        modelConfig.text_config.hidden_activation = "gelu_pytorch_tanh"
        modelConfig.text_config.torch_dtype = "float16"
        modelConfig.text_config.vocab_size = 257152
        modelConfig.text_config.use_adarms = use_adarms[1]
        modelConfig.text_config.adarms_cond_dim = 1024 if use_adarms[1] else None
        modelConfig.text_config.vision_tokens_num = vision_tokens_num
        modelConfig.text_config.action_horizon = action_horizon
        gemma_expert = GemmaExpert(modelConfig.text_config)
        gemma_expert.load_state_dict(filtered_state_dict, strict=True)

        return GemmaExpertModel(gemma_expert, modelConfig)

    def __init__(self, model: GemmaExpert, model_args: PaliGemmaConfig):
        self.model = model
        self.model_args = model_args

    def get_leap_input_types(
        self, action_dim, action_horizon, token_len
    ) -> list[leap.TensorType]:
        input_types = [
            leap.TensorType([1, action_dim], leap.float16),
            leap.TensorType([1, action_horizon, action_dim], leap.float16),
            leap.TensorType([1], leap.int32),
            leap.TensorType(
                [1, 1, action_horizon, action_horizon + token_len],
                leap.float16,
            ),
            leap.TensorType([1, action_horizon], leap.int32),
        ]
        for _ in range(self.model_args.text_config.num_hidden_layers * 2):
            input_types.append(leap.TensorType([1, token_len, 256], leap.float16))
        return input_types

    def compile(
        self,
        output_model_path: str,
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        inputs = self.get_leap_input_types(
            32,
            self.model_args.text_config.action_horizon,
            self.model_args.text_config.vision_tokens_num * 3 + 200
        )
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "gemma_expert", bc_path)
        # 编译 HBO 模型并链接成最终模型
        hbos = []
        bc_path = str(Path(output_model_path).with_suffix(".convert.bc"))
        mlir_module = self.model.convert_mlir(
            bc_module,
            save_path=bc_path,
            march=kwargs["march"],
            dynamic_quant=True,
            softmax_version="skip",
        )

        kwargs["core_num"] = 4
        kwargs["max_l2m_size"] = 25165824

        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        hbo_model = self.model.compile_hbo(
            mlir_module,
            hbo_path,
            **kwargs,
        )
        hbos.append(hbo_model)

        hbm_path = str(Path(output_model_path).with_suffix(".hbm"))
        return self.model.link_models(hbos, hbm_path)
