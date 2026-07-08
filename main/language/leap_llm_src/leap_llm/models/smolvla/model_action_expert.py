"""SmolVLA action expert (cross-attn + flow matching) for BPU compile."""

import math
from pathlib import Path

import torch
from hbdk4.compiler import leap

from leap_llm.models.smolvla.blocks.attention import CrossExpertAttention, SmolVLAExpertSelfAttention
from leap_llm.models.smolvla.blocks.configuration_smolvlm import (
    SmolLM2Config,
    SmolVLAPolicyConfig,
)
from leap_llm.models.smolvla.blocks.mlp import SmolLM2MLP
from leap_llm.models.smolvla.blocks.rmsnorm import SmolLM2RMSNorm
from leap_llm.models.smolvla.smolvla_utils import load_policy_config
from leap_llm.models.smolvla.weight_mapper import expert_state_dict, load_full_state_dict
from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Model, timeit


class ExpertDecoderLayer(Model):
    def __init__(
        self,
        config: SmolLM2Config,
        layer_idx: int,
        use_self_attn: bool,
    ):
        super().__init__()
        if use_self_attn:
            self.self_attn = SmolVLAExpertSelfAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = CrossExpertAttention(config=config, layer_idx=layer_idx)
        self.mlp = SmolLM2MLP(config)
        self.input_layernorm = SmolLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SmolLM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._use_self_attn = use_self_attn

    def build(self, hidden_states, attention_mask, cache_k, cache_v, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self._use_self_attn:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                cos=cos,
                sin=sin,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                cos=cos,
                sin=sin,
            )
        hidden_states = leap.add(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(hidden_states, residual)
        return hidden_states

    def forward(self, hidden_states, attention_mask, cache_k, cache_v, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self._use_self_attn:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                cos=cos,
                sin=sin,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                cos=cos,
                sin=sin,
            )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class SmolVLAExpertCore(Model):
    def __init__(self, config: SmolLM2Config, policy_cfg: SmolVLAPolicyConfig):
        super().__init__()
        self.config = config
        self.policy_cfg = policy_cfg
        every = policy_cfg.self_attn_every_n_layers
        self.layers = torch.nn.ModuleList(
            [
                ExpertDecoderLayer(
                    config,
                    i,
                    use_self_attn=(every > 0 and i % every == 0),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = SmolLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        cos, sin = self._rope_cache(
            config.max_position_embeddings, config.head_dim, config.rope_theta
        )
        self.cos = cos
        self.sin = sin
        # Pre-computed RoPE at normalized positions [0..chunk_size-1] for cross-attention.
        # HF SmolVLA normalizes expert position IDs to start from 0 before applying RoPE
        # (forward_cross_attn_layer: expert_position_id -= min(expert_position_id)).
        norm_pos = torch.arange(policy_cfg.chunk_size, dtype=torch.int64)
        self.norm_cos = cos[norm_pos, :]
        self.norm_sin = sin[norm_pos, :]

    def _rope_cache(self, max_seq_len_cached, head_dim, base=100000.0):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.float16), emb.sin().to(torch.float16)

    def build(self, inputs_embeds, attention_mask, position_ids, caches):
        _bsz, seqlen = position_ids.type.shape
        caches_k = caches[: len(caches) // 2]
        caches_v = caches[len(caches) // 2 :]
        position_ids_flat = leap.reshape(position_ids, [seqlen, _bsz])
        # Absolute cos/sin for self-attention layers.
        cos_abs = leap.gather_nd(self.cos, position_ids_flat, 0)
        sin_abs = leap.gather_nd(self.sin, position_ids_flat, 0)
        # Normalized cos/sin for cross-attention layers: HF normalises expert
        # position IDs to start from 0 before applying RoPE, so the cross-attn
        # query sees positions [0..chunk_size-1] rather than absolute offsets.
        cos_norm = self.norm_cos
        sin_norm = self.norm_sin
        hidden_states = inputs_embeds
        for decoder_layer, cache_k, cache_v in zip(
            self.layers, caches_k, caches_v, strict=True
        ):
            cos = cos_abs if decoder_layer._use_self_attn else cos_norm
            sin = sin_abs if decoder_layer._use_self_attn else sin_norm
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                cos=cos,
                sin=sin,
            )
        return self.norm(hidden_states)

    def forward(self, inputs_embeds, attention_mask, position_ids, caches):
        caches_k = caches[: len(caches) // 2]
        caches_v = caches[len(caches) // 2 :]
        hidden_states = inputs_embeds
        norm_pos = position_ids - position_ids.min(dim=1, keepdim=True).values
        for decoder_layer, cache_k, cache_v in zip(
            self.layers, caches_k, caches_v, strict=True
        ):
            if decoder_layer._use_self_attn:
                cos = self.cos.to(position_ids.device)[position_ids]
                sin = self.sin.to(position_ids.device)[position_ids]
            else:
                cos = self.cos.to(position_ids.device)[norm_pos]
                sin = self.sin.to(position_ids.device)[norm_pos]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                cache_k=cache_k,
                cache_v=cache_v,
                cos=cos,
                sin=sin,
            )
        return self.norm(hidden_states)


class SmolVLAExpert(Model):
    def __init__(self, config: SmolLM2Config, policy_cfg: SmolVLAPolicyConfig):
        super().__init__()
        self.model = SmolVLAExpertCore(config, policy_cfg)
        self.action_in_proj = DynamicQuantLinear(
            policy_cfg.max_action_dim, config.hidden_size
        )
        self.action_out_proj = DynamicQuantLinear(
            config.hidden_size, policy_cfg.max_action_dim
        )
        self.action_time_mlp_in = DynamicQuantLinear(
            2 * config.hidden_size, config.hidden_size
        )
        self.action_time_mlp_out = DynamicQuantLinear(
            config.hidden_size, config.hidden_size
        )
        self.action_horizon = policy_cfg.chunk_size
        self.sinusoidal_lookup_table = self._build_sinusoidal_lookup_table(
            1.0,
            0.0,
            1.0 / policy_cfg.num_steps,
            config.hidden_size,
            policy_cfg.min_period,
            policy_cfg.max_period,
        )
        self.dt = -1.0 / policy_cfg.num_steps

    def _build_sinusoidal_lookup_table(
        self, start, end, step, dimension, min_period, max_period
    ):
        assert dimension % 2 == 0
        times = torch.arange(start, end, -step)
        fraction = torch.linspace(0.0, 1.0, dimension // 2)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 2 * math.pi / period
        sin_input = times[:, None] * scaling_factor[None, :]
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)

    def _embed_suffix(self, x_t, denoise_idx):
        time_emb = self.sinusoidal_lookup_table[denoise_idx.item()]  # [hidden]
        time_emb = time_emb.to(dtype=x_t.dtype, device=x_t.device)
        action_emb = self.action_in_proj(x_t)  # [B, L, hidden]
        time_emb = time_emb[None, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        x = torch.nn.functional.silu(self.action_time_mlp_in(action_time_emb))
        return self.action_time_mlp_out(x)

    def build(self, x_t, denoise_idx, attention_mask, position_ids, *caches):
        time_emb = leap.gather_nd(self.sinusoidal_lookup_table, denoise_idx, 0)
        action_emb = self.action_in_proj(x_t)
        time_emb = leap.reshape(time_emb, [1, 1, -1])
        time_emb = leap.tile(time_emb, [1, action_emb.type.shape[1], 1])
        time_emb = leap.cast_type(time_emb, output_type=leap.float16)
        action_time_emb = leap.concat([action_emb, time_emb], dim=2)
        x = self.action_time_mlp_in(action_time_emb)
        x = leap.swish(x)
        action_time_emb = self.action_time_mlp_out(x)
        outputs_embeds = self.model(
            inputs_embeds=action_time_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            caches=caches,
        )
        hidden_dim = outputs_embeds.type.shape[2]
        suffix_out = leap.slice(
            outputs_embeds,
            [0, 0, 0],
            [1, self.action_horizon, hidden_dim],
            [1, 1, 1],
        )
        suffix_out = self.action_out_proj(suffix_out)
        suffix_out = leap.mul(suffix_out, self.dt)
        return leap.add(x_t, suffix_out)

    def forward(self, x_t, denoise_idx, attention_mask, position_ids, caches):
        action_time_emb = self._embed_suffix(x_t, denoise_idx)
        outputs_embeds = self.model(
            inputs_embeds=action_time_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            caches=caches,
        )
        suffix_out = outputs_embeds[:, -self.action_horizon :]
        x_t = x_t + self.dt * self.action_out_proj(suffix_out)
        return x_t


class SmolVLMActionExpert:
    @staticmethod
    @timeit
    def build(
        model_path: str,
        policy_cfg: SmolVLAPolicyConfig | None = None,
        vision_tokens_num: int | None = None,
    ) -> "SmolVLMActionExpert":
        root = Path(model_path)
        if policy_cfg is None:
            policy_cfg = load_policy_config(root)
        if vision_tokens_num is not None:
            policy_cfg.vision_tokens_num = vision_tokens_num

        state = load_full_state_dict(root)
        e_state = expert_state_dict(state)
        expert_cfg = SmolLM2Config.from_policy(policy_cfg, for_expert=True)
        expert_cfg.vision_tokens_num = policy_cfg.vision_tokens_num

        model = SmolVLAExpert(expert_cfg, policy_cfg)
        model.load_state_dict(e_state, strict=False)
        return SmolVLMActionExpert(model, policy_cfg)

    def __init__(self, model: SmolVLAExpert, policy_cfg: SmolVLAPolicyConfig):
        self.model = model
        self.policy_cfg = policy_cfg

    def get_leap_input_types(
        self, action_dim: int, action_horizon: int, prefix_len: int
    ) -> list[leap.TensorType]:
        input_types = [
            leap.TensorType([1, action_horizon, action_dim], leap.float16),
            leap.TensorType([1], leap.int32),
            leap.TensorType(
                [1, 1, action_horizon, prefix_len + action_horizon], leap.float16
            ),
            leap.TensorType([1, action_horizon], leap.int32),
        ]
        nkv = self.policy_cfg.text_num_key_value_heads
        head_dim = self.policy_cfg.text_head_dim
        for _ in range(self.policy_cfg.active_expert_layers * 2):
            input_types.append(
                leap.TensorType([nkv, prefix_len, head_dim], leap.float16)
            )
        return input_types

    def compile(self, output_model_path: str, **kwargs):
        assert self.model.is_compiled, "Model must be compiled before compiling."
        from leap_llm.models.smolvla.smolvla_utils import prefix_sequence_len

        p = self.policy_cfg
        prefix_len = prefix_sequence_len(p)
        inputs = self.get_leap_input_types(
            p.max_action_dim, p.chunk_size, prefix_len
        )
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "smolvla_action_expert", bc_path)
        hbos = []
        bc_path = str(Path(output_model_path).with_suffix(".convert.bc"))
        mlir_module = self.model.convert_mlir(
            bc_module,
            save_path=bc_path,
            march=kwargs["march"],
            dynamic_quant=True,
        )
        kwargs["core_num"] = kwargs.get("core_num", 1)
        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        hbo_model = self.model.compile_hbo(mlir_module, hbo_path, **kwargs)
        hbos.append(hbo_model)
        return self.model.link_models(hbos, str(Path(output_model_path).with_suffix(".hbm")))
