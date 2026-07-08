"""SmolLM2 prefix (VLM text) model for SmolVLA."""

import math
from pathlib import Path

import torch
from hbdk4.compiler import leap

from leap_llm.models.smolvla.blocks.attention import SmolLM2Attention
from leap_llm.models.smolvla.blocks.configuration_smolvlm import (
    SmolLM2Config,
    SmolVLAPolicyConfig,
)
from leap_llm.models.smolvla.blocks.mlp import SmolLM2MLP
from leap_llm.models.smolvla.blocks.rmsnorm import SmolLM2RMSNorm
from leap_llm.models.smolvla.smolvla_utils import load_policy_config
from leap_llm.models.smolvla.weight_mapper import load_full_state_dict, text_state_dict
from leap_llm.nn.modules import DynamicQuantLinear, Embedding
from leap_llm.nn.utils import Model, timeit


class SmolLM2DecoderLayer(Model):
    def __init__(self, config: SmolLM2Config, layer_idx: int):
        super().__init__()
        self.self_attn = SmolLM2Attention(config=config, layer_idx=layer_idx)
        self.mlp = SmolLM2MLP(config)
        self.input_layernorm = SmolLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SmolLM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def build(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, new_k, new_v = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )
        hidden_states = leap.add(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(hidden_states, residual)
        return hidden_states, new_k, new_v

    def forward(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, new_k, new_v = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states, new_k, new_v


class SmolLM2PrefixModel(Model):
    def __init__(self, config: SmolLM2Config, policy_cfg: SmolVLAPolicyConfig):
        super().__init__()
        self.config = config
        self.policy_cfg = policy_cfg
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.state_proj = DynamicQuantLinear(
            policy_cfg.max_state_dim, config.hidden_size
        )
        self.layers = torch.nn.ModuleList(
            [
                SmolLM2DecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = SmolLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        cos, sin = self._rope_cache(
            config.max_position_embeddings, config.head_dim, config.rope_theta
        )
        # RoPE cache must cover the full prefix: all cameras + lang tokens + state token
        max_seq = config.vision_tokens_num * policy_cfg.num_images + policy_cfg.tokenizer_max_length + 1
        self.cos = cos[:max_seq, :]
        self.sin = sin[:max_seq, :]
        self._embed_scale = math.sqrt(config.hidden_size)

    def _rope_cache(self, max_seq_len_cached, head_dim, base=100000.0):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.float16), emb.sin().to(torch.float16)

    def _build_prefix_hidden(self, tokens, inputs_embeds, state):
        _bsz, seqlen = tokens.type.shape
        tokens = leap.reshape(tokens, [seqlen, _bsz])
        lang_emb = self.embed_tokens(tokens)
        lang_emb = leap.reshape(lang_emb, (1, seqlen, self.config.hidden_size))
        inputs_embeds = leap.mul(inputs_embeds, float(self._embed_scale))
        lang_emb = leap.mul(lang_emb, float(self._embed_scale))
        state_emb = self.state_proj(state)
        state_emb = leap.reshape(state_emb, [1, 1, -1])
        return leap.concat([inputs_embeds, lang_emb, state_emb], dim=1)

    def build(self, tokens, inputs_embeds, state, attention_mask, position_ids):
        hidden_states = self._build_prefix_hidden(tokens, inputs_embeds, state)
        _bsz, seqlen = position_ids.type.shape
        position_ids = leap.reshape(position_ids, [seqlen, _bsz])
        cos = leap.gather_nd(self.cos, position_ids, 0)
        sin = leap.gather_nd(self.sin, position_ids, 0)
        new_keys, new_values = [], []
        for decoder_layer in self.layers:
            hidden_states, new_k, new_v = decoder_layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
            )
            new_keys.append(new_k)
            new_values.append(new_v)
        hidden_states = self.norm(hidden_states)
        return hidden_states, *new_keys, *new_values

    def forward(self, tokens, inputs_embeds, state, attention_mask, position_ids=None):
        lang_emb = self.embed_tokens(tokens)
        inputs_embeds = inputs_embeds * self._embed_scale
        lang_emb = lang_emb * self._embed_scale
        state_emb = self.state_proj(state)
        if state_emb.ndim == 2:
            state_emb = state_emb[:, None, :]
        hidden_states = torch.concat([inputs_embeds, lang_emb, state_emb], dim=1)
        seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        cos = self.cos.to(hidden_states.device)[position_ids]
        sin = self.sin.to(hidden_states.device)[position_ids]

        new_keys, new_values = [], []
        for decoder_layer in self.layers:
            hidden_states, new_k, new_v = decoder_layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
            )
            new_keys.append(new_k)
            new_values.append(new_v)
        hidden_states = self.norm(hidden_states)
        return hidden_states, *new_keys, *new_values


class SmolVLMPrefix:
    @staticmethod
    @timeit
    def build(
        model_path: str,
        policy_cfg: SmolVLAPolicyConfig | None = None,
        vision_tokens_num: int | None = None,
    ) -> "SmolVLMPrefix":
        root = Path(model_path)
        if policy_cfg is None:
            policy_cfg = load_policy_config(root)
        if vision_tokens_num is not None:
            policy_cfg.vision_tokens_num = vision_tokens_num

        state = load_full_state_dict(root)
        t_state = text_state_dict(state)
        text_cfg = SmolLM2Config.from_policy(policy_cfg, for_expert=False)
        text_cfg.vision_tokens_num = policy_cfg.vision_tokens_num

        model = SmolLM2PrefixModel(text_cfg, policy_cfg)
        model.load_state_dict(t_state, strict=False)
        return SmolVLMPrefix(model, policy_cfg)

    def __init__(self, model: SmolLM2PrefixModel, policy_cfg: SmolVLAPolicyConfig):
        self.model = model
        self.policy_cfg = policy_cfg

    def get_leap_input_types(
        self, seq_len: int, token_id_len: int, state_dim: int | None = None
    ) -> list[leap.TensorType]:
        h = self.policy_cfg.text_hidden_size
        state_dim = state_dim or self.policy_cfg.max_state_dim
        total_len = seq_len + token_id_len + 1
        return [
            leap.TensorType([1, token_id_len], leap.int32),
            leap.TensorType([1, seq_len, h], leap.float16),
            leap.TensorType([1, state_dim], leap.float16),
            leap.TensorType([1, 1, total_len, total_len], leap.float16),
            leap.TensorType([1, total_len], leap.int32),
        ]

    def compile(self, output_model_path: str, **kwargs):
        assert self.model.is_compiled, "Model must be compiled before compiling."
        from leap_llm.models.smolvla.smolvla_utils import prefix_sequence_len

        vision_len = prefix_sequence_len(self.policy_cfg) - self.policy_cfg.tokenizer_max_length - 1
        token_len = self.policy_cfg.tokenizer_max_length
        inputs = self.get_leap_input_types(vision_len, token_len)
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "smolvla_vlm_prefix", bc_path)
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
