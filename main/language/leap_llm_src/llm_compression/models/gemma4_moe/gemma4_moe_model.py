"""Gemma4 BaseQModel - text-only LLM with MoE, aligned with transformers Gemma4ForConditionalGeneration."""

import copy

import torch
import torch.nn as nn
from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.quantization.qconfig_setter import SetDynamicQuantTemplate
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from llm_compression.models.base_qmodel import (
    BaseQModel,
    ModuleNameTemplate,
    load_state_dict_with_metadata,
    nashp_default_qconfig_template,
    qint8,
    qint16,
    update_config_from_custom_config,
)
from llm_compression.models.generate_utils import get_module_device
from llm_compression.registry_factory import MODEL_REGISTRY
from llm_compression.utils import AttentionManager
from llm_compression.utils.logger import get_logger

from .model import Gemma4TextModel
from .process_utils import generate_func

logger = get_logger(__name__)


def _map_hf_key_to_ours(key):
    """Map HF state_dict key to llm_compression format (lm.* prefix)."""
    nk = key

    if "language_model." in nk:
        lm_idx = nk.index("language_model.")
        nk = nk[lm_idx + len("language_model.") :]

    if nk.startswith("model."):
        nk = nk[len("model.") :]

    if nk == "lm_head.weight":
        nk = "lm.lm_head.weight"
    elif not nk.startswith("lm."):
        nk = "lm." + nk

    return nk


class Gemma4ModelForGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm = Gemma4TextModel(config)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_config(self):
        return self.config

    def generate(self, inputs, do_sample=False, chunk_prefill=False):
        config = self.get_config()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return generate_func(
            config,
            self.prefill,
            self.decode,
            input_ids,
            attention_mask,
            do_sample=do_sample,
            chunk_prefill=chunk_prefill,
        )


@MODEL_REGISTRY
class Gemma4Moe(BaseQModel):
    def __init__(self, model_dir, custom_config=None):
        super().__init__(model_dir, custom_config)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

    def get_model_dtype(self):
        dtype_name = str(getattr(self.custom_config.model, "model_dtype", "float32"))
        assert hasattr(torch, dtype_name), f"Unsupported dtype: {dtype_name}"
        return dtype_name

    def build_model(self, model_dir):
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

        # Gemma4 has nested config
        text_config = model_config.text_config

        if (
            self.custom_config is not None
            and hasattr(self.custom_config, "model")
            and hasattr(self.custom_config.model, "text_config")
        ):
            update_config_from_custom_config(text_config, self.custom_config.model.text_config)

        _tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        text_config.pad_token_id = _tokenizer.pad_token_id
        # Use top-level eos_token_id list to catch all EOS tokens (e.g. [1, 106])
        top_eos = getattr(model_config, "eos_token_id", None)
        if isinstance(top_eos, list) and len(top_eos) > 1:
            text_config.eos_token_id = top_eos
        elif text_config.eos_token_id is None:
            text_config.eos_token_id = _tokenizer.eos_token_id

        model = Gemma4ModelForGeneration(text_config)

        # ── Load weights from HF ──
        from transformers import Gemma4ForConditionalGeneration

        hf_model = Gemma4ForConditionalGeneration.from_pretrained(
            model_dir, config=model_config, trust_remote_code=True
        )
        checkpoint = hf_model.state_dict()
        del hf_model

        new_state_dict = {}
        for key, value in checkpoint.items():
            if (
                "language_model." not in key
                and not key.startswith("model.layers.")
                and not key.startswith("model.embed_tokens")
                and key != "model.lm_head.weight"
                and key != "lm_head.weight"
                and "language_model." not in key
            ):
                continue
            nk = _map_hf_key_to_ours(key)
            new_state_dict[nk] = value

        # tie_word_embeddings
        if "lm.lm_head.weight" not in new_state_dict and "lm.embed_tokens.weight" in new_state_dict:
            new_state_dict["lm.lm_head.weight"] = new_state_dict["lm.embed_tokens.weight"]

        miss_key, unexpected_key = load_state_dict_with_metadata(model, new_state_dict)
        del checkpoint, new_state_dict
        model.lm.embed_tokens.fuse_embed_scale_into_weight()

        # prefill / decode
        if self.is_shared_lm_mode():
            model.prefill = model.lm
            model.decode = model.lm
        else:
            model.prefill = model.lm
            model.decode = copy.deepcopy(model.lm)

        dtype_name = self.get_model_dtype()
        model.to(dtype=getattr(torch, dtype_name))

        logger.info(f"miss_key: {miss_key}")
        logger.info(f"unexpected_key: {unexpected_key}")

        return model

    def get_generated_model_cfg(self, model_name):
        cfg = getattr(self.generated_model, model_name, None)
        if cfg is None:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return cfg.config

    def get_model_input_output_name(self, model_name):
        cfg = self.get_generated_model_cfg(model_name)
        n_layers = cfg.num_hidden_layers
        input_names = ["input_embeddings", "position_ids", "attention_mask", "slide_attention_mask"]
        for idx in range(n_layers):
            input_names.append(f"in_key_cache_{idx}")
            input_names.append(f"in_value_cache_{idx}")
        output_names = [
            "output_logits",
            *[f"out_key_cache_{idx}" for idx in range(n_layers)],
            *[f"out_value_cache_{idx}" for idx in range(n_layers)],
        ]
        return input_names, output_names

    def get_generated_model(self):
        return self.generated_model

    def get_model_trace_dummy_input(self, model_name):
        example_inputs = []
        cfg = self.get_generated_model_cfg(model_name)
        dtype_name = self.get_model_dtype()
        dtype = getattr(torch, dtype_name)
        max_kvcache_len = cfg.max_kvcache_len
        n_layers = cfg.num_hidden_layers
        layer_types = cfg.layer_types
        sliding_window = getattr(cfg, "sliding_window", None)

        if model_name in ("prefill", "lm"):
            seq_len = cfg.max_lm_input_len
            device = get_module_device(self.generated_model.prefill)
        elif model_name == "decode":
            seq_len = 1
            device = get_module_device(self.generated_model.decode)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Sliding attention mask has smaller KV dimension
        slide_kv_len = min(max_kvcache_len, sliding_window) if sliding_window else max_kvcache_len

        example_inputs.append(torch.ones(1, seq_len, cfg.hidden_size, device=device, dtype=dtype))
        example_inputs.append(torch.ones(1, seq_len, device=device, dtype=torch.int32))
        example_inputs.append(torch.randn(1, seq_len, max_kvcache_len, device=device, dtype=dtype))
        example_inputs.append(torch.randn(1, seq_len, slide_kv_len, device=device, dtype=dtype))
        caches = []
        for i in range(n_layers):
            if layer_types[i] == "sliding_attention":
                hd, nkv = cfg.head_dim, cfg.num_key_value_heads
                cache_len = sliding_window if sliding_window else max_kvcache_len
            else:
                hd = getattr(cfg, "global_head_dim", cfg.head_dim)
                nkv = getattr(cfg, "num_global_key_value_heads", cfg.num_key_value_heads)
                cache_len = max_kvcache_len
            caches.append(torch.randn(1, cache_len, nkv, hd, device=device, dtype=dtype))
        caches = caches + [c.clone() for c in caches]
        example_inputs.append(caches)
        return example_inputs

    def _message_content_to_text(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts) if parts else ""
        return str(content)

    def input_preprocess(self, message):
        normalized = []
        for msg in message:
            msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = self._message_content_to_text(content)
            normalized.append(msg)
        enable_thinking = getattr(self.custom_config.model, "enable_thinking", False)
        text = self.processor.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self.processor(text=text, return_tensors="pt")
        return inputs

    def output_postprocess(self, generated_ids):
        response = self.processor.decode(generated_ids[0], skip_special_tokens=False)
        parsed = self.processor.parse_response(response)
        if isinstance(parsed, dict):
            return parsed.get("content", str(parsed))
        return str(parsed)

    def get_kvcache_names(self, model_name):
        n_layers = self.get_generated_model_cfg(model_name).num_hidden_layers
        return [
            name
            for i in range(n_layers)
            for name in (
                f"layers.{i}.self_attn.cache_k_fq",
                f"layers.{i}.self_attn.cache_v_fq",
            )
        ]

    def get_qconfig_setting(self, model_name):
        cfg = self.get_generated_model_cfg(model_name)
        if get_march() in ("nash-e", "nash-m"):
            return super().get_qconfig_setting(model_name)
        q_template = nashp_default_qconfig_template()
        n_layers = cfg.num_hidden_layers

        module_name_config = {}
        for name in self.get_kvcache_names(model_name):
            if name.endswith("cache_k_fq"):
                module_name_config[name] = {"output": qint16}
            elif name.endswith("cache_v_fq"):
                module_name_config[name] = {"output": qint8}

        use_flash_attention = AttentionManager.is_flash_attn()
        for i in range(n_layers):
            if use_flash_attention:
                module_name_config[f"layers.{i}.self_attn.attention.qk_matmul"] = {"input": [qint8, "del"]}
                module_name_config[f"layers.{i}.self_attn.attention.sv_matmul"] = {"input": [qint16, "del"]}
            else:
                module_name_config[f"layers.{i}.self_attn._generated_matmul_0"] = {"input": [qint8, "del"]}
                module_name_config[f"layers.{i}.self_attn._generated_matmul_1"] = {"input": [qint16, "del"]}

        q_template = q_template + [ModuleNameTemplate(module_name_config, freeze=True)]
        q_template.append(
            SetDynamicQuantTemplate(
                op_kwargs={
                    nn.Linear: {"block_size": "full", "dim": -1},
                }
            )
        )
        return q_template
