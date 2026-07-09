"""Qwen3Moe BaseQModel - pure LLM MoE, aligned with transformers Qwen3MoeForCausalLM."""

import copy

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

try:
    from transformers import Qwen3MoeForCausalLM
except ImportError:
    Qwen3MoeForCausalLM = None

from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.quantization.qconfig_setter import SetDynamicQuantTemplate

from llm_compression.models.base_qmodel import (
    BaseQModel,
    ModuleNameTemplate,
    load_state_dict_with_metadata,
    nashp_default_qconfig_template,
    qint8,
    update_config_from_custom_config,
)
from llm_compression.models.generate_utils import get_module_device
from llm_compression.registry_factory import MODEL_REGISTRY
from llm_compression.utils.logger import get_logger

from .model import Qwen3MoeTextModel
from .process_utils import generate_func

logger = get_logger(__name__)


class Qwen3MoeModelForGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm = Qwen3MoeTextModel(config)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_config(self):
        return self.config

    def generate(self, inputs, do_sample=False, chunk_prefill=False):
        config = self.get_config()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        result = generate_func(
            config,
            self.prefill,
            self.decode,
            input_ids,
            attention_mask,
            do_sample=do_sample,
            chunk_prefill=chunk_prefill,
        )
        return result


@MODEL_REGISTRY
class Qwen3Moe(BaseQModel):
    def __init__(self, model_dir, custom_config=None):
        super().__init__(model_dir, custom_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def get_model_dtype(self):
        dtype_name = str(getattr(self.custom_config.model, "model_dtype", "float32"))
        assert hasattr(torch, dtype_name), f"Unsupported dtype: {dtype_name}"
        return dtype_name

    def build_model(self, model_dir):
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

        if (
            self.custom_config is not None
            and hasattr(self.custom_config, "model")
            and hasattr(self.custom_config.model, "text_config")
        ):
            update_config_from_custom_config(model_config, self.custom_config.model.text_config)

        _tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model_config.pad_token_id = _tokenizer.pad_token_id
        model_config.eos_token_id = _tokenizer.eos_token_id

        model = Qwen3MoeModelForGeneration(model_config)
        hf_model = Qwen3MoeForCausalLM.from_pretrained(model_dir, config=model_config, trust_remote_code=True)
        checkpoint = hf_model.state_dict()

        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            if new_key == "lm_head.weight":
                new_key = "lm.lm_head.weight"
            elif not new_key.startswith("lm."):
                new_key = "lm." + new_key
            # HF: layers.*.mlp.gate.weight -> our nn.Linear lives at gate.linear
            if new_key.endswith(".mlp.gate.weight"):
                new_key = new_key[: -len("weight")] + "linear.weight"
            new_state_dict[new_key] = value

        miss_key, unexpected_key = load_state_dict_with_metadata(model, new_state_dict)

        if self.is_shared_lm_mode():
            model.prefill = model.lm
            model.decode = model.lm
        else:
            model.prefill = model.lm
            model.decode = copy.deepcopy(model.lm)

        del hf_model
        del checkpoint
        del new_state_dict

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
        input_names = ["input_token_ids", "position_ids", "attention_mask"]
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
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        num_kv_heads = cfg.num_key_value_heads
        max_kvcache_len = cfg.max_kvcache_len
        n_layers = cfg.num_hidden_layers
        if model_name in ("prefill", "lm"):
            seq_len = cfg.max_lm_input_len
            device = get_module_device(self.generated_model.prefill)
        elif model_name == "decode":
            seq_len = 1
            device = get_module_device(self.generated_model.decode)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        example_inputs.append(torch.ones(1, seq_len, device=device, dtype=torch.int32))
        example_inputs.append(torch.ones(1, seq_len, device=device, dtype=torch.int32))
        example_inputs.append(torch.randn(1, seq_len, max_kvcache_len, device=device, dtype=dtype))
        example_inputs.append(
            [
                torch.randn(1, max_kvcache_len, num_kv_heads, head_dim, device=device, dtype=dtype)
                for _ in range(2 * n_layers)
            ]
        )
        return example_inputs

    def _message_content_to_text(self, content):
        """Convert vlm_json content (list of {type, text/image}) to plain text for LLM."""
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
        # format: content can be [{"type": "text", "text": "..."}]
        normalized = []
        for msg in message:
            msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = self._message_content_to_text(content)
            normalized.append(msg)
        enable_thinking = getattr(self.custom_config.model, "enable_thinking", False)
        text = self.tokenizer.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    def output_postprocess(self, generated_ids):
        generated_text = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return generated_text

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
        if get_march() in ("nash-e", "nash-m"):
            return super().get_qconfig_setting(model_name)
        q_template = nashp_default_qconfig_template()

        module_name_config = {}
        for name in self.get_kvcache_names(model_name):
            if name.endswith("cache_k_fq") or name.endswith("cache_v_fq"):
                module_name_config[name] = {"output": qint8}

        module_name_config["embed_tokens"] = {"weight": qint8}

        q_template = q_template + [
            ModuleNameTemplate(module_name_config, freeze=True),
        ]

        q_template.append(
            SetDynamicQuantTemplate(
                op_kwargs={
                    nn.Linear: {"block_size": "full", "dim": -1},
                }
            )
        )
        return q_template
