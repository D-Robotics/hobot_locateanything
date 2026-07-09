"""Eagle3 QModel Wrapper for llm_compression.

Wraps any BaseQModel with Eagle3 speculative decoding capability.
Delegates all interface methods to the base QModel, except for eagle3-specific parts.
"""

import torch
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    SetDynamicQuantTemplate,
)
from torch import nn

from llm_compression.models.base_qmodel import BaseQModel, nashp_default_qconfig_template
from llm_compression.registry_factory import MODEL_REGISTRY
from llm_compression.utils.device_manager import DeviceManager
from llm_compression.utils.logger import get_logger

from .eagle3_generate import Eagle3ModelForGeneration
from .model import Eagle3LlmModel
from .utils import HiddenStateCollector, get_llm_config

logger = get_logger(__name__)


@MODEL_REGISTRY
class Eagle3(BaseQModel):
    """Eagle3 speculative decoding wrapper.

    Wraps any registered base model with an Eagle3 draft model.
    Compatible with calib/compile/torch_eval without any modifications to those tools.

    YAML config example:
        model:
            model_name: Eagle3
            base_model_name: Qwen3          # or InternVL_2B, any registered model
            model_path: /path/to/base-model
            model_list: [prefill, decode, eagle3]  # VLM adds vision_model
            eagle3_config:
                eagle3_model_path: /path/to/eagle3-model
                top_k: 10
                depth: 6
                total_tokens: 60
                hidden_state_layers: [1, 14, 25]
    """

    def __init__(self, model_dir, custom_config=None):
        self.custom_config = custom_config

        # 1. Instantiate base QModel
        base_model_name = custom_config.model.base_model_name
        assert base_model_name == "Qwen3", f"Eagle3 currently only supports Qwen3 as base model, got: {base_model_name}"
        self.base_qmodel = MODEL_REGISTRY[base_model_name](model_dir, custom_config)

        # 2. Parse eagle3_config
        eagle3_config = custom_config.model.eagle3_config

        # 3. Load Eagle3 draft model
        eagle3_model_path = eagle3_config["eagle3_model_path"]
        dtype_name = str(getattr(custom_config.model, "model_dtype", "float32"))
        dtype = getattr(torch, dtype_name)
        self.eagle3_model = Eagle3LlmModel.from_pretrained(eagle3_model_path, dtype=dtype)

        # 4. Set up hidden state collectors and register hooks
        hidden_state_layers = list(eagle3_config["hidden_state_layers"])
        base_gen_model = self.base_qmodel.get_generated_model()

        self._prefill_collector = HiddenStateCollector()
        self._decode_collector = HiddenStateCollector()
        self._prefill_collector.register(base_gen_model.prefill.layers, hidden_state_layers)
        self._decode_collector.register(base_gen_model.decode.layers, hidden_state_layers)

        # 5. Compose the generation model
        self.generated_model = Eagle3ModelForGeneration(
            base_gen_model,
            self.eagle3_model,
            self._prefill_collector,
            self._decode_collector,
            eagle3_config,
        )

        # 6. Assign devices via DeviceManager
        self.device_manager = DeviceManager(self.generated_model, custom_config.model.model_list)

        # Copy tokenizer from base model
        if hasattr(self.base_qmodel, "tokenizer"):
            self.tokenizer = self.base_qmodel.tokenizer

    def build_model(self, model_dir):
        pass

    def get_generated_model(self):
        return self.generated_model

    def get_model_trace_dummy_input(self, model_part):
        if model_part == "eagle3":
            return self._eagle3_dummy_input()
        return self.base_qmodel.get_model_trace_dummy_input(model_part)

    def get_qconfig_setting(self, model_part):
        if model_part == "eagle3":
            return self._eagle3_qconfig()
        return self.base_qmodel.get_qconfig_setting(model_part)

    def get_model_input_output_name(self, model_part):
        if model_part == "eagle3":
            return self._eagle3_io_names()
        return self.base_qmodel.get_model_input_output_name(model_part)

    def get_kvcache_names(self, model_part):
        if model_part == "eagle3":
            return ["midlayer.self_attn.cache_k_fq", "midlayer.self_attn.cache_v_fq"]
        return self.base_qmodel.get_kvcache_names(model_part)

    def input_preprocess(self, message):
        return self.base_qmodel.input_preprocess(message)

    def output_postprocess(self, output):
        return self.base_qmodel.output_postprocess(output)

    def get_generated_model_cfg(self, model_name):
        if model_name == "eagle3":
            return self.eagle3_model.config
        return self.base_qmodel.get_generated_model_cfg(model_name)

    def is_shared_lm_mode(self):
        return self.base_qmodel.is_shared_lm_mode()

    def _eagle3_dummy_input(self):
        """Generate dummy inputs for tracing the Eagle3 model."""
        config = self.eagle3_model.config
        hidden_size = config["hidden_size"]
        num_kv_heads = config["num_key_value_heads"]
        head_dim = hidden_size // config["num_attention_heads"]
        top_k = self.custom_config.model.eagle3_config.top_k

        llm_cfg = get_llm_config(self.generated_model.config)
        max_kvcache_len = llm_cfg.max_kvcache_len

        dtype_name = str(getattr(self.custom_config.model, "model_dtype", "float32"))
        dtype = getattr(torch, dtype_name)
        device = next(self.eagle3_model.parameters()).device

        token_embeds = torch.randn(1, top_k, hidden_size, device=device, dtype=dtype)
        hidden_states = torch.randn(1, top_k, hidden_size * 3, device=device, dtype=dtype)
        position_ids = torch.arange(top_k, device=device, dtype=torch.int32).unsqueeze(0)
        mask = torch.randn(1, 1, top_k, max_kvcache_len, device=device, dtype=dtype)
        caches = [torch.randn(1, max_kvcache_len, num_kv_heads, head_dim, device=device, dtype=dtype) for _ in range(2)]

        return [token_embeds, hidden_states, position_ids, mask, caches]

    def _eagle3_qconfig(self):
        """Quantization config for the Eagle3 model."""
        q_template = nashp_default_qconfig_template()
        q_template.append(SetDynamicQuantTemplate(op_kwargs={nn.Linear: {"block_size": "full", "dim": -1}}))
        return q_template

    def _eagle3_io_names(self):
        """Input/output names for Eagle3 model compilation."""
        input_names = [
            "input_embeddings",
            "in_hidden_states",
            "position_ids",
            "attention_mask",
            "in_key_cache_0",
            "in_value_cache_0",
        ]
        output_names = ["output_logits", "out_key_cache_0", "out_value_cache_0", "out_hidden_states"]
        return input_names, output_names
