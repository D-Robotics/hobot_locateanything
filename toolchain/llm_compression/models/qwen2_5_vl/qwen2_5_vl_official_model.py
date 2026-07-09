# Warning: This model try to use transformers official model especially by generated API.
# NOTE: this model is still under experimental stage and not supported yet

import torch
from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.quantization.qconfig_setter import SetDynamicQuantTemplate
from torch import nn
from transformers import AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration as hf_Qwen2_5_VLForConditionalGeneration

from llm_compression.models.base_qmodel import (
    BaseQModel,
    ModuleNameTemplate,
    load_state_dict_with_metadata,
    nashp_default_qconfig_template,
    qint8,
    update_config_from_custom_config,
)
from llm_compression.registry_factory import MODEL_REGISTRY
from llm_compression.utils.logger import get_logger

from .model import Qwen2_5_VLTextModel, Qwen2_5_VLVisionModel

logger = get_logger(__name__)


@MODEL_REGISTRY
class Qwen2_5_VL_OFFICIAL(BaseQModel):
    def __init__(self, model_dir, custom_config=None):
        super().__init__(model_dir, custom_config)

    def build_model(self, model_dir):
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # Update config from custom_config if provided
        if self.custom_config is not None and hasattr(self.custom_config, "model"):
            if hasattr(self.custom_config.model, "vision_config"):
                update_config_from_custom_config(model_config.vision_config, self.custom_config.model.vision_config)
            if hasattr(self.custom_config.model, "text_config"):
                update_config_from_custom_config(model_config.text_config, self.custom_config.model.text_config)
        visual = Qwen2_5_VLVisionModel(
            model_config.vision_config,
        )
        language_model = Qwen2_5_VLTextModel(
            model_config.text_config,
        )

        hf_model = hf_Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, config=model_config, trust_remote_code=True
        )
        checkpoint = hf_model.state_dict()
        mapping = {
            "model.visual.merger.mlp.0.weight": "model.visual.merger.mlp.proj0.weight",
            "model.visual.merger.mlp.0.bias": "model.visual.merger.mlp.proj0.bias",
            "model.visual.merger.mlp.2.weight": "model.visual.merger.mlp.proj1.weight",
            "model.visual.merger.mlp.2.bias": "model.visual.merger.mlp.proj1.bias",
            "lm_head.weight": "model.language_model.lm_head.weight",
        }
        visual_state_dict = {}
        language_model_state_dict = {}
        for key, value in checkpoint.items():
            if key in mapping:
                key = mapping[key]
            key = key[len("model.") :]
            new_key = key
            if new_key.startswith("visual."):
                visual_state_dict[new_key[len("visual.") :]] = value
            if new_key.startswith("language_model."):
                language_model_state_dict[new_key[len("language_model.") :]] = value
        miss_key, unexpected_key = load_state_dict_with_metadata(visual, visual_state_dict)
        logger.info(f"miss_key: {miss_key}")
        logger.info(f"unexpected_key: {unexpected_key}")
        miss_key, unexpected_key = load_state_dict_with_metadata(language_model, language_model_state_dict)
        logger.info(f"miss_key: {miss_key}")
        logger.info(f"unexpected_key: {unexpected_key}")
        hf_model.model.visual = visual
        hf_model.model.language_model = language_model
        return hf_model

    def get_generated_model(self):
        return self.genreated_model

    def get_model_trace_dummy_input(self, model_name):
        """Get dummy input for model tracing.

        Note: For OFFICIAL model, we use the same dummy inputs as the custom model
        since the underlying model structure is the same.
        """
        if model_name == "visual":
            example_inputs = [torch.randn(1, 2040, 588)]
        elif model_name == "prefill":
            example_inputs = [
                torch.randn(1, 1024, 2048),
                torch.ones(1, 3, 1024).to(torch.int32),
                torch.randn(1, 1024, 4096),
                [torch.randn(1, 4096, 2, 128) for _ in range(72)],
            ]
        elif model_name == "decode":
            example_inputs = [
                torch.randn(1, 1, 2048),
                torch.ones(1, 1, 1).to(torch.int32),
                torch.randn(1, 4096),
                [torch.randn(1, 4096, 2, 128) for _ in range(72)],
            ]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return example_inputs

    def get_qconfig_setting(self, model_name):
        if get_march() in ("nash-e", "nash-m"):
            return super().get_qconfig_setting(model_name)
        if model_name == "visual":
            q_template = nashp_default_qconfig_template()
        elif model_name == "prefill" or model_name == "decode":
            q_template = nashp_default_qconfig_template()
            output_int8_list = []
            for i in range(36):
                output_int8_list.append(f"layers.{i}.self_attn.cache_k_fq")
                output_int8_list.append(f"layers.{i}.self_attn.cache_v_fq")
            q_template = q_template + [
                ModuleNameTemplate(
                    {m: {"output": qint8} for m in output_int8_list},
                    freeze=True,
                ),
            ]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        q_template.append(SetDynamicQuantTemplate(op_kwargs={nn.Linear: {"block_size": "full", "dim": -1}}))
        return q_template
