import copy

import torch
from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.nn import Matmul
from horizon_plugin_pytorch.quantization.qconfig_setter import SetDynamicQuantTemplate
from torch import nn
from transformers import AutoConfig, AutoProcessor
from transformers import (
    Qwen2_5_VLForConditionalGeneration as hf_Qwen2_5_VLForConditionalGeneration,
)

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

from .model import Qwen2_5_VLTextModel, Qwen2_5_VLVisionModel
from .process_utils import generate_func

logger = get_logger(__name__)


class Qwen2_5_VLModelForGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen2_5_VLVisionModel(
            config.vision_config,
        )
        self.lm = Qwen2_5_VLTextModel(
            config.text_config,
        )
        self.rope_deltas = None

    def get_image_feature(self, pixel_values, image_grid_thw):
        return self.visual(pixel_values)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_rotary_emb(self):
        return self.lm.get_rotary_emb()

    def get_config(self):
        return self.config

    def generate(self, inputs, do_sample=False, chunk_prefill=False):
        input_embeddings = self.get_input_embeddings()
        config = self.get_config()
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        target_dtype = next(self.visual.parameters()).dtype
        pixel_values = pixel_values.to(dtype=target_dtype)
        image_grid_thw = inputs["image_grid_thw"]
        attention_mask = inputs["attention_mask"]
        generated_ids = generate_func(
            config,
            input_embeddings,
            self.visual,
            self.prefill,
            self.decode,
            input_ids,
            pixel_values,
            image_grid_thw,
            attention_mask,
            do_sample=do_sample,
            chunk_prefill=chunk_prefill,
        )
        return generated_ids


@MODEL_REGISTRY
class Qwen2_5_VL(BaseQModel):
    def __init__(self, model_dir, custom_config=None):
        super().__init__(model_dir, custom_config)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    def get_model_dtype(self):
        dtype_name = str(getattr(self.custom_config.model, "model_dtype", "float32"))
        assert hasattr(torch, dtype_name), f"Unsupported dtype: {dtype_name}"
        return dtype_name

    def build_model(self, model_dir):
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # Update config from custom_config if provided
        if self.custom_config is not None and hasattr(self.custom_config, "model"):
            if hasattr(self.custom_config.model, "vision_config"):
                update_config_from_custom_config(model_config.vision_config, self.custom_config.model.vision_config)
            if hasattr(self.custom_config.model, "text_config"):
                update_config_from_custom_config(model_config.text_config, self.custom_config.model.text_config)

        # Qwen2_5_VLConfig has eos_token_id in text_config but not at top level.
        # Copy it up so process_utils.generate_func can access config.eos_token_id
        # directly (consistent with other models).
        model_config.eos_token_id = model_config.text_config.eos_token_id

        # Create custom model
        model = Qwen2_5_VLModelForGeneration(model_config)
        hf_model = hf_Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, config=model_config, trust_remote_code=True
        )
        checkpoint = hf_model.state_dict()
        mapping = {
            "model.visual.merger.mlp.0.weight": "model.visual.merger.mlp.proj0.weight",  # noqa: E501
            "model.visual.merger.mlp.0.bias": "model.visual.merger.mlp.proj0.bias",
            "model.visual.merger.mlp.2.weight": "model.visual.merger.mlp.proj1.weight",  # noqa: E501
            "model.visual.merger.mlp.2.bias": "model.visual.merger.mlp.proj1.bias",
            "lm_head.weight": "model.lm.lm_head.weight",
        }
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key in mapping:
                key = mapping[key]
            key = key[len("model.") :]
            if key.startswith("language_model."):
                key = "lm." + key[len("language_model.") :]
            new_state_dict[key] = value
        miss_key, unexpected_key = load_state_dict_with_metadata(model, new_state_dict)

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
        if model_name == "prefill" or model_name == "decode" or model_name == "lm":
            cfg = self.get_generated_model_cfg(model_name)
            n_layers = cfg.num_hidden_layers
        if model_name == "visual":
            input_names = ["pixel_values"]
            output_names = ["image_features"]
        elif model_name == "prefill" or model_name == "decode" or model_name == "lm":
            input_names = ["input_embeddings", "position_ids", "attention_mask"]
            kcache_names = []
            vcache_names = []
            for idx in range(n_layers):
                kcache_names.append(f"in_key_cache_{idx}")
                vcache_names.append(f"in_value_cache_{idx}")
            input_names = input_names + kcache_names + vcache_names
            output_names = [
                "output_logits",
                *[f"out_key_cache_{idx}" for idx in range(n_layers)],
                *[f"out_value_cache_{idx}" for idx in range(n_layers)],
            ]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return input_names, output_names

    def get_generated_model(self):
        return self.generated_model

    def get_model_trace_dummy_input(self, model_name):
        example_inputs = []
        cfg = self.get_generated_model_cfg(model_name)
        dtype_name = self.get_model_dtype()
        dtype = getattr(torch, dtype_name)
        if model_name == "visual":
            H, W, P, C = cfg.image_height, cfg.image_width, cfg.patch_size, cfg.in_channels
            num_patches = (H // P) * (W // P)
            patch_flat_dim = C * P * P
            visual_device = get_module_device(self.generated_model.visual)
            example_inputs.append(torch.randn(1, num_patches, patch_flat_dim, device=visual_device, dtype=dtype))

        elif model_name in ("prefill", "lm"):
            hidden_size = cfg.hidden_size
            num_kv_heads = cfg.num_key_value_heads
            head_dim = hidden_size // cfg.num_attention_heads
            max_kvcache_len = cfg.max_kvcache_len
            n_layers = cfg.num_hidden_layers
            seq_len = cfg.max_lm_input_len
            mrope_streams = getattr(cfg, "mrope_streams", 3)
            prefill_device = get_module_device(self.generated_model.prefill)
            example_inputs.append(torch.randn(1, seq_len, hidden_size, device=prefill_device, dtype=dtype))
            example_inputs.append(torch.ones(1, mrope_streams, seq_len, device=prefill_device, dtype=torch.int32))
            example_inputs.append(torch.randn(1, seq_len, max_kvcache_len, device=prefill_device, dtype=dtype))
            example_inputs.append(
                [
                    torch.randn(1, max_kvcache_len, num_kv_heads, head_dim, device=prefill_device, dtype=dtype)
                    for _ in range(2 * n_layers)
                ]
            )

        elif model_name == "decode":
            hidden_size = cfg.hidden_size
            num_kv_heads = cfg.num_key_value_heads
            head_dim = hidden_size // cfg.num_attention_heads
            max_kvcache_len = cfg.max_kvcache_len
            n_layers = cfg.num_hidden_layers
            decode_device = get_module_device(self.generated_model.decode)
            example_inputs.append(torch.randn(1, 1, hidden_size, device=decode_device, dtype=dtype))
            example_inputs.append(torch.ones(1, 1, 1, device=decode_device, dtype=torch.int32))
            example_inputs.append(torch.randn(1, max_kvcache_len, device=decode_device, dtype=dtype))
            example_inputs.append(
                [
                    torch.randn(1, max_kvcache_len, num_kv_heads, head_dim, device=decode_device, dtype=dtype)
                    for _ in range(2 * n_layers)
                ]
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        return example_inputs

    def input_preprocess(self, message):
        from qwen_vl_utils import process_vision_info

        """
        Convert a message into a format compatible with model.generate for qwen_vl.
        """
        text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        if hasattr(self.custom_config.model, "vision_config"):
            vc = self.custom_config.model.vision_config
            image_height = getattr(vc, "image_height", None)
            image_width = getattr(vc, "image_width", None)
        for msg in message:
            contents = msg.get("content")
            if not isinstance(contents, list):
                continue
            for ele in contents:
                if ele.get("type") == "image":
                    ele["resized_width"] = image_width
                    ele["resized_height"] = image_height
        images, videos = process_vision_info([message])
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def output_postprocess(self, generated_ids):
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return generated_text

    def get_kvcache_names(self, model_name):
        """Return the module names of KV-cache fake-quant stubs (cache_k_fq / cache_v_fq).

        These names serve two purposes:
        - In get_qconfig_setting: configure their output dtype as int8.
        - In sync_kvcache_scales: locate the activation_post_process submodule
          and unify scales between prefill and decode after calibration.
        """
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
        if model_name == "visual":
            q_template = nashp_default_qconfig_template()
            q_template.append(
                SetDynamicQuantTemplate(
                    op_kwargs={
                        nn.Linear: {"block_size": "full", "dim": -1},
                        Matmul: [
                            {"block_size": "full", "dim": -1},
                            {"block_size": "full", "dim": -2},
                        ],
                    }
                )
            )
        elif model_name in ("prefill", "decode", "lm"):
            q_template = nashp_default_qconfig_template()
            output_kvint8_list = self.get_kvcache_names(model_name)
            q_template = q_template + [
                SetDynamicQuantTemplate(op_kwargs={nn.Linear: {"block_size": "full", "dim": -1}}),
                ModuleNameTemplate(
                    {m: {"output": qint8} for m in output_kvint8_list},
                    freeze=True,
                ),
            ]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return q_template
