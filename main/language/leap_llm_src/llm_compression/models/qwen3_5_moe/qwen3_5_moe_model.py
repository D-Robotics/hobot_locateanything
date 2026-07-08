import copy

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

try:
    from transformers import Qwen3_5MoeForConditionalGeneration
except ImportError:
    Qwen3_5MoeForConditionalGeneration = None

from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.nn import Matmul
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

from .model import Qwen3_5MoeTextModel, Qwen3_5MoeVisionModel
from .process_utils import generate_func

logger = get_logger(__name__)


class Qwen3_5MoeModelForGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5MoeVisionModel(config.vision_config)
        self.lm = Qwen3_5MoeTextModel(config.text_config)

    def get_image_feature(self, pixel_values, grid_thw=None):
        return self.visual(pixel_values, grid_thw=grid_thw)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_config(self):
        return self.config

    def generate(self, inputs, do_sample=False, chunk_prefill=False):
        return generate_func(
            self.config,
            self.get_input_embeddings(),
            self.visual,
            self.prefill,
            self.decode,
            inputs["input_ids"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["attention_mask"],
            do_sample=do_sample,
            chunk_prefill=chunk_prefill,
        )


@MODEL_REGISTRY
class Qwen3_5_Moe(BaseQModel):
    def __init__(self, model_dir, custom_config=None):
        super().__init__(model_dir, custom_config)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    def get_model_dtype(self):
        dtype_name = str(getattr(self.custom_config.model, "model_dtype", "float32"))
        assert hasattr(torch, dtype_name), f"Unsupported dtype: {dtype_name}"
        return dtype_name

    def build_model(self, model_dir):
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        if self.custom_config is not None and hasattr(self.custom_config, "model"):
            if hasattr(self.custom_config.model, "vision_config"):
                update_config_from_custom_config(model_config.vision_config, self.custom_config.model.vision_config)
            if hasattr(self.custom_config.model, "text_config"):
                update_config_from_custom_config(model_config.text_config, self.custom_config.model.text_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model_config.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        model = Qwen3_5MoeModelForGeneration(model_config)

        hf_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_dir,
            config=model_config,
            trust_remote_code=True,
        )
        checkpoint = hf_model.state_dict()

        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            if new_key.startswith("language_model."):
                new_key = "lm." + new_key[len("language_model.") :]
            if new_key == "lm_head.weight":
                new_key = "lm.lm_head.weight"
            if "visual.patch_embed.proj.weight" in new_key and value.dim() == 5:
                out_ch = value.shape[0]
                value = value.reshape(out_ch, -1).contiguous()
            new_state_dict[new_key] = value

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
        if model_name == "visual":
            return self.generated_model.visual.config
        cfg = getattr(self.generated_model, model_name, None)
        if cfg is None:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return cfg.config

    def get_model_input_output_name(self, model_name):
        if model_name == "visual":
            input_names = ["pixel_values"]
            output_names = ["image_features"]
        elif model_name in ("prefill", "decode", "lm"):
            cfg = self.get_generated_model_cfg(model_name)
            n_layers = cfg.num_hidden_layers
            input_names = [
                "input_embeddings",
                "position_ids",
                "attention_mask",
                "linear_attention_mask",
            ]
            k_cache_names = []
            v_cache_names = []
            conv_state_names = []
            recurrent_state_names = []
            for idx in range(n_layers):
                k_cache_names.append(f"in_key_cache_{idx}")
                v_cache_names.append(f"in_value_cache_{idx}")
                conv_state_names.append(f"in_conv_state_{idx}")
                recurrent_state_names.append(f"in_recurrent_state_{idx}")
            input_names = input_names + k_cache_names + v_cache_names + conv_state_names + recurrent_state_names
            output_names = [
                "output_logits",
                *[f"out_key_cache_{idx}" for idx in range(n_layers)],
                *[f"out_value_cache_{idx}" for idx in range(n_layers)],
                *[f"out_conv_state_{idx}" for idx in range(n_layers)],
                *[f"out_recurrent_state_{idx}" for idx in range(n_layers)],
            ]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return input_names, output_names

    def get_generated_model(self):
        return self.generated_model

    def get_model_trace_dummy_input(self, model_name):
        example_inputs = []
        dtype_name = self.get_model_dtype()
        dtype = getattr(torch, dtype_name)
        cfg = self.get_generated_model_cfg(model_name)

        if model_name == "visual":
            H = getattr(cfg, "image_height", 448)
            W = getattr(cfg, "image_width", 448)
            P = cfg.patch_size
            C = cfg.in_channels
            T = cfg.temporal_patch_size
            num_patches = (H // P) * (W // P)
            patch_flat_dim = C * T * P * P
            visual_device = get_module_device(self.generated_model.visual)
            example_inputs.append(torch.randn(1, num_patches, patch_flat_dim, device=visual_device, dtype=dtype))
        else:
            head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
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

            example_inputs.append(torch.randn(1, seq_len, cfg.hidden_size, device=device, dtype=dtype))
            example_inputs.append(torch.ones(3, 1, seq_len, device=device, dtype=torch.int32))
            example_inputs.append(torch.randn(1, 1, seq_len, max_kvcache_len, device=device, dtype=dtype))
            example_inputs.append(torch.ones(1, seq_len, device=device, dtype=torch.int32))

            # conv_dim is shared between full_attention and linear_attention layers
            conv_dim = (
                cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
                + cfg.linear_num_value_heads * cfg.linear_value_head_dim
            )
            caches_k = []
            caches_v = []
            caches_conv = []
            caches_recurrent = []
            for _ in range(n_layers):
                caches_k.append(
                    torch.randn(
                        1,
                        max_kvcache_len,
                        cfg.num_key_value_heads,
                        head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                caches_v.append(
                    torch.randn(
                        1,
                        max_kvcache_len,
                        cfg.num_key_value_heads,
                        head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                # Zero-tensor placeholders for unused linear-attn state slots.
                # layer_type is a static attribute; the linear-attn branch is
                # folded away at trace time, so these tensors are dead code.
                caches_conv.append(
                    torch.zeros(
                        1,
                        conv_dim,
                        cfg.linear_conv_kernel_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                caches_recurrent.append(
                    torch.zeros(
                        1,
                        cfg.linear_num_value_heads,
                        cfg.linear_key_head_dim,
                        cfg.linear_value_head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
            caches = caches_k + caches_v + caches_conv + caches_recurrent
            example_inputs.append(caches)
        return example_inputs

    def input_preprocess(self, message):
        from qwen_vl_utils import process_vision_info

        enable_thinking = getattr(self.custom_config.model, "enable_thinking", False)
        text = self.processor.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
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
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def get_kvcache_names(self, model_name):
        cfg = self.get_generated_model_cfg(model_name)
        names = []
        for idx, layer_type in enumerate(cfg.layer_types):
            if layer_type == "full_attention":
                names.extend(
                    [
                        f"layers.{idx}.self_attn.cache_k_fq",
                        f"layers.{idx}.self_attn.cache_v_fq",
                    ]
                )
            elif layer_type == "linear_attention":
                names.extend(
                    [
                        f"layers.{idx}.linear_attn.quant_conv_state",
                        f"layers.{idx}.linear_attn.quant_recurrent_state",
                    ]
                )
        return names

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
            module_name_config = {}

            for name in self.get_kvcache_names(model_name):
                module_name_config[name] = {"output": qint8}
            q_template = q_template + [
                ModuleNameTemplate(module_name_config, freeze=True),
            ]
            q_template.append(SetDynamicQuantTemplate(op_kwargs={nn.Linear: {"block_size": "full", "dim": -1}}))
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return q_template
