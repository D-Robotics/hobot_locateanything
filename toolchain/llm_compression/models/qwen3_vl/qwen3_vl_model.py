import copy
import types

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration as HFQwen3VL

from horizon_plugin_pytorch.dtype import qint16
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
from llm_compression.utils import AttentionManager
from llm_compression.utils.logger import get_logger

from .model import Qwen3VLTextModel, Qwen3VLVisionModel
from .process_utils import generate_func

logger = get_logger(__name__)


def _qwen3vl_decode_forward(self, input_embeddings, position_ids, attention_mask, caches=None):
    """Decode-only forward that excludes deepstack_visual_embeds from the signature.

    JIT trace and hbir export determine input nodes from the forward signature.
    Decode does not need deepstack inputs, so this wrapper hides the parameter
    to keep the traced graph clean. Without this, caches would be passed to
    deepstack_visual_embeds positionally, causing a matmul shape mismatch.
    """
    return Qwen3VLTextModel.forward(
        self,
        input_embeddings,
        position_ids,
        attention_mask,
        deepstack_visual_embeds=None,
        caches=caches,
    )


class Qwen3VLModelForGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.lm = Qwen3VLTextModel(config.text_config)

    def get_image_feature(self, pixel_values, grid_thw=None):
        return self.visual(pixel_values, grid_thw=grid_thw)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_config(self):
        return self.config

    def generate(self, inputs, do_sample=False, chunk_prefill=False):
        config = self.get_config()
        return generate_func(
            config,
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
class Qwen3_VL(BaseQModel):
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
                update_config_from_custom_config(
                    model_config.vision_config,
                    self.custom_config.model.vision_config,
                )
            if hasattr(self.custom_config.model, "text_config"):
                update_config_from_custom_config(
                    model_config.text_config,
                    self.custom_config.model.text_config,
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model_config.eos_token_id = self.tokenizer.eos_token_id
        model_config.vision_config.out_hidden_size = model_config.text_config.hidden_size

        model = Qwen3VLModelForGeneration(model_config)

        hf_model = HFQwen3VL.from_pretrained(model_dir, config=model_config, trust_remote_code=True)
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
            model.decode = copy.deepcopy(model.prefill)
            model.decode.forward = types.MethodType(_qwen3vl_decode_forward, model.decode)

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
        n_deepstack = len(getattr(self.generated_model.visual, "deepstack_visual_indexes", []))
        if model_name == "visual":
            input_names = ["pixel_values"]
            output_names = ["image_features"] + [f"deepstack_features_{i}" for i in range(n_deepstack)]
        elif model_name in ("prefill", "decode", "lm"):
            cfg = self.get_generated_model_cfg(model_name)
            n_layers = cfg.num_hidden_layers
            n_deepstack = len(getattr(self.generated_model.visual, "deepstack_visual_indexes", []))
            input_names = ["input_embeddings", "position_ids", "attention_mask"]
            if model_name == "prefill":
                deepstack_input_names = []
                for i in range(n_deepstack):
                    deepstack_input_names.append(f"deepstack_features_{i}")
                input_names = input_names + deepstack_input_names
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

    def setup_decode_model(self, decode_model):
        decode_model.forward = types.MethodType(_qwen3vl_decode_forward, decode_model)

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
        elif model_name in ("prefill", "decode", "lm"):
            hidden_size = cfg.hidden_size
            num_kv_heads = cfg.num_key_value_heads
            head_dim = getattr(cfg, "head_dim", hidden_size // cfg.num_attention_heads)
            max_kvcache_len = cfg.max_kvcache_len
            n_layers = cfg.num_hidden_layers
            if model_name in ("prefill", "lm"):
                seq_len = cfg.max_lm_input_len
                device = get_module_device(getattr(self.generated_model, model_name))
            else:  # decode
                seq_len = 1
                device = get_module_device(self.generated_model.decode)
            example_inputs.append(torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype))
            example_inputs.append(torch.ones(3, 1, seq_len, device=device, dtype=torch.int32))
            example_inputs.append(torch.randn(1, seq_len, max_kvcache_len, device=device, dtype=dtype))
            if model_name in ("prefill", "lm"):
                n_deepstack = len(getattr(self.generated_model.visual, "deepstack_visual_indexes", []))
                deepstack_embeds = [
                    torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype) for _ in range(n_deepstack)
                ]
                example_inputs.append(deepstack_embeds)
            example_inputs.append(
                [
                    torch.randn(1, max_kvcache_len, num_kv_heads, head_dim, device=device, dtype=dtype)
                    for _ in range(2 * n_layers)
                ]
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return example_inputs

    def input_preprocess(self, message):
        from qwen_vl_utils import process_vision_info

        enable_thinking = getattr(self.custom_config.model, "enable_thinking", False)
        text = self.processor.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
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
                    }
                )
            )
            return q_template
        elif model_name in ("prefill", "decode", "lm"):
            cfg = self.get_generated_model_cfg(model_name)
            n_layers = cfg.num_hidden_layers
            q_template = nashp_default_qconfig_template()

            module_name_config = {}
            # KV-cache fake-quant stubs: use get_kvcache_names to keep
            # naming consistent with sync_kvcache_scales and other logic.
            for name in self.get_kvcache_names(model_name):
                if name.endswith("cache_k_fq"):
                    module_name_config[name] = {"output": qint16}
                elif name.endswith("cache_v_fq"):
                    module_name_config[name] = {"output": qint8}

            # Attention matmul configs keep using per-layer patterns.
            use_flash_attention = AttentionManager.is_flash_attn()
            for i in range(n_layers):
                if use_flash_attention:
                    module_name_config[f"layers.{i}.self_attn.attention.qk_matmul"] = {"input": [qint8, qint16]}
                    module_name_config[f"layers.{i}.self_attn.attention.sv_matmul"] = {"input": [qint16, qint8]}
                else:
                    module_name_config[f"layers.{i}.self_attn._generated_matmul_0"] = {"input": [qint8, qint16]}
                    module_name_config[f"layers.{i}.self_attn._generated_matmul_1"] = {"input": [qint16, qint8]}

            q_template = q_template + [
                ModuleNameTemplate(module_name_config, freeze=True),
            ]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        q_template.append(
            SetDynamicQuantTemplate(
                op_kwargs={
                    nn.Linear: {"block_size": "full", "dim": -1},
                }
            )
        )
        return q_template
