import copy

import torch
import torchvision.transforms as T
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.nn import Matmul
from horizon_plugin_pytorch.quantization.qconfig_setter import SetDynamicQuantTemplate
from PIL import Image
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoProcessor
from transformers.models.auto import AutoModelForCausalLM

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

from .model import InternVisionModel, Qwen3Model
from .process_utils import generate_func

logger = get_logger(__name__)


class InternVL3_5ModelForGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = InternVisionModel(
            config.vision_config,
            config.downsample_ratio,
            config.llm_config.hidden_size,
        )
        self.lm = Qwen3Model(config.llm_config)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_config(self):
        return self.config

    def generate(self, inputs, do_sample=False, chunk_prefill=False):
        input_embeddings = self.get_input_embeddings()
        config = self.get_config()
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs["attention_mask"]

        if pixel_values is not None:
            target_dtype = next(self.vision_model.parameters()).dtype
            pixel_values = pixel_values.to(dtype=target_dtype)

        generated_ids = generate_func(
            config,
            input_embeddings,
            self.vision_model,
            self.prefill,
            self.decode,
            input_ids,
            pixel_values,
            attention_mask,
            do_sample=do_sample,
            chunk_prefill=chunk_prefill,
        )
        return generated_ids


@MODEL_REGISTRY
class InternVL3_5(BaseQModel):
    def __init__(self, model_dir, custom_config=None):
        super().__init__(model_dir, custom_config)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = self.processor
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.generated_model.config.img_context_token_id = img_context_token_id

    def get_model_dtype(self):
        dtype_name = str(getattr(self.custom_config.model, "model_dtype", "float32"))
        assert hasattr(torch, dtype_name), f"Unsupported dtype: {dtype_name}"
        return dtype_name

    def build_model(self, model_dir):
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        original_image_size = model_config.vision_config.image_size
        # Update config from custom_config if provided
        if self.custom_config is not None and hasattr(self.custom_config, "model"):
            if hasattr(self.custom_config.model, "vision_config"):
                update_config_from_custom_config(
                    model_config.vision_config,
                    self.custom_config.model.vision_config,
                )
            if hasattr(self.custom_config.model, "llm_config"):
                update_config_from_custom_config(
                    model_config.llm_config,
                    self.custom_config.model.llm_config,
                )

        assert model_config.vision_config.image_size == original_image_size, (
            f"InternVL vision encoder requires image_size={original_image_size} "
            f"(from pretrained checkpoint), got {model_config.vision_config.image_size}. "
            f"image_size is determined by the pretrained position_embedding and cannot be changed."
        )

        # Create custom model
        model = InternVL3_5ModelForGeneration(model_config)

        # Load pretrained weights
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            trust_remote_code=True,
        )
        checkpoint = hf_model.state_dict()

        # Weight key mapping
        mapping = {
            "mlp1.0.weight": "vision_model.mlp1.norm.weight",
            "mlp1.0.bias": "vision_model.mlp1.norm.bias",
            "mlp1.1.weight": "vision_model.mlp1.fc1.weight",
            "mlp1.1.bias": "vision_model.mlp1.fc1.bias",
            "mlp1.3.weight": "vision_model.mlp1.fc2.weight",
            "mlp1.3.bias": "vision_model.mlp1.fc2.bias",
        }
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key in mapping:
                new_key = mapping[key]
            elif key.startswith("vision_model.encoder."):
                new_key = "vision_model." + key[len("vision_model.encoder.") :]
            elif key.startswith("vision_model."):
                new_key = "vision_model." + key[len("vision_model.") :]
            elif key.startswith("language_model.model."):
                new_key = "lm." + key[len("language_model.model.") :]
            elif key.startswith("language_model.lm_head."):
                new_key = "lm.lm_head." + key[len("language_model.lm_head.") :]
            else:
                continue
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
        if model_name == "vision_model":
            return self.generated_model.vision_model.config
        cfg = getattr(self.generated_model, model_name, None)
        if cfg is None:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return cfg.config

    def get_model_input_output_name(self, model_name):
        if model_name == "vision_model":
            input_names = ["pixel_values"]
            output_names = ["image_features"]
        elif model_name in ("prefill", "decode", "lm"):
            cfg = self.get_generated_model_cfg(model_name)
            n_layers = cfg.num_hidden_layers
            input_names = ["input_embeddings", "position_ids", "attention_mask"]
            for idx in range(n_layers):
                input_names = input_names + [f"in_key_cache_{idx}"]
            for idx in range(n_layers):
                input_names = input_names + [f"in_value_cache_{idx}"]
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

        if model_name == "vision_model":
            H = cfg.image_size
            W = cfg.image_size
            vision_model_device = get_module_device(self.generated_model.vision_model)
            example_inputs.append(torch.randn(1, 3, H, W, device=vision_model_device, dtype=dtype))
        else:
            hidden_size = cfg.hidden_size
            num_kv_heads = cfg.num_key_value_heads
            head_dim = cfg.head_dim
            max_kvcache_len = cfg.max_kvcache_len
            n_layers = cfg.num_hidden_layers
            model_device = get_module_device(getattr(self.generated_model, model_name))
            if model_name in ("prefill", "lm"):
                seq_len = cfg.max_lm_input_len
                position_ids = torch.arange(seq_len, device=model_device, dtype=torch.int32).unsqueeze(0)
            else:  # decode
                seq_len = 1
                position_ids = torch.tensor(
                    [[max_kvcache_len - 1]],
                    device=model_device,
                    dtype=torch.int32,
                )
            # inputs_embeds
            example_inputs.append(torch.randn(1, seq_len, hidden_size, device=model_device, dtype=dtype))
            # position_ids
            example_inputs.append(position_ids)
            # attention_mask
            example_inputs.append(torch.randn(1, seq_len, max_kvcache_len, device=model_device, dtype=dtype))
            # caches (keys + values)
            example_inputs.append(
                [
                    torch.randn(
                        1,
                        max_kvcache_len,
                        num_kv_heads,
                        head_dim,
                        device=model_device,
                        dtype=dtype,
                    )
                    for _ in range(2 * n_layers)
                ]
            )

        return example_inputs

    def input_preprocess(self, message):
        """
        Convert a message into a format compatible with model.generate for InternVL3.5.
        """
        # Step 1: Extract image paths and text from message
        contents = message[0]["content"]
        text = ""

        for ele in contents:
            if ele.get("type") == "image":
                img_path = ele["image"]
            if ele.get("type") == "text":
                text += ele.get("text", "")

        # Remove file:// prefix if present
        if img_path.startswith("file://"):
            img_path = img_path[len("file://") :]

        # Ensure <image> placeholder is present when there's an image
        if "<image>" not in text:
            text = "<image>\n" + text

        # Step 2: Load and process image, build image token string
        config = self.generated_model.config
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        downsample_ratio = config.downsample_ratio

        # Number of visual tokens after ViT + pixel_shuffle
        num_image_tokens = int((image_size // patch_size * downsample_ratio) ** 2)

        # InternVL standard image transform
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        image = Image.open(img_path).convert("RGB")

        # Single tile: resize to (image_size, image_size) via transform
        # pixel_values shape: (1, 3, image_size, image_size)
        pixel_values = transform(image).unsqueeze(0)

        image_tokens = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"
        text = text.replace("<image>", image_tokens, 1)

        # Step 3: Apply chat template and tokenize
        tokenizer = self.processor
        system_message = (
            "你是书生·万象，英文名是InternVL，"
            "是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
        )
        chat_message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]
        chat_text = tokenizer.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=True)
        toks = tokenizer(chat_text, return_tensors="pt", padding=True)

        inputs = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }
        inputs["pixel_values"] = pixel_values

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
        if model_name == "vision_model":
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
            cfg = self.get_generated_model_cfg(model_name)
            n_layers = cfg.num_hidden_layers
            q_template = nashp_default_qconfig_template()
            output_int8_list = self.get_kvcache_names(model_name)
            config_mapping = {m: {"output": qint8} for m in output_int8_list}
            config_mapping["layers.0.self_attn.cache_k_fq"] = {"output": qint16}
            for i in range(n_layers):
                config_mapping[f"layers.{i}.self_attn._generated_matmul_0"] = {"input": [qint8, "del"]}
                if model_name in ("prefill", "lm"):
                    config_mapping[f"layers.{i}.self_attn._generated_matmul_1"] = {"input": [qint8, "del"]}
                else:
                    config_mapping[f"layers.{i}.self_attn._generated_matmul_1"] = {"input": [qint16, "del"]}
            config_mapping["layers.0.self_attn._generated_matmul_0"] = {"input": [qint8, qint16]}
            q_template = q_template + [
                SetDynamicQuantTemplate(op_kwargs={nn.Linear: {"block_size": "full", "dim": -1}}),
                ModuleNameTemplate(
                    config_mapping,
                    freeze=True,
                ),
            ]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return q_template
