import copy

import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoProcessor
from transformers.models.auto import AutoModelForCausalLM

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

from .model import InternLM2Model, InternVisionModel
from .process_utils import generate_func

logger = get_logger(__name__)


def split_wqkv(wqkv, num_heads, num_kv_heads):
    """Split fused wqkv weight into separate q, k, v weights.

    InternLM2 uses fused wqkv with interleaved layout:
    [q_group0, k_group0, v_group0, q_group1, k_group1, v_group1, ...]
    Each group has (num_heads // num_kv_heads) q heads + 1 k head + 1 v head.
    """
    size, hidden_dim = wqkv.shape
    head_dim = hidden_dim // num_heads
    all_heads_num = num_heads + 2 * num_kv_heads
    assert head_dim * all_heads_num == size, "wqkv output size invalid"
    wqkv = wqkv.view(all_heads_num, head_dim, hidden_dim)
    groups = wqkv.chunk(num_kv_heads, dim=0)

    q_tensors, k_tensors, v_tensors = [], [], []
    q_per_group = num_heads // num_kv_heads
    for group in groups:
        q_tensors.append(group[:q_per_group])
        k_tensors.append(group[q_per_group : q_per_group + 1])
        v_tensors.append(group[-1])

    wq = torch.cat(q_tensors, dim=0).view(num_heads * head_dim, hidden_dim)
    wk = torch.cat(k_tensors, dim=0).view(num_kv_heads * head_dim, hidden_dim)
    wv = torch.cat(v_tensors, dim=0).view(num_kv_heads * head_dim, hidden_dim)
    return wq, wk, wv


class InternVL2BModelForGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = InternVisionModel(
            config.vision_config,
            config.downsample_ratio,
            config.llm_config.hidden_size,
        )
        self.lm = InternLM2Model(config.llm_config)

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
class InternVL_2B(BaseQModel):
    def __init__(self, model_dir, custom_config=None):
        super().__init__(model_dir, custom_config)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = self.processor
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.generated_model.config.img_context_token_id = img_context_token_id
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.generated_model.config.llm_config.eos_token_id = im_end_id

    def get_model_dtype(self):
        dtype_name = str(getattr(self.custom_config.model, "model_dtype", "float32"))
        assert hasattr(torch, dtype_name), f"Unsupported dtype: {dtype_name}"
        return dtype_name

    def build_model(self, model_dir):
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        original_image_size = model_config.vision_config.image_size
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

        model = InternVL2BModelForGeneration(model_config)

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            trust_remote_code=True,
        )
        checkpoint = hf_model.state_dict()

        # Split VIT fused qkv weights into separate q/k/v proj
        vit_n_layers = model_config.vision_config.num_hidden_layers
        vit_prefix = "vision_model.encoder.layers."
        for layer_id in range(vit_n_layers):
            qkv_w_key = f"{vit_prefix}{layer_id}.attn.qkv.weight"
            qkv_b_key = f"{vit_prefix}{layer_id}.attn.qkv.bias"
            if qkv_w_key in checkpoint:
                qkv_w = checkpoint[qkv_w_key]
                embed_dim = qkv_w.shape[0] // 3
                checkpoint[f"{vit_prefix}{layer_id}.attn.q_proj.weight"] = qkv_w[:embed_dim]
                checkpoint[f"{vit_prefix}{layer_id}.attn.k_proj.weight"] = qkv_w[embed_dim : 2 * embed_dim]
                checkpoint[f"{vit_prefix}{layer_id}.attn.v_proj.weight"] = qkv_w[2 * embed_dim :]
                checkpoint.pop(qkv_w_key)
            if qkv_b_key in checkpoint:
                qkv_b = checkpoint[qkv_b_key]
                embed_dim = qkv_b.shape[0] // 3
                checkpoint[f"{vit_prefix}{layer_id}.attn.q_proj.bias"] = qkv_b[:embed_dim]
                checkpoint[f"{vit_prefix}{layer_id}.attn.k_proj.bias"] = qkv_b[embed_dim : 2 * embed_dim]
                checkpoint[f"{vit_prefix}{layer_id}.attn.v_proj.bias"] = qkv_b[2 * embed_dim :]
                checkpoint.pop(qkv_b_key)

        # InternLM2 uses fused wqkv; split into q/k/v before mapping
        num_heads = model_config.llm_config.num_attention_heads
        num_kv_heads = model_config.llm_config.num_key_value_heads
        n_layers = model_config.llm_config.num_hidden_layers
        prefix = "language_model.model.layers."
        for layer_id in range(n_layers):
            wqkv_key = f"{prefix}{layer_id}.attention.wqkv.weight"
            if wqkv_key in checkpoint:
                wq, wk, wv = split_wqkv(checkpoint[wqkv_key], num_heads, num_kv_heads)
                checkpoint[f"{prefix}{layer_id}.attention.q_proj.weight"] = wq
                checkpoint[f"{prefix}{layer_id}.attention.k_proj.weight"] = wk
                checkpoint[f"{prefix}{layer_id}.attention.v_proj.weight"] = wv
                checkpoint.pop(wqkv_key)

        mlp1_mapping = {
            "mlp1.0.weight": "vision_model.mlp1.norm.weight",
            "mlp1.0.bias": "vision_model.mlp1.norm.bias",
            "mlp1.1.weight": "vision_model.mlp1.fc1.weight",
            "mlp1.1.bias": "vision_model.mlp1.fc1.bias",
            "mlp1.3.weight": "vision_model.mlp1.fc2.weight",
            "mlp1.3.bias": "vision_model.mlp1.fc2.bias",
        }
        attn_sub_mapping = {
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.q_proj.weight": "self_attn.q_proj.weight",
            "attention.k_proj.weight": "self_attn.k_proj.weight",
            "attention.v_proj.weight": "self_attn.v_proj.weight",
            "feed_forward.w1.weight": "mlp.gate_proj.weight",
            "feed_forward.w3.weight": "mlp.up_proj.weight",
            "feed_forward.w2.weight": "mlp.down_proj.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key in mlp1_mapping:
                new_state_dict[mlp1_mapping[key]] = value
            elif key.startswith("vision_model.encoder."):
                new_state_dict["vision_model." + key[len("vision_model.encoder.") :]] = value
            elif key.startswith("vision_model."):
                new_state_dict["vision_model." + key[len("vision_model.") :]] = value
            elif key == "language_model.model.tok_embeddings.weight":
                new_state_dict["lm.embed_tokens.weight"] = value
            elif key == "language_model.model.norm.weight":
                new_state_dict["lm.norm.weight"] = value
            elif key == "language_model.output.weight":
                new_state_dict["lm.lm_head.weight"] = value
            elif key.startswith("language_model.model.layers."):
                rest = key[len("language_model.model.layers.") :]
                layer_id, _, sub = rest.partition(".")
                mapped_sub = attn_sub_mapping.get(sub)
                if mapped_sub:
                    new_state_dict[f"lm.layers.{layer_id}.{mapped_sub}"] = value

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
                input_names.append(f"in_key_cache_{idx}")
            for idx in range(n_layers):
                input_names.append(f"in_value_cache_{idx}")
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
            H = W = cfg.image_size
            vision_model_device = get_module_device(self.generated_model.vision_model)
            example_inputs.append(torch.randn(1, 3, H, W, device=vision_model_device, dtype=dtype))
        else:
            hidden_size = cfg.hidden_size
            num_kv_heads = cfg.num_key_value_heads
            head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            max_kvcache_len = cfg.max_kvcache_len
            n_layers = cfg.num_hidden_layers
            model_device = get_module_device(getattr(self.generated_model, model_name))
            if model_name in ("prefill", "lm"):
                seq_len = cfg.max_lm_input_len
                position_ids = torch.arange(seq_len, device=model_device, dtype=torch.int32).unsqueeze(0)
            else:  # decode
                seq_len = 1
                position_ids = torch.tensor([[max_kvcache_len - 1]], device=model_device, dtype=torch.int32)
            example_inputs.append(torch.randn(1, seq_len, hidden_size, device=model_device, dtype=dtype))
            example_inputs.append(position_ids)
            example_inputs.append(torch.randn(1, seq_len, max_kvcache_len, device=model_device, dtype=dtype))
            example_inputs.append(
                [
                    torch.randn(1, max_kvcache_len, num_kv_heads, head_dim, device=model_device, dtype=dtype)
                    for _ in range(2 * n_layers)
                ]
            )
        return example_inputs

    def input_preprocess(self, message):
        """Convert a message into model.generate compatible inputs for InternVL 2B."""
        contents = message[0]["content"]
        text = ""
        img_path = None

        for ele in contents:
            if ele.get("type") == "image":
                img_path = ele["image"]
            if ele.get("type") == "text":
                text += ele.get("text", "")

        if img_path is not None and img_path.startswith("file://"):
            img_path = img_path[len("file://") :]

        if img_path is not None and "<image>" not in text:
            text = "<image>\n" + text

        config = self.generated_model.config
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        downsample_ratio = config.downsample_ratio
        num_image_tokens = int((image_size // patch_size * downsample_ratio) ** 2)

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        pixel_values = None
        if img_path is not None:
            image = Image.open(img_path).convert("RGB")
            pixel_values = transform(image).unsqueeze(0)
            image_tokens = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"
            text = text.replace("<image>", image_tokens, 1)

        # Build prompt matching HF model.chat MPT template exactly
        tokenizer = self.processor
        config = self.generated_model.config
        template_name = getattr(config, "template", "internlm2-chat")

        if template_name == "internvl2_5":
            system_message = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
            sep = "<|im_end|>\n"
        else:
            system_message = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"
            sep = "<|im_end|>"

        query = (
            f"<|im_start|>system\n{system_message}{sep}"
            f"<|im_start|>user\n{text}{sep}"
            f"<|im_start|>assistant\n"
        )
        toks = tokenizer(query, return_tensors="pt")
        inputs = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
        return inputs

    def output_postprocess(self, generated_ids):
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

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
            q_template = nashp_default_qconfig_template()

            # cache_k_fq / cache_v_fq -> qint8
            output_int8_list = self.get_kvcache_names(model_name)
            config_mapping = {m: {"output": qint8} for m in output_int8_list}

            q_template = q_template + [
                SetDynamicQuantTemplate(op_kwargs={nn.Linear: {"block_size": "full", "dim": -1}}),
                ModuleNameTemplate(config_mapping, freeze=True),
            ]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        return q_template
