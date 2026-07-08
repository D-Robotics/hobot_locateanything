import gc
import os
from pathlib import Path
from typing import Optional

import torch
from qwen_vl_utils.vision_process import process_vision_info
from transformers import AutoProcessor

from leap_llm.apis.calibration.data_loader import load_message_data, load_tsv_data
from leap_llm.models.qwen2_5_vl.model import Qwen2_5_VL, save_model_checkpoint
from leap_llm.nn.utils import (
    standard_lm_name,
    standard_token_embeddings_name,
    standard_vit_name,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


IMAGE_WIDTH = 448
IMAGE_HEIGHT = 252
VIDEO_WIDTH = 448
VIDEO_HEIGHT = 252


def text_model_forward_multi_gpu(
    text_model,
    inputs_embeds,
    position_ids,
    attention_mask,
    caches,
):
    """
    Multi-GPU version of text_model forward, transfer hidden_states between GPUs
    """
    if (
        not hasattr(text_model, "_multi_gpu_devices")
        or text_model._multi_gpu_devices is None
    ):
        # Single GPU mode, use original forward
        return text_model.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            caches=caches,
        )

    # Multi-GPU mode
    multi_gpu_devices = text_model._multi_gpu_devices
    primary_device = multi_gpu_devices[0]

    # Ensure inputs are on first GPU
    inputs_embeds = inputs_embeds.to(device=primary_device)
    position_ids = position_ids.to(device=primary_device)
    attention_mask = attention_mask.to(device=primary_device)

    # Ensure cache_cos and cache_sin are on primary device
    if hasattr(text_model, "cache_cos"):
        text_model.cache_cos = text_model.cache_cos.to(device=primary_device)
    if hasattr(text_model, "cache_sin"):
        text_model.cache_sin = text_model.cache_sin.to(device=primary_device)

    # Prepare position embeddings (compute on first GPU)
    dim = position_ids.shape[1]
    if dim > 1:
        split_cache_cos = text_model.cache_cos.split(text_model.mrope_section, dim=-1)
        split_cache_sin = text_model.cache_sin.split(text_model.mrope_section, dim=-1)
        split_position_ids = position_ids.split([1, 1, 1], dim=1)
        used_cos = []
        used_sin = []
        for (i, cos), sin in zip(enumerate(split_cache_cos), split_cache_sin):
            cur_position_ids = split_position_ids[i % 3]
            cos = cos.contiguous()
            used_cos.append(cos[cur_position_ids])
            sin = sin.contiguous()
            used_sin.append(sin[cur_position_ids])
        cos = torch.cat(used_cos, dim=-1)
        sin = torch.cat(used_sin, dim=-1)
    else:
        cos = text_model.cache_cos[position_ids]
        sin = text_model.cache_sin[position_ids]

    position_embeddings = (cos, sin)

    # Split caches
    cache_keys = caches[: len(caches) // 2]
    cache_values = caches[len(caches) // 2 :]

    # Move caches to corresponding GPUs (if caches are not empty)
    if len(cache_keys) > 0:
        for layer_idx in range(len(text_model.layers)):
            target_device = text_model._layer_to_device[layer_idx]
            if layer_idx < len(cache_keys):
                cache_keys[layer_idx] = cache_keys[layer_idx].to(device=target_device)
            if layer_idx < len(cache_values):
                cache_values[layer_idx] = cache_values[layer_idx].to(
                    device=target_device
                )

    # Forward pass, transfer hidden_states between GPUs
    hidden_states = inputs_embeds
    new_keys = []
    new_values = []

    for layer_idx, decoder_layer in enumerate(text_model.layers):
        target_device = text_model._layer_to_device[layer_idx]
        # Move hidden_states to current layer's GPU
        hidden_states = hidden_states.to(device=target_device)
        position_embeddings_gpu = (
            position_embeddings[0].to(device=target_device),
            position_embeddings[1].to(device=target_device),
        )
        attention_mask_gpu = attention_mask.to(device=target_device)

        # Call decoder layer
        hidden_states, new_key, new_value = decoder_layer(
            hidden_states,
            attention_mask=attention_mask_gpu,
            position_embeddings=position_embeddings_gpu,
            cache_keys=cache_keys[layer_idx] if len(cache_keys) else None,
            cache_values=cache_values[layer_idx] if len(cache_values) else None,
        )
        new_keys.append(new_key)
        new_values.append(new_value)

    # Move hidden_states back to first GPU for norm and lm_head
    hidden_states = hidden_states.to(device=primary_device)
    hidden_states = text_model.norm(hidden_states)
    token_logits = text_model.lm_head(hidden_states)
    if hasattr(text_model, "dequant"):
        token_logits = text_model.dequant(token_logits)

    return token_logits, new_keys, new_values


def init_kv_cache(
    cache_keys, cache_values, attention_mask, num_valid_tokens, max_lm_tokens
):
    num_layers = len(cache_keys)

    for idx in range(num_layers):
        cache_keys[idx] = cache_keys[idx][:, :num_valid_tokens]
        cache_values[idx] = cache_values[idx][:, :num_valid_tokens]
        attention_mask = attention_mask[:, :num_valid_tokens]
    bs, cur_tokens, num_heads, embed_dim = cache_keys[0].shape
    pad_tokens = max_lm_tokens - cur_tokens
    for idx in range(num_layers):
        pad_keys = torch.zeros(bs, pad_tokens, num_heads, embed_dim).to(
            device=cache_keys[idx].device
        )
        cache_keys[idx] = torch.cat([pad_keys, cache_keys[idx]], dim=1)

        pad_values = torch.zeros(bs, pad_tokens, num_heads, embed_dim).to(
            device=cache_values[idx].device
        )
        cache_values[idx] = torch.cat([pad_values, cache_values[idx]], dim=1)
    return cache_keys, cache_values, attention_mask


def remove_repeat(pixel_values):
    tokens = pixel_values.shape[0]
    pixel_values = pixel_values.reshape([-1, 3, 2, 14, 14])
    pixel_values = pixel_values[:, :, 0]
    pixel_values = pixel_values.reshape(tokens, -1)
    return pixel_values


def get_rope_index(
    config,
    input_ids,
    image_grid_thw,
    attention_mask=None,
):
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)
    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_index = 0
    attention_mask = attention_mask.to(total_input_ids.device)
    for i, input_ids in enumerate(total_input_ids):
        input_ids = input_ids[attention_mask[i] == 1]
        image_nums = 0
        vision_start_indices = torch.argwhere(
            input_ids == vision_start_token_id
        ).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images = image_nums
        for _ in range(image_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            t, h, w = (
                image_grid_thw[image_index][0],
                image_grid_thw[image_index][1],
                image_grid_thw[image_index][2],
            )
            second_per_grid_t = 0
            image_index += 1
            remain_images -= 1
            ed = ed_image
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            range_tensor = torch.arange(llm_grid_t).view(-1, 1)
            expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

            second_per_grid_t = torch.as_tensor(
                second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
            )

            time_tensor = (
                expanded_range
                * second_per_grid_t
                * config.vision_config.tokens_per_second
            )

            time_tensor_long = time_tensor.long()
            t_index = time_tensor_long.flatten()

            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
            position_ids.device
        )
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device
    ).unsqueeze(1)
    return position_ids, mrope_position_deltas


def padding_input_ids(input_ids, max_len, left=True, pad_token_id=151643):
    bs, cur_len = input_ids.shape
    pad_len = max_len - cur_len

    pad_input_ids = torch.full(
        (bs, pad_len),
        pad_token_id,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )

    if left:
        pad_input_ids = torch.cat([pad_input_ids, input_ids], dim=1)
    else:
        pad_input_ids = torch.cat([input_ids, pad_input_ids], dim=1)
    return pad_input_ids


def padding_mask(mask, max_len, left=True):
    bs, cur_len = mask.shape
    pad_len = max_len - cur_len
    pad_mask = torch.zeros((bs, pad_len)).to(device=mask.device, dtype=mask.dtype)
    pad_mask = torch.cat([pad_mask, mask], dim=1) if left else torch.cat([mask, pad_mask], dim=1)
    return pad_mask


def get_causal_mask(attention_mask, max_lm_tokens, min_value=-512):
    bs, seq_len = attention_mask.shape
    causal_mask = (
        torch.triu(torch.ones(seq_len, seq_len), 1)
        .bool()
        .to(device=attention_mask.device)
    )
    attention_mask = 1 - attention_mask
    q_attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)
    k_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    qk_attention_mask = q_attention_mask | k_attention_mask
    attention_mask = causal_mask.unsqueeze(0) | qk_attention_mask.bool()
    pad_tokens = max_lm_tokens - seq_len
    pad_mask = torch.ones(bs, 1, seq_len, pad_tokens).to(device=attention_mask.device)
    attention_mask = torch.cat([pad_mask, attention_mask], dim=-1)
    attention_mask = torch.where(attention_mask == 1, min_value, 0)
    return attention_mask


def gen_inputs_embeds(config, input_ids, inputs_embeds, image_embeds, window_index):
    reverse_indices = torch.argsort(window_index)
    image_embeds = image_embeds[:, reverse_indices, :]

    image_mask = input_ids == config.image_token_id
    image_mask = (
        image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    )
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    return inputs_embeds


def init_prefill_kv_cachev(
    bs, num_layers, num_heads, head_dim, attention_mask, max_lm_tokens
):
    cache_keys = []
    cache_values = []
    for _ in range(num_layers):
        cache_keys.append(
            torch.zeros(bs, max_lm_tokens, num_heads, head_dim).to(
                device=attention_mask.device
            )
        )
        cache_values.append(
            torch.zeros(bs, max_lm_tokens, num_heads, head_dim).to(
                device=attention_mask.device
            )
        )
    return cache_keys, cache_values


def ensure_visual_dimensions(conversation, image_width, image_height):
    # Predefined dimensions for video elements
    video_width = 448
    video_height = 252

    # Predefined dimensions for image elements
    image_width = image_width
    image_height = image_height

    for msg in conversation:
        contents = msg.get("content")
        if not isinstance(contents, list):
            continue

        for ele in contents:
            if ele.get("type") == "video":
                ele["resized_width"] = video_width
                ele["resized_height"] = video_height

            if ele.get("type") == "image":
                ele["resized_width"] = image_width
                ele["resized_height"] = image_height

    return conversation


class Qwen2_5VlApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_tsv_path: str,
        calib_message_path: str,
        chunk_size: int = 256,
        batch_size: int = 1,
        cache_len: int = 512,
        image_width: int = 448,
        image_height: int = 448,
        devices: list[str] = None,
        device: str = None,  # For backward compatibility
        model_type: str = "qwen2_5-vl-3b",
        dtype: str = "float16",
        w_bits: int = 8,
        mask_value: int = -32768,
        vit_core_num: list[int] = None,
        prefill_core_num: list[int] = None,
        decode_core_num: list[int] = None,
        input_model_format: str = "hf",
        march: str = "nash-p",
    ):
        if vit_core_num is None:
            vit_core_num = [1]
        if prefill_core_num is None:
            prefill_core_num = [1]
        if decode_core_num is None:
            decode_core_num = [1]
        self.input_model_path = input_model_path
        self.calib_tsv_path = calib_tsv_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.cache_len = cache_len
        self.image_width = image_width
        self.image_height = image_height
        # Support both devices (list) and device (single string) for backward compatibility
        if devices is not None:
            self.devices = devices if isinstance(devices, list) else [devices]
        elif device is not None:
            self.devices = [device] if isinstance(device, str) else device
        else:
            self.devices = ["cpu"]
        self.primary_device = self.devices[0]
        self.device = self.primary_device  # Keep for backward compatibility
        self.dtype = dtype
        self.model_type = model_type
        self.w_bits = w_bits
        self.vit_core_num = vit_core_num
        self.prefill_core_num = prefill_core_num
        self.decode_core_num = decode_core_num
        self.input_model_format = input_model_format
        self.mask_value = mask_value

        self.output_vit_model_path = standard_vit_name(
            input_model_path,
            output_model_path,
            march,
            vit_core_num,
            image_width,
            image_height,
        )

        self.output_lm_model_path = standard_lm_name(
            input_model_path,
            output_model_path,
            chunk_size,
            cache_len,
            w_bits,
            march,
            prefill_core_num,
            decode_core_num,
            batch_size=batch_size,
        )

        self.token_embeddings_file_name = standard_token_embeddings_name(
            input_model_path, output_model_path
        )

        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_dir = output_model_path

        if self.input_model_format == "hf":
            self.ckpt_dir = save_model_checkpoint(
                model_dir=self.input_model_path,
                output_model_path=self.output_model_dir,
            )
            model_dir = self.ckpt_dir
        elif self.input_model_format == "llmc":
            model_dir = self.input_model_path
        else:
            raise ValueError(
                f"input_model_format {self.input_model_format} is not supported"
            )

        self.model_wrapper = Qwen2_5_VL.build(
            model_dir=model_dir,
            chunk_size=self.chunk_size,
            batch_size=self.batch_size,
            cache_len=self.cache_len,
            w_bits=self.w_bits,
            mask_value=mask_value,
            input_model_format=self.input_model_format,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        self.model = self.model_wrapper.model
        self.processor = AutoProcessor.from_pretrained(
            self.input_model_path, use_fast=True
        )
        if calib_tsv_path:
            self.calib_datas = load_tsv_data(calib_tsv_path)
        else:
            self.calib_datas = load_message_data(
                calib_message_path, model_type=model_type
            )

    def save_embed_tokens(self):
        if hasattr(self.model, "get_input_embeddings"):
            emb_mod = self.model.get_input_embeddings()
            if hasattr(emb_mod, "weight"):
                emb = emb_mod.weight.detach().to(dtype=torch.float16).cpu().numpy()
                out = self.token_embeddings_file_name
                if not os.path.exists(out):
                    emb.tofile(out)

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        # Use primary device for model setup
        device = (
            self.primary_device
            if torch.cuda.is_available() and self.primary_device.startswith("cuda")
            else "cpu"
        )

        dtype = torch.float32
        
        # Setup multi-GPU if needed (only for text model layers)
        if len(self.devices) > 1 and self.primary_device != "cpu":
            # Multi-GPU mode: distribute text model layers
            text_model = self.model.get_text_model()
            visual_model = self.model.get_visual_model()
            input_embeddings = self.model.get_input_embeddings()
            
            # Move visual_model and input_embeddings to primary device
            visual_model.to(device=self.primary_device, dtype=dtype)
            if hasattr(input_embeddings, "to"):
                input_embeddings.to(device=self.primary_device, dtype=dtype)
            
            if hasattr(text_model, "layers"):
                num_layers = len(text_model.layers)
                num_devices = len(self.devices)
                layers_per_device = num_layers // num_devices
                remainder = num_layers % num_devices

                # Store multi-GPU info
                text_model._layer_to_device = {}
                layer_idx = 0
                for device_idx, device_name in enumerate(self.devices):
                    start_layer = layer_idx
                    end_layer = (
                        layer_idx + layers_per_device + (1 if device_idx < remainder else 0)
                    )

                    # Move corresponding layers to current device
                    for i in range(start_layer, end_layer):
                        text_model.layers[i] = text_model.layers[i].to(device=device_name, dtype=dtype)
                        text_model._layer_to_device[i] = device_name

                    layer_idx = end_layer

                # Move other text model components to first device
                if hasattr(text_model, "embed_tokens"):
                    text_model.embed_tokens = text_model.embed_tokens.to(device=self.primary_device, dtype=dtype)
                if hasattr(text_model, "norm"):
                    text_model.norm = text_model.norm.to(device=self.primary_device, dtype=dtype)
                if hasattr(text_model, "lm_head"):
                    text_model.lm_head = text_model.lm_head.to(device=self.primary_device, dtype=dtype)
                if hasattr(text_model, "cache_cos"):
                    text_model.cache_cos = text_model.cache_cos.to(device=self.primary_device)
                if hasattr(text_model, "cache_sin"):
                    text_model.cache_sin = text_model.cache_sin.to(device=self.primary_device)

                text_model._multi_gpu_devices = self.devices

                print(
                    f"Multi-device setup for text model: {num_layers} layers distributed across {num_devices} devices"
                )
                layer_idx = 0
                for device_idx, device_name in enumerate(self.devices):
                    start_layer = layer_idx
                    end_layer = (
                        layer_idx + layers_per_device + (1 if device_idx < remainder else 0)
                    )
                    print(
                        f"  Device {device_name}: layers {start_layer}-{end_layer-1} ({end_layer-start_layer} layers)"
                    )
                    layer_idx = end_layer
            else:
                # Fallback to single device
                self.model.to(device=device, dtype=dtype)
        else:
            # Single device mode
            self.model.to(device=device, dtype=dtype)
        
        self.model.compile_mode(False)

        compile_vit_kwargs = vit_kwargs or {}
        compile_llm_kwargs = llm_kwargs or {}
        compile_kwargs = {}
        compile_kwargs.update(compile_vit_kwargs)
        compile_kwargs.update(compile_llm_kwargs)

        # Save embedding weights for engine consumption before to fp16
        self.save_embed_tokens()
        self._calibrate_forward(device=device, dtype=dtype, **compile_kwargs)

        self.model.compile_mode(True)
        self.model.to("cpu", dtype=torch.float16)

        gc.collect()
        if device != "cpu":
            # Clear CUDA cache on all devices used (handles multi-GPU case)
            for dev in self.devices:
                if dev != "cpu" and dev.startswith("cuda"):
                    with torch.cuda.device(dev):
                        torch.cuda.empty_cache()
            print(f"[GPU Memory] Released CUDA cache on {self.devices} after calibration.")

        self.model_wrapper.compile(
            output_lm_model_path=self.output_lm_model_path,
            output_vit_model_path=self.output_vit_model_path,
            vit_core_num=self.vit_core_num,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            enable_vpu=True,
            vit_kwargs=vit_kwargs,
            llm_kwargs=llm_kwargs,
        )

    def _calibrate_forward(self, *, device: str, dtype, **kwargs):
        config = self.model.get_config()
        visual_model = self.model.get_visual_model()
        text_model = self.model.get_text_model()
        input_embeddings = self.model.get_input_embeddings()
        padding_side = "left"
        left = padding_side == "left"

        # Ensure visual_model and input_embeddings are on the correct device
        # In multi-GPU mode, they should be on primary_device
        target_device = self.primary_device if len(self.devices) > 1 and self.primary_device != "cpu" else device
        if visual_model is not None:
            visual_model.to(device=target_device, dtype=dtype)
        if input_embeddings is not None and hasattr(input_embeddings, "to"):
            input_embeddings.to(device=target_device, dtype=dtype)

        image_grid_thw = [
            [
                1,
                config.vision_config.image_height // config.vision_config.patch_size,
                config.vision_config.image_width // config.vision_config.patch_size,
            ]
        ]
        image_grid_thw = torch.tensor(image_grid_thw, device=target_device)
        window_index, _ = visual_model.get_window_index(image_grid_thw)

        num_vision_token = (image_grid_thw[0][1] * image_grid_thw[0][2]) // (
            config.vision_config.spatial_merge_size**2
        )
        max_prefill_text_token = config.text_config.max_prefill_text_token
        max_prefill_token = max_prefill_text_token + num_vision_token.cpu().numpy()
        max_lm_tokens = 4096
        num_heads = config.text_config.num_attention_heads
        head_dim = config.text_config.hidden_size // num_heads
        num_layers = config.text_config.num_hidden_layers
        num_key_value_heads = config.text_config.num_key_value_heads
        target_image_width = config.vision_config.image_width
        target_image_height = config.vision_config.image_height

        for messages in self.calib_datas:
            if isinstance(messages, dict):
                messages = [messages]
            text = self.processor.apply_chat_template(
                [messages], tokenize=False, add_generation_prompt=True
            )
            messages = ensure_visual_dimensions(
                messages, target_image_width, target_image_height
            )
            images, videos = process_vision_info([messages])  # type: ignore
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )

            for k, v in inputs.items():
                inputs[k] = v.to(target_device)

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]
            attention_mask = inputs["attention_mask"]

            attention_mask = padding_mask(attention_mask, max_prefill_token, left=left)
            paded_input_ids = padding_input_ids(input_ids, max_prefill_token, left=left)
            position_ids, _ = get_rope_index(
                config, paded_input_ids, image_grid_thw, attention_mask
            )
            position_ids = position_ids.squeeze().unsqueeze(0)

            with torch.no_grad():
                inputs_embeds = input_embeddings(paded_input_ids.to(target_device))
                pixel_values = remove_repeat(pixel_values).to(target_device)
                image_embeds = visual_model.forward(pixel_values.unsqueeze(0))
                inputs_embeds = gen_inputs_embeds(
                    config, paded_input_ids.to(target_device), inputs_embeds, image_embeds, window_index
                )

                bs = inputs_embeds.shape[0]
                prefill_cache_keys, prefill_cache_values = init_prefill_kv_cachev(
                    bs,
                    num_layers,
                    num_key_value_heads,
                    head_dim,
                    attention_mask,
                    max_lm_tokens,
                )

                # Move KV caches to target device
                for i in range(len(prefill_cache_keys)):
                    prefill_cache_keys[i] = prefill_cache_keys[i].to(device=target_device)
                    prefill_cache_values[i] = prefill_cache_values[i].to(device=target_device)

                causal_mask = get_causal_mask(
                    attention_mask, max_lm_tokens, self.mask_value
                ).squeeze(0).to(target_device)
                prefill_caches = prefill_cache_keys + prefill_cache_values
                
                # Use multi-GPU forward if needed
                if len(self.devices) > 1 and self.primary_device != "cpu" and hasattr(text_model, "_multi_gpu_devices"):
                    # Distribute KV caches to corresponding devices in multi-GPU mode
                    if hasattr(text_model, "_layer_to_device"):
                        for layer_idx in range(num_layers):
                            target_gpu = text_model._layer_to_device[layer_idx]
                            prefill_cache_keys[layer_idx] = prefill_cache_keys[layer_idx].to(device=target_gpu)
                            prefill_cache_values[layer_idx] = prefill_cache_values[layer_idx].to(device=target_gpu)
                        prefill_caches = prefill_cache_keys + prefill_cache_values
                    
                    text_model_forward_multi_gpu(
                        text_model,
                        inputs_embeds=inputs_embeds.to(target_device),
                        position_ids=position_ids.to(target_device),
                        attention_mask=causal_mask,
                        caches=prefill_caches,
                    )
                else:
                    text_model.forward(
                        attention_mask=causal_mask,
                        inputs_embeds=inputs_embeds.to(target_device),
                        position_ids=position_ids.to(target_device),
                        caches=prefill_caches,
                    )

    def get_quant_path(self) -> tuple[Optional[str], Optional[str]]:
        """Return fixed Qwen2.5 VL BC paths."""
        llm_bc = str(Path(self.output_model_dir).with_suffix(".prefill_convert_rm.bc"))
        vlm_bc = str(Path(self.output_model_dir).with_suffix(".convert.bc"))
        return llm_bc, vlm_bc

    def get_hbm_path(self) -> tuple[Optional[str], Optional[str]]:
        """Return fixed Qwen2.5 VL HBM paths."""
        return self.output_lm_model_path, self.output_vit_model_path
