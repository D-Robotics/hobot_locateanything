import importlib
import importlib.util
import logging
import os
from typing import List, Optional

import torch
from qwen_vl_utils import (
    process_vision_info,
)

# NOTE: qwen-vl-utils==0.0.14 for qwen3-vl
from transformers import AutoProcessor

from leap_llm.apis.calibration.data_loader import load_message_data, load_tsv_data
from leap_llm.models.qwen3_vl.model import (
    Qwen3VL_Wrapper,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


LEAP_LLM_LOC = os.path.dirname(importlib.util.find_spec("leap_llm").origin)


def create_logger(name, log_lvl=logging.INFO):
    logger = logging.getLogger(name)
    handle = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s:%(lineno)d %(message)s")
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    logger.setLevel(log_lvl)
    return logger


def ensure_visual_dimensions(
    conversation,
    image_width: int = 448,
    image_height: int = 448,
    video_width: int = 448,
    video_height: int = 448,
):
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


def get_rope_index(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Different from the original implementation,
    Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to separate videos, like <t1>
    # <vision_start> <frame1> <vision_end> <t2> <vision_start>
    # <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
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
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1
                # (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1)
                t_index = t_index.expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1)
                h_index = h_index.expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1)
                w_index = w_index.expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_vision_placeholder_mask(config, input_ids, image_features):
    image_mask = input_ids == config.image_token_id
    video_mask = input_ids == config.video_token_id
    n_image_tokens = image_mask.sum()
    if image_features.shape[1] != n_image_tokens:
        raise ValueError(
            f"Image features shape mismatch with #image_tokens,"
            f"tokens {n_image_tokens}, features {image_features.shape[1]}"
        )
    return image_mask, video_mask


def get_causal_mask(attention_mask, max_lm_tokens, min_value=-512):
    bs, seq_len = attention_mask.shape
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), 1).bool().to(device=attention_mask.device)
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


def get_kv_cache(bs, num_layers, num_kv_heads, head_dim, attention_mask, max_lm_tokens):
    cache_keys = []
    cache_values = []
    for _ in range(num_layers):
        cache_keys.append(torch.zeros(bs, max_lm_tokens, num_kv_heads, head_dim).to(device=attention_mask.device))
        cache_values.append(torch.zeros(bs, max_lm_tokens, num_kv_heads, head_dim).to(device=attention_mask.device))
    return cache_keys, cache_values


def pad_input_ids(input_ids: torch.Tensor, max_len, left=True, pad_token_id: int = 151643):
    """
    pad_token_id shall <|endoftext|> in tokenizer config.
    """
    bs, cur_len = input_ids.shape
    pad_len = max_len - cur_len

    if pad_len <= 0:
        # NOTE: no padding needed, splitting is needed here,
        # return input_ids for fast calib.
        return input_ids

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


def pad_mask(mask: torch.Tensor, max_len, left=True):
    bs, cur_len = mask.shape
    pad_len = max_len - cur_len

    if pad_len <= 0:
        # NOTE: no padding needed, splitting is needed here,
        # return mask for fast calib.
        return mask

    pad_mask = torch.zeros((bs, pad_len)).to(device=mask.device, dtype=mask.dtype)
    pad_mask = torch.cat([pad_mask, mask], dim=1) if left else torch.cat([mask, pad_mask], dim=1)

    return pad_mask


def pad_deepstack(deepstack_feature, visual_mask):
    """pad deepstack visual embedding according to visual_mask

    Args:
        visual_mask (bool): (bs, seq_len)
        deepstack_feature (float16): (bs, visual_dim, hidden_size)
    """
    bs, seq_len = visual_mask.shape
    _, _, hs = deepstack_feature.shape
    new_feature = torch.zeros(
        (bs, seq_len, hs),
        dtype=deepstack_feature.dtype,
        device=deepstack_feature.device,
    )
    new_feature[visual_mask] = deepstack_feature
    return new_feature


class Qwen3VlApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_tsv_path: str,
        calib_message_path: str,
        chunk_size: int = 256,
        cache_len: int = 512,
        image_width: int = 448,
        image_height: int = 448,
        device: str | List[str] = "cpu",
        model_type: str = "qwen3-vl-4b",
        march: str = "nash-p",
        dtype: str = "float16",
        w_bits: int = 8,
        mask_value: int = -32768,
        vit_core_num: list[int] = None,
        prefill_core_num: list[int] = None,
        decode_core_num: list[int] = None,
        input_model_format: str = "hf",
    ):
        self.logger = create_logger(__name__)
        self.input_model_path = input_model_path
        self.calib_tsv_path = calib_tsv_path
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.image_width = image_width
        self.image_height = image_height
        self.dtype = dtype
        self.model_type = model_type
        self.w_bits = w_bits
        self.vit_core_num = vit_core_num
        self.prefill_core_num = prefill_core_num
        self.decode_core_num = decode_core_num
        self.input_model_format = input_model_format
        self.mask_value = mask_value

        # model prefix for output model name
        if self.model_type == "qwen3-vl-2b":
            prefix = "Qwen3-VL-2B-Instruct"
        elif self.model_type == "qwen3-vl-4b":
            prefix = "Qwen3-VL-4B-Instruct"
        elif self.model_type == "qwen3-vl-8b":
            prefix = "Qwen3-VL-8B-Instruct"
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        # multiple device for 8B compilation
        if isinstance(device, list):
            if len(device) > 1:
                self.multi_gpu = True
                self.device = device
            else:
                self.multi_gpu = False
                self.device = device[0]
        else:
            self.multi_gpu = False
            self.device = device

        self.output_visual_model_path = os.path.join(
            output_model_path,
            f"{prefix}_vision_{self.image_height}x{self.image_width}_"
            f"w8_{march}_corenum_"
            f"{self.vit_core_num[0]}.hbm",
        )
        self.output_lang_model_path = os.path.join(
            output_model_path,
            f"{prefix}_language_chunk_{chunk_size}_cache_{cache_len}_"
            f"w{w_bits}_{march}_corenum_"
            f"{self.prefill_core_num[0]}_{self.decode_core_num[0]}.hbm",
        )

        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_dir = output_model_path

        self.model_wrapper = Qwen3VL_Wrapper.build(
            model_dir=self.input_model_path,
            model_type=self.model_type,
            chunk_size=self.chunk_size,
            cache_len=self.cache_len,
            w_bits=self.w_bits,
            mask_value=self.mask_value,
            output_model_dir=output_model_path,
            logger=self.logger,
        )

        self.model = self.model_wrapper.get_model()
        self.config = self.model_wrapper.get_model_args()
        self.processor = AutoProcessor.from_pretrained(self.input_model_path)
        if calib_tsv_path:
            self.calib_data = load_tsv_data(calib_tsv_path)
        else:
            # NOTE: using None to let engine resolve path automatically
            self.calib_data = load_message_data(None, model_type=model_type)

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        self.logger.info(f"vit_compile_args:\n{vit_kwargs}")
        self.logger.info(f"llm_compile_args:\n{llm_kwargs}")

        if not self.multi_gpu:
            device = self.device if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu"
        else:
            device = self.device

        dtype = torch.float32

        if self.multi_gpu and self.device[0].startswith("cuda"):
            vision_model = self.model.visual
            text_model = self.model.language_model

            primary_device = self.device[0]

            # vision_model to primary device
            vision_model.to(device=primary_device, dtype=dtype)

            # text_model auxillaries to primary device
            if hasattr(text_model, "embed_tokens"):
                text_model.embed_tokens = text_model.embed_tokens.to(device=primary_device, dtype=dtype)

            if hasattr(text_model, "norm"):
                text_model.norm = text_model.norm.to(device=primary_device, dtype=dtype)

            if hasattr(text_model, "lm_head"):
                text_model.lm_head = text_model.lm_head.to(device=primary_device, dtype=dtype)

            if hasattr(text_model, "rotary_emb"):
                text_model.rotary_emb = text_model.rotary_emb.to(device=primary_device)

            if hasattr(text_model, "layers"):
                num_layers = len(text_model.layers)
                num_devices = len(self.device)
                text_model._layer_to_device = {}
                layer_idx = 0

                # Distribute layers across devices
                # (first devices get extra layer if remainder exists)
                for device_idx, device_name in enumerate(self.device):
                    layers_this_device = num_layers // num_devices + (device_idx < num_layers % num_devices)
                    start_layer = layer_idx
                    end_layer = layer_idx + layers_this_device

                    # Move layers to device and store mapping
                    for i in range(start_layer, end_layer):
                        text_model.layers[i] = text_model.layers[i].to(device=device_name, dtype=dtype)
                        text_model._layer_to_device[i] = device_name

                    # Log distribution summary
                    self.logger.info(
                        f"  Device {device_name}: layer {start_layer}-{end_layer-1} " f"({layers_this_device} layers)"
                    )
                    layer_idx = end_layer
        else:
            # maintain one GPU logic for calibration
            self.model.to(device=device, dtype=dtype)
            self.logger.info(f"device={device}, dtype={dtype}")

        self.model.compile_mode(False)

        self._calib_forward(device=device)

        self.model.to(device="cpu", dtype=torch.float16)

        if not self.multi_gpu:
            if self.device.startswith('cuda'):
                with torch.cuda.device(self.device):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            for d in self.device:
                with torch.cuda.device(d):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        self.model.compile_mode(True)

        self.model.visual.compile(
            self.output_visual_model_path,
            enable_vpu=True,
            core_num=self.vit_core_num[0],
            **vit_kwargs,
        )

        self.model.language_model.compile(
            self.output_lang_model_path,
            enable_vpu=True,
            prefill_core_num=self.prefill_core_num[0],
            decode_core_num=self.decode_core_num[0],
            **llm_kwargs,
        )

    def _calib_forward(self, device):
        if self.multi_gpu:
            device = self.device[0]

        config = self.config
        self.calib_max_len_token_id = 0  # calib 120 data's maximum input_ids = 413
        self.logger.debug(f"leap_llm location: {LEAP_LLM_LOC}")
        visual_model: Qwen3VLVisionModel = self.model.visual
        lang_model: Qwen3VLTextModel = self.model.language_model
        # Vision model
        image_grid_thw = [
            [
                1,
                self.image_height // self.config.vision_config.patch_size,
                self.image_width // self.config.vision_config.patch_size,
            ]
        ]
        image_grid_thw = torch.tensor(image_grid_thw, device=device)
        num_vision_token = (image_grid_thw[0][1] * image_grid_thw[0][2]) // (
            self.config.vision_config.spatial_merge_size**2
        )
        num_vision_token = int(num_vision_token.item())
        # Language model
        lang_num_heads = self.config.text_config.num_attention_heads
        lang_head_dim = getattr(
            self.config.text_config,
            "head_dim",
            self.config.text_config.hidden_size // lang_num_heads,
        )
        lang_num_layers = self.config.text_config.num_hidden_layers
        lang_num_kv_heads = self.config.text_config.num_key_value_heads
        # lang_ctx_len = config.text_config.cache_len # pad to maximum context length
        lang_chunk_size = config.text_config.chunk_size

        for msg_idx, message in enumerate(self.calib_data):
            if isinstance(message, dict):
                message = [message]

            text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

            self.logger.debug(f"text:\n{text}")

            message = ensure_visual_dimensions(message)

            images, _ = process_vision_info([message])  # NOTE: video not taken for now

            inputs = self.processor(text=text, images=images, padding=True, return_tensors="pt")

            for k, v in inputs.items():
                inputs[k] = v.to(device)

            # input_ids: (bs, #token),
            # mask: (bs, #token)
            # pixel: (#visual_token, #visual_emb_size)
            # grid_thw: (bs, 3)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]

            self.calib_max_len_token_id = max(self.calib_max_len_token_id, input_ids.size(1))

            attention_mask = pad_mask(attention_mask, lang_chunk_size)

            input_ids = pad_input_ids(input_ids, lang_chunk_size)

            position_ids, _ = get_rope_index(
                config=config,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )

            input_embeds = lang_model.get_input_embeddings()(input_ids)

            bs = input_embeds.size(0)

            cache_keys, cache_values = get_kv_cache(
                bs,
                lang_num_layers,
                lang_num_kv_heads,
                lang_head_dim,
                attention_mask,
                self.cache_len,
            )

            processed_num_vision_token = (
                image_grid_thw[0][1] * image_grid_thw[0][2] // (self.config.vision_config.spatial_merge_size**2)
            )

            processed_num_vision_token = int(processed_num_vision_token.item())

            assert num_vision_token == processed_num_vision_token, "number of vision tokens mismatched."
            f"got user input image_height={self.image_height}, "
            f"image_width={self.image_width} "
            "got inputs['image_grid_thw']=\n"
            f"{inputs['image_grid_thw']}"

            # (bs, #visual_token, #visual_emb_size)
            pixel_values = pixel_values.unsqueeze(0)

            pixel_values.to(device)

            self.logger.debug(f"pixel_values.shape = {pixel_values.shape}")

            with torch.no_grad():
                img_embed, deepstack_img_emb = visual_model(pixel_values)

                image_mask, _ = get_vision_placeholder_mask(config, input_ids, img_embed)
                self.logger.debug(f"image_mask.shape = {image_mask.shape}")
                input_embeds[image_mask] = img_embed  # runtime impl this
                self.logger.debug(f"number of deepstack features = {len(deepstack_img_emb)}")

                # prepare deepstack image embedding according to {image, video}_mask
                deepstack_visual_embeds = [pad_deepstack(d, image_mask) for d in deepstack_img_emb]
                for d in deepstack_visual_embeds:
                    self.logger.debug(f"padded deepstack feature = {d.shape}")

                causal_attention_mask = get_causal_mask(attention_mask, self.cache_len, self.mask_value)

                if self.multi_gpu:
                    outputs = self._multi_gpu_forward(
                        input_embeds,
                        position_ids,
                        causal_attention_mask,
                        cache_keys,
                        cache_values,
                        deepstack_visual_embeds,
                    )
                else:
                    outputs = lang_model(
                        input_embeds,
                        position_ids,
                        causal_attention_mask,
                        cache_keys,
                        cache_values,
                        deepstack_visual_embeds,
                    )

                if msg_idx == 0:
                    self.logger.debug(f"logits.shape = {outputs[0].shape}")
                    self.logger.debug(f"cache.shape = {outputs[1].shape}")

            self.logger.info(f"MMStar calibrated [{msg_idx}]")

        self.logger.info("MMStar Calibration Done.")

    def _multi_gpu_forward(
        self, input_embeds, position_ids, causal_attention_mask, cache_keys, cache_values, deepstack_visual_embeds
    ):
        """
        Multi-GPU forward function for language model
        """
        hidden_states = input_embeds
        text_model = self.model.language_model
        position_embeddings = text_model.rotary_emb(hidden_states, position_ids)
        new_keys, new_values = [], []
        for layer_idx, decoder_layer in enumerate(text_model.layers):
            target_device = text_model._layer_to_device[layer_idx]
            hidden_states = hidden_states.to(device=target_device)
            position_embeddings_gpu = (
                position_embeddings[0].to(device=target_device),
                position_embeddings[1].to(device=target_device),
            )
            causal_attention_mask_gpu = causal_attention_mask.to(device=target_device)
            ck = cache_keys[layer_idx] if cache_keys else None
            cache_keys_gpu = ck.to(device=target_device) if ck is not None else None
            cv = cache_values[layer_idx] if cache_values else None
            cache_values_gpu = cv.to(device=target_device) if cv is not None else None
            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask_gpu,
                position_embeddings=position_embeddings_gpu,
                past_keys=cache_keys_gpu,
                past_values=cache_values_gpu,
            )
            new_keys.append(new_key)
            new_values.append(new_value)
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = text_model._deepstack_process(
                    hidden_states, deepstack_visual_embeds[layer_idx].to(device=target_device)
                )

        # Move all new_keys and new_values to the primary device
        primary_device = self.device[0]
        new_keys = [k.to(device=primary_device) for k in new_keys]
        new_values = [v.to(device=primary_device) for v in new_values]

        hidden_states = hidden_states.to(device=primary_device)
        hidden_states = text_model.norm(hidden_states)
        token_logits = text_model.lm_head(hidden_states)
        return token_logits, *new_keys, *new_values
