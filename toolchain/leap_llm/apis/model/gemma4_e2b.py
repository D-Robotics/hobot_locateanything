import logging
import os

import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoProcessor, AutoTokenizer

from leap_llm.apis.calibration.data_loader import load_message_data, load_text_data
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
)
from leap_llm.models.gemma4_e2b.model import Gemma4ModelWrapper, Gemma4TextModel, Gemma4VisionModel


def create_logger(name, log_lvl=logging.INFO):
    logger = logging.getLogger(name)
    handle = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s:%(lineno)d %(message)s")
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    logger.setLevel(log_lvl)
    return logger


def get_kv_cache(config: Gemma4TextConfig, dtype=torch.float16):
    bs = 1
    num_prev_layers = config.num_hidden_layers - config.num_kv_shared_layers
    k_cache, v_cache = [], []
    for i in range(num_prev_layers):
        if config.layer_types[i] == "sliding_attention":
            # sliding_attention should take advantage of the sliding_window size
            # the maximum valid window of the seq_len is actually double of the sliding_window
            k_cache.append(
                torch.zeros(bs, 2 * config.sliding_window, config.num_key_value_heads, config.head_dim, dtype=dtype)
            )
            v_cache.append(
                torch.zeros(bs, 2 * config.sliding_window, config.num_key_value_heads, config.head_dim, dtype=dtype)
            )
        elif config.layer_types[i] == "full_attention":
            # full_attention KV cache is identical to that of the conventional context cache
            k_cache.append(
                torch.zeros(bs, config.cache_len, config.num_key_value_heads, config.global_head_dim, dtype=dtype)
            )
            v_cache.append(
                torch.zeros(bs, config.cache_len, config.num_key_value_heads, config.global_head_dim, dtype=dtype)
            )
        else:
            raise ValueError(f"Unknown layer type: {config.layer_types[i]}")
    return [*k_cache, *v_cache]


def get_fa_mask(attention_mask, max_lm_tokens, min_value=-512):
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


def get_sa_mask(attention_mask, max_lm_tokens, sliding_window, cache_valid_len=0, min_value=-512):
    """Sliding-window causal attention mask for edge NPU deployment.

    Same layout as get_fa_mask: [bs, 1, seq_len, max_lm_tokens]
    KV axis: [empty_cache | valid_cache | current_chunk (seq_len)]
    where valid_cache = cache_valid_len tokens of real data.

    For SA layers, max_lm_tokens = 2 * sliding_window (the KV cache size).
    The mask enforces: (1) causality, (2) sliding window constraint,
    (3) padding mask, (4) empty cache masking.

    Args:
        attention_mask: 2D [bs, seq_len] padding mask (1=valid, 0=pad)
        max_lm_tokens: total KV dimension (= 2 * sliding_window for SA)
        sliding_window: sw, the sliding window size
        cache_valid_len: how many tokens in cache are valid (0 for first chunk)
        min_value: mask value for hidden positions
    """
    bs, seq_len = attention_mask.shape
    device = attention_mask.device
    pad_tokens = max_lm_tokens - seq_len

    # ---- Padding mask for current chunk tokens
    pad_mask = (1 - attention_mask).bool()
    q_pad_mask = pad_mask.unsqueeze(1).unsqueeze(3)  # [bs, 1, S, 1]
    k_pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, S]
    qk_pad_mask = q_pad_mask | k_pad_mask  # [bs, 1, S, S]

    # ---- Causal + sliding window on full KV axis
    # Q absolute position = pad_tokens + q_idx (aligned with get_fa_mask)
    q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
    kv_idx = torch.arange(max_lm_tokens, device=device).view(1, 1, 1, max_lm_tokens)
    q_global = q_idx + pad_tokens

    # Causal: kv_idx > q_global (future tokens)
    causal_mask = kv_idx > q_global

    # Sliding window: kv_idx <= q_global - sliding_window (too old)
    out_of_window = kv_idx <= (q_global - sliding_window)

    # Cache empty region: mask positions that don't have valid data
    # Valid cache: [pad_tokens - cache_valid_len, pad_tokens)
    # Empty cache: [0, pad_tokens - cache_valid_len)
    empty_cache_len = max(pad_tokens - cache_valid_len, 0)
    if empty_cache_len > 0:
        empty_cache_mask = torch.zeros(bs, 1, seq_len, pad_tokens, device=device, dtype=torch.bool)
        empty_cache_mask[:, :, :, :empty_cache_len] = True
    else:
        empty_cache_mask = torch.zeros(bs, 1, seq_len, pad_tokens, device=device, dtype=torch.bool)

    # Combine masks across full KV axis
    full_qk_pad_mask = torch.cat([empty_cache_mask, qk_pad_mask], dim=-1)
    full_mask = causal_mask | out_of_window | full_qk_pad_mask

    # ---- Convert: True (masked) -> min_value, False (visible) -> 0
    attention_mask = torch.where(full_mask, min_value, 0)
    return attention_mask


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


def pad_input_ids(input_ids, max_len, left=True):
    bs, cur_len = input_ids.shape
    pad_len = max_len - cur_len

    if pad_len <= 0:
        # NOTE: no padding needed, splitting is needed here,
        # return mask for fast calib.
        return input_ids

    pad_input_ids = torch.zeros((bs, pad_len)).to(device=input_ids.device, dtype=input_ids.dtype)
    if left:
        pad_input_ids = torch.cat([pad_input_ids, input_ids], dim=1)
    else:
        pad_input_ids = torch.cat([input_ids, pad_input_ids], dim=1)

    return pad_input_ids


def pad_position_ids(position_ids, max_len, left=True):
    bs, cur_len = position_ids.shape
    pad_len = max_len - cur_len

    if pad_len <= 0:
        # NOTE: no padding needed, splitting is needed here,
        # return mask for fast calib.
        return position_ids

    pad_position_ids = torch.ones((bs, pad_len)).to(device=position_ids.device, dtype=position_ids.dtype)
    if left:
        pad_position_ids = torch.cat([pad_position_ids, position_ids], dim=1)
    else:
        pad_position_ids = torch.cat([position_ids, pad_position_ids], dim=1)

    return pad_position_ids


def sliding_window_mask_function(sliding_window: tuple[int, int]):
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        left_window_size, right_window_size = sliding_window

        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask


def _convert_4d_mask_to_blocked_5d(config: Gemma4AudioConfig, mask_4d: torch.Tensor) -> torch.Tensor:
    """
    Convert a standard 4D attention mask `[batch_size, 1, seq_len, seq_len]` to the 5D blocked format
    `[batch_size, 1, num_blocks, chunk_size, context_size]` expected by the chunked local attention,
    """
    batch_size, _, seq_len, _ = mask_4d.shape
    device = mask_4d.device

    chunk_size = config.attention_chunk_size
    max_past_horizon = config.attention_context_left - 1
    max_future_horizon = config.attention_context_right

    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    padded_seq_len = num_blocks * chunk_size
    pad_amount = padded_seq_len - seq_len

    mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
    mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
    mask_5d = F.pad(mask_5d, (max_past_horizon, max_future_horizon), value=False)

    block_starts = torch.arange(num_blocks, device=device) * chunk_size
    offsets = torch.arange(chunk_size + max_past_horizon + max_future_horizon, device=device)
    kv_indices = block_starts[:, None] + offsets[None, :]
    kv_indices = kv_indices[None, None, :, None, :].expand(batch_size, 1, -1, chunk_size, -1)

    mask_5d = mask_5d.gather(-1, kv_indices)

    return mask_5d


def get_placeholder_mask(config: Gemma4Config, input_ids, image_features=None, audio_features=None):
    special_image_mask = input_ids == config.image_token_id
    special_video_mask = input_ids == config.video_token_id
    special_audio_mask = input_ids == config.audio_token_id

    if image_features is not None:
        n_image_tokens = special_image_mask.sum()
        if image_features.shape[1] != n_image_tokens:
            raise ValueError(
                f"image_features shape {image_features.shape} does not match " f"n_image_tokens {n_image_tokens}"
            )

    if audio_features is not None:
        n_audio_tokens = special_audio_mask.sum()
        if audio_features.shape[1] != n_audio_tokens:
            raise ValueError(
                f"audio_features shape {audio_features.shape} does not match " f"n_audio_tokens {n_audio_tokens}"
            )

    return special_image_mask, special_video_mask, special_audio_mask


class Gemma4E2BApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        chunk_size: int = 512,
        cache_len: int = 4096,
        image_width: int = 384,
        image_height: int = 384,
        device: str = "cuda",
        dtype: str = "float32",
        model_type: str = "gemma-4-e2b-it",
        w_bits: int = 8,
        mask_value: float = -32768.0,
        march: str = "nash-p",
        vit_core_num: list[int] = None,
        prefill_core_num: list[int] = None,
        decode_core_num: list[int] = None,
        input_model_format: str = "hf",
        **kwargs,
    ):
        self.logger = create_logger(__name__)
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.image_width = image_width
        self.image_height = image_height
        self.device = device
        self.dtype = dtype
        self.model_type = model_type
        self.w_bits = w_bits
        self.mask_value = mask_value
        self.march = march
        self.vit_core_num = vit_core_num[0]
        self.prefill_core_num = prefill_core_num[0]
        self.decode_core_num = decode_core_num[0]
        self.calib_text_data = load_text_data(calib_text_path)

        os.makedirs(output_model_path, exist_ok=True)

        if model_type == "gemma-4-e2b-it":
            prefix = "Gemma-4-E2B-it"
        else:
            raise ValueError(f"model_type {model_type} is not supported")

        self.output_llm_model_path = os.path.join(
            output_model_path,
            f"{prefix}_language_chunk_{chunk_size}_cache_{cache_len}_w{w_bits}"
            f"_{self.march}_corenum_{self.prefill_core_num}_{self.decode_core_num}.hbm",
        )

        self.output_vision_model_path = os.path.join(
            output_model_path,
            f"{prefix}_vision_{self.image_height}x{self.image_width}_"
            f"w8_{self.march}_corenum_"
            f"{self.vit_core_num}.hbm",
        )

        os.makedirs(output_model_path, exist_ok=True)

        self.gemma4_model: Gemma4ModelWrapper = Gemma4ModelWrapper.build(
            input_model_path,
            output_model_dir=output_model_path,
            model_type=model_type,
            chunk_size=chunk_size,
            cache_len=cache_len,
            w_bits=w_bits,
            mask_value=mask_value,
            march=self.march,
            vit_core_num=self.vit_core_num,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            image_thw=[image_height // 16, image_width // 16],
        )

        self.model = self.gemma4_model.get_model()
        self.config = self.gemma4_model.get_model_args()

        self._resolution_sanity_check(image_width, image_height)

        self.sliding_window = self.config.text_config.sliding_window
        self.tokenizer = AutoTokenizer.from_pretrained(self.input_model_path)
        self.processor = AutoProcessor.from_pretrained(self.input_model_path)
        self.calib_data = load_message_data(None, model_type=model_type)

    def _resolution_sanity_check(self, width, height):
        pooling_kernel_size = self.config.vision_config.pooling_kernel_size
        patch_size = self.config.vision_config.patch_size
        multiple_of = pooling_kernel_size * patch_size
        assert width % multiple_of == 0, f"width {width} must be divisible by {multiple_of}"
        assert height % multiple_of == 0, f"height {height} must be divisible by {multiple_of}"
        height_patches = height // patch_size
        width_patches = width // patch_size
        self.num_patches = height_patches * width_patches

    def _update_key_value_caches(self, kv_history: list, kv_result: list, is_prefill=False):
        # kv_history is with the reserved k/v cache size (bs, kv_len, 1, hidden_dim)
        # kv_result is with the k/v result size (bs, chunk_size, 1, hidden_dim)
        # as prefill we use chunked_prefill, we directly concat the kv_result to the kv_history along dim=1
        # with chunk_size, as the original is left-padding to the multiply of the chunk_size
        kv_history = [kvh.to(self.device) for kvh in kv_history]
        key_result, value_result = kv_result[len(kv_result) // 2 :], kv_result[: len(kv_result) // 2]
        key_history, value_history = kv_history[len(kv_history) // 2 :], kv_history[: len(kv_history) // 2]

        history_shift = self.chunk_size if is_prefill else 1

        for ki, kr in enumerate(key_result):
            kc = key_history[ki]
            history = kc[:, history_shift:]
            key_history[ki] = torch.cat([history, kr], dim=1)

        for vi, vr in enumerate(value_result):
            vc = value_history[vi]
            history = vc[:, history_shift:]
            value_history[vi] = torch.cat([history, vr], dim=1)

        kv_cache = key_history + value_history

        return kv_cache

    def _vision_llm_calib_forward(self, device, dtype, **kwargs):
        config = self.config
        text_config: Gemma4TextConfig = self.config.text_config
        lang_model: Gemma4TextModel = self.model.language
        vision_model: Gemma4VisionModel = self.model.visual

        for msg_idx, message in enumerate(self.calib_data):
            self.logger.debug(message)
            if isinstance(message, dict):
                message = [message]

            # let's load the image to PIL.Image and resize to the (height, width) accordingly
            # pass into the template as PIL.Image
            image = Image.open(message[0]["content"][0]["image"])
            image = image.resize((self.image_height, self.image_width))
            message[0]["content"][0]["image"] = image

            # Process input
            inputs = self.processor.apply_chat_template(
                [message],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
                add_generation_prompt=True,
                processor_kwargs={"max_soft_tokens": 70},
            ).to(device)

            pixel_values = inputs["pixel_values"].to(dtype=dtype)
            pixel_values = pixel_values[..., : self.num_patches, :]

            input_ids = inputs["input_ids"]

            with torch.no_grad():
                real_seq_len = input_ids.shape[-1]
                self.logger.debug(f"real_seq_len = {real_seq_len}")
                seq_len = self.chunk_size

                vision_feature = vision_model(pixel_values)

                # input_ids
                input_ids = pad_input_ids(input_ids, seq_len)
                self.logger.debug(f"input_ids.shape = {input_ids.shape}")

                # position_ids
                position_ids = torch.arange(real_seq_len, device=device, dtype=torch.long).unsqueeze(0)
                position_ids = pad_position_ids(position_ids, seq_len)
                self.logger.debug(f"position_ids.shape = {position_ids.shape}")

                image_mask, _, _ = get_placeholder_mask(config, input_ids, image_features=vision_feature)

                llm_input_ids = input_ids.clone()
                llm_input_ids[image_mask] = text_config.pad_token_id

                # inputs_embeds handling
                inputs_embeds = lang_model.get_input_embeddings()(llm_input_ids)
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).to(device),
                    vision_feature.to(device),
                )
                self.logger.debug(f"inputs_embeds.shape = {inputs_embeds.shape}")

                # per-layer embedding handling (token identity only; projection happens inside TextModel)
                per_layer_inputs = lang_model.get_per_layer_input_embeddings()(llm_input_ids)
                per_layer_inputs = per_layer_inputs.reshape(
                    *llm_input_ids.shape,
                    text_config.num_hidden_layers,
                    text_config.hidden_size_per_layer_input,
                )
                self.logger.debug(f"per_layer_inputs.shape = {per_layer_inputs.shape}")

                attention_mask = pad_mask(inputs["attention_mask"], seq_len)

                full_attention_mask = get_fa_mask(
                    attention_mask,
                    max_lm_tokens=self.cache_len,
                    min_value=self.mask_value,
                )
                self.logger.debug(f"full_attention_mask.shape = {full_attention_mask.shape}")

                sa_kv_len = self.chunk_size + text_config.sliding_window

                sliding_attention_mask = get_sa_mask(
                    attention_mask,
                    max_lm_tokens=sa_kv_len,
                    sliding_window=text_config.sliding_window,
                    min_value=self.mask_value,
                )
                self.logger.debug(f"sliding_attention_mask.shape = {sliding_attention_mask.shape}")

                past_key_values = get_kv_cache(text_config)
                past_key_values = [kv.to(device) for kv in past_key_values]

                for i, kv in enumerate(past_key_values):
                    self.logger.debug(f"cache[{i}].shape = {kv.shape}")

                _ = lang_model(
                    None,
                    inputs_embeds,
                    per_layer_inputs,
                    full_attention_mask,
                    sliding_attention_mask,
                    position_ids,
                    *past_key_values,
                )

            self.logger.info(f"calibrated [{msg_idx}]")

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        device = self.device if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu"

        dtype = torch.float16

        self.model.compile_mode(False)
        self.model.to(device=device, dtype=dtype)
        self.model.eval()

        self._vision_llm_calib_forward(device=device, dtype=dtype, **vit_kwargs)

        # Move vision tower back to CPU and switch to compile mode for export
        self.model.to(device="cpu", dtype=dtype)
        self.model.compile_mode(True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.gemma4_model.model.vision_tower.compile(
            self.output_vision_model_path,
            enable_vpu=True,
            vision_core_num=self.vit_core_num,
            num_patches=self.num_patches,
            **vit_kwargs,
        )

        self.gemma4_model.model.language_model.compile(
            self.output_llm_model_path,
            enable_vpu=True,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            **llm_kwargs,
        )

        self.logger.info(f"Model compiled successfully: {self.output_model_path}")
