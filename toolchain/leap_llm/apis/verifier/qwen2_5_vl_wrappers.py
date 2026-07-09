"""Qwen2.5-VL wrappers and data processing utilities for verifier."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from leap_llm.models.qwen2_5_vl.model import (
    Qwen2_5_VL,
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionModel,
)

# Default Qwen2.5-VL image size
QWEN2_5_VL_RESIZED_WIDTH = 952
QWEN2_5_VL_RESIZED_HEIGHT = 420


@dataclass
class Qwen2_5VLModelArgs:
    """Model args adapter for Qwen2.5-VL to match Backend expectations."""

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int
    prefill_seq_len: int
    cache_len: int


class Qwen2_5VLLlmWrapper(nn.Module):
    """Verifier-side wrapper that adapts Qwen2_5_VLTextModel to the generic interface."""  # noqa: E501

    def __init__(
        self,
        model: Qwen2_5_VLTextModel,
        config: Qwen2_5_VLConfig,
        chunk_size: int = 256,
        cache_len: int = 4096,
    ):
        super().__init__()
        self.inner = model
        self.config = config
        self.text_config = config.text_config
        self.chunk_size = chunk_size
        self.cache_len = cache_len

        # Build model args for Backend compatibility
        self._model_args = Qwen2_5VLModelArgs(
            num_hidden_layers=self.text_config.num_hidden_layers,
            num_attention_heads=self.text_config.num_attention_heads,
            num_key_value_heads=self.text_config.num_key_value_heads,
            hidden_size=self.text_config.hidden_size,
            head_dim=self.text_config.hidden_size
            // self.text_config.num_attention_heads,  # noqa: E501
            prefill_seq_len=chunk_size,
            cache_len=cache_len,
        )

    @staticmethod
    def load_model(
        input_model_path: str,
        chunk_size: int = 256,
        cache_len: int = 4096,
        prebuilt: Qwen2_5_VL | None = None,
        **kwargs,
    ) -> "Qwen2_5VLLlmWrapper":
        qwen_wrapper = prebuilt or Qwen2_5_VL.build(
            input_model_path,
            chunk_size=chunk_size,
            cache_len=cache_len,
            input_model_format="llmc",
        )
        text_model = qwen_wrapper.get_text_model()
        config = qwen_wrapper.model_args
        return Qwen2_5VLLlmWrapper(text_model, config, chunk_size, cache_len)

    def get_input_embeddings(self):
        return self.inner.get_input_embeddings()

    def get_model_args(self):
        return self._model_args

    def get_config(self):
        return self.config

    def compile_mode(self, mode: bool = True):
        if hasattr(self.inner, "compile_mode"):
            self.inner.compile_mode(mode)
        return self

    def to(self, device, dtype=None):
        if dtype is not None:
            self.inner.to(device, dtype=dtype)
        else:
            self.inner.to(device)
        return self

    def eval(self):
        self.inner.eval()
        return self

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value_list: List[torch.Tensor],
    ):
        """Forward pass matching Backend's expected interface.

        Args:
            inputs_embeds: Input embeddings [batch, seq_len, hidden_size]
            position_ids: Position IDs [batch, 3, seq_len] for prefill or [batch, 1, seq_len] for decode
            attention_mask: Attention mask [batch, seq_len, cache_len]
            past_key_value_list: List of KV cache tensors

        Returns:
            Tuple of (logits, *new_keys, *new_values)
        """  # noqa: E501
        num_layers = self._model_args.num_hidden_layers
        cache_keys = past_key_value_list[:num_layers]
        cache_values = past_key_value_list[num_layers:]

        caches = cache_keys + cache_values

        logits, new_keys, new_values = self.inner.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            caches=caches,
        )

        flattened: list[torch.Tensor] = []
        for tensor in new_keys:
            flattened.append(tensor)
        for tensor in new_values:
            flattened.append(tensor)
        return (logits, *flattened)

    def __getattr__(self, name: str):
        if name in {
            "inner",
            "config",
            "text_config",
            "_model_args",
            "chunk_size",
            "cache_len",
        }:  # noqa: E501
            return super().__getattr__(name)
        return getattr(self.inner, name)


class Qwen2_5VLVisionWrapper(nn.Module):
    """Verifier-side wrapper for Qwen2.5-VL vision model."""

    def __init__(
        self,
        model: Qwen2_5_VLVisionModel,
        config: Qwen2_5_VLConfig,
    ):
        super().__init__()
        self.inner = model
        self.config = config
        self.vision_config = config.vision_config

    @staticmethod
    def load_model(
        input_model_path: str,
        prebuilt: Qwen2_5_VL | None = None,
        **kwargs,
    ) -> "Qwen2_5VLVisionWrapper":
        qwen_wrapper = prebuilt or Qwen2_5_VL.build(
            input_model_path, input_model_format="llmc"
        )
        vision_model = qwen_wrapper.get_visual_model()
        config = qwen_wrapper.model_args
        return Qwen2_5VLVisionWrapper(vision_model, config)

    def get_config(self):
        return self.config

    def compile_mode(self, mode: bool = True):
        if hasattr(self.inner, "compile_mode"):
            self.inner.compile_mode(mode)
        return self

    def to(self, device, dtype=None):
        if dtype is not None:
            self.inner.to(device, dtype=dtype)
        else:
            self.inner.to(device)
        return self

    def eval(self):
        self.inner.eval()
        return self

    def prepare_vision_input(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image input for Qwen2.5-VL VisionModel.

        Converts image from [N, C, H, W] format to [1, seq_len, patch_dim] format
        expected by Qwen2.5-VL VisionModel.

        Args:
            image: Input image tensor in [N, C, H, W] format

        Returns:
            Tensor in [1, seq_len, patch_size*patch_size*in_channels] format
        """
        target_height = self.vision_config.image_height
        target_width = self.vision_config.image_width
        patch_size = self.vision_config.patch_size

        # Handle different input shapes
        if image.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            image = image.unsqueeze(0)

        # Take only the first image if multiple are provided
        if image.shape[0] > 1:
            image = image[0:1]

        # Resize to target dimensions
        if image.shape[2] != target_height or image.shape[3] != target_width:
            image = F.interpolate(
                image,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )

        # Convert [1, C, H, W] to [1, seq_len, patch_dim]
        # seq_len = (H // patch_size) * (W // patch_size)
        # patch_dim = patch_size * patch_size * in_channels
        batch_size, channels, height, width = image.shape
        grid_h = height // patch_size
        grid_w = width // patch_size

        # Reshape: [1, C, H, W] -> [1, C, grid_h, patch_size, grid_w, patch_size]
        image = image.view(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
        # Permute: -> [1, grid_h, grid_w, patch_size, patch_size, C]
        image = image.permute(0, 2, 4, 3, 5, 1)
        # Reshape: -> [1, seq_len, patch_size * patch_size * C]
        patch_dim = patch_size * patch_size * channels
        image = image.reshape(batch_size, grid_h * grid_w, patch_dim)

        return image

    def forward(
        self, pixel_values: torch.Tensor, auto_preprocess: bool = False
    ) -> torch.Tensor:
        """Forward pass for vision model.

        Args:
            pixel_values: Image tensor. If auto_preprocess is False, should be
                [batch, seq_len, patch_size*patch_size*channels]. If auto_preprocess
                is True, can be [N, C, H, W] format.
            auto_preprocess: If True, automatically preprocess [N,C,H,W] input
                to the expected format.

        Returns:
            Image embeddings
        """
        if auto_preprocess and pixel_values.dim() == 4:
            pixel_values = self.prepare_vision_input(pixel_values)
        return self.inner.forward(pixel_values)

    def __getattr__(self, name: str):
        if name in {"inner", "config", "vision_config"}:
            return super().__getattr__(name)
        return getattr(self.inner, name)


# ============================================================================
# Data processing functions for Qwen2.5-VL
# ============================================================================


def get_rope_index_text_only(
    config: Qwen2_5_VLConfig,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rope position indices for text-only input (no images).

    For text-only input, all three dimensions of position_ids are the same.

    Args:
        config: Qwen2.5-VL config
        input_ids: Token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]

    Returns:
        position_ids: [3, batch, seq_len]
        mrope_position_deltas: [batch, 1]
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    batch_size, seq_len = input_ids.shape
    position_ids = torch.zeros(
        3, batch_size, seq_len, dtype=input_ids.dtype, device=input_ids.device
    )
    mrope_position_deltas = []

    for i in range(batch_size):
        mask = attention_mask[i]
        valid_len = mask.sum().item()

        # For text-only, position ids are simply cumulative within valid tokens
        pos = torch.zeros(seq_len, dtype=input_ids.dtype, device=input_ids.device)
        if valid_len > 0:
            valid_positions = torch.arange(valid_len, device=input_ids.device)
            # Find where valid tokens are (mask == 1)
            valid_indices = torch.where(mask == 1)[0]
            pos[valid_indices] = valid_positions

        # All three dimensions have the same position for text-only
        position_ids[0, i] = pos
        position_ids[1, i] = pos
        position_ids[2, i] = pos

        mrope_position_deltas.append(valid_len - seq_len)

    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device
    ).unsqueeze(1)

    return position_ids, mrope_position_deltas


def padding_input_ids(
    input_ids: torch.Tensor, max_len: int, left: bool = True
) -> torch.Tensor:  # noqa: E501
    """Pad input_ids to max_len.

    Args:
        input_ids: Token IDs [batch, seq_len]
        max_len: Target length
        left: If True, pad on left side

    Returns:
        Padded input_ids [batch, max_len]
    """
    bs, cur_len = input_ids.shape
    if cur_len >= max_len:
        return input_ids[:, :max_len]

    pad_len = max_len - cur_len
    pad_ids = torch.zeros((bs, pad_len), device=input_ids.device, dtype=input_ids.dtype)

    if left:
        return torch.cat([pad_ids, input_ids], dim=1)
    else:
        return torch.cat([input_ids, pad_ids], dim=1)


def padding_mask(mask: torch.Tensor, max_len: int, left: bool = True) -> torch.Tensor:
    """Pad attention mask to max_len.

    Args:
        mask: Attention mask [batch, seq_len]
        max_len: Target length
        left: If True, pad on left side

    Returns:
        Padded mask [batch, max_len]
    """
    bs, cur_len = mask.shape
    if cur_len >= max_len:
        return mask[:, :max_len]

    pad_len = max_len - cur_len
    pad_mask = torch.zeros((bs, pad_len), device=mask.device, dtype=mask.dtype)

    if left:
        return torch.cat([pad_mask, mask], dim=1)
    else:
        return torch.cat([mask, pad_mask], dim=1)


def get_causal_mask(
    attention_mask: torch.Tensor,
    cache_len: int,
    min_value: float = -512,
) -> torch.Tensor:
    """Create causal attention mask for Qwen2.5-VL.

    Args:
        attention_mask: Binary attention mask [batch, seq_len]
        cache_len: KV cache length
        min_value: Mask fill value for invalid positions

    Returns:
        Causal mask [batch, 1, seq_len, cache_len]
    """
    bs, seq_len = attention_mask.shape

    # Create causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=attention_mask.device), diagonal=1
    ).bool()

    # Combine with attention mask
    inv_attention_mask = 1 - attention_mask
    q_mask = inv_attention_mask.unsqueeze(1).unsqueeze(3)
    k_mask = inv_attention_mask.unsqueeze(1).unsqueeze(2)
    qk_mask = q_mask | k_mask

    combined_mask = causal_mask.unsqueeze(0) | qk_mask.bool()

    # Pad to cache_len
    pad_len = cache_len - seq_len
    if pad_len > 0:
        pad_mask = torch.ones(bs, 1, seq_len, pad_len, device=attention_mask.device)
        combined_mask = torch.cat([pad_mask, combined_mask], dim=-1)

    # Convert to float mask
    mask = torch.where(combined_mask == 1, min_value, 0.0)
    return mask


def init_kv_cache(
    batch_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    cache_len: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """Initialize KV cache for Qwen2.5-VL.

    Qwen2.5-VL uses cache shape [batch, cache_len, num_kv_heads, head_dim].

    Args:
        batch_size: Batch size
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value heads
        head_dim: Head dimension
        cache_len: Cache length
        device: Device
        dtype: Data type

    Returns:
        List of cache tensors: [key_0, key_1, ..., value_0, value_1, ...]
    """
    cache_list = []

    # Keys
    for _ in range(num_layers):
        cache_list.append(
            torch.zeros(
                batch_size,
                cache_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )  # noqa: E501
        )

    # Values
    for _ in range(num_layers):
        cache_list.append(
            torch.zeros(
                batch_size,
                cache_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )  # noqa: E501
        )

    return cache_list


def prepare_qwen2_5_vl_inputs(
    text_input: str,
    tokenizer,
    llm_wrapper: Qwen2_5VLLlmWrapper,
    chunk_size: int,
    cache_len: int,
    device: str,
    mask_value: float = -512,
) -> Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
]:  # noqa: E501
    """Prepare inputs for Qwen2.5-VL text-only inference.

    This function prepares chunked inputs compatible with Backend's inference loop.

    Args:
        text_input: Text string to process
        tokenizer: Tokenizer instance
        llm_wrapper: Qwen2.5-VL LLM wrapper
        chunk_size: Chunk size for prefill
        cache_len: KV cache length
        device: Device string
        mask_value: Mask fill value

    Returns:
        Tuple of:
            - input_chunks: List of input embedding tensors
            - causal_mask_chunks: List of attention mask tensors
            - position_ids_chunks: List of position ID tensors
            - past_key_value_list: Initial KV cache tensors
    """
    model_args = llm_wrapper.get_model_args()
    config = llm_wrapper.get_config()

    # Tokenize
    tokens = tokenizer(text_input, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids)).to(device)

    batch_size, seq_len = input_ids.shape

    # Calculate number of chunks needed
    num_chunks = math.ceil(seq_len / chunk_size)
    padded_len = num_chunks * chunk_size

    # Pad input_ids and attention_mask (left padding for Qwen2.5-VL)
    input_ids = padding_input_ids(input_ids, padded_len, left=True)
    attention_mask = padding_mask(attention_mask, padded_len, left=True)

    # Get embeddings
    embed_layer = llm_wrapper.get_input_embeddings()
    with torch.no_grad():
        inputs_embeds = embed_layer(input_ids)

    # Compute position IDs for the full sequence
    position_ids, _ = get_rope_index_text_only(config, input_ids, attention_mask)
    # Shape: [3, batch, padded_len] -> [batch, 3, padded_len]
    position_ids = position_ids.permute(1, 0, 2)

    # Initialize KV cache
    past_key_value_list = init_kv_cache(
        batch_size=batch_size,
        num_layers=model_args.num_hidden_layers,
        num_kv_heads=model_args.num_key_value_heads,
        head_dim=model_args.head_dim,
        cache_len=cache_len,
        device=device,
        dtype=torch.float32,
    )

    # Split into chunks
    input_chunks = []
    causal_mask_chunks = []
    position_ids_chunks = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        # Chunk embeddings
        chunk_embeds = inputs_embeds[:, start_idx:end_idx, :]
        input_chunks.append(chunk_embeds)

        # Chunk position IDs
        chunk_pos_ids = position_ids[:, :, start_idx:end_idx]
        position_ids_chunks.append(chunk_pos_ids)

        # Create causal mask for this chunk
        # The mask needs to attend to: past KV cache + current chunk
        chunk_attention_mask = attention_mask[:, :end_idx]
        chunk_attention_mask = padding_mask(chunk_attention_mask, cache_len, left=True)

        # Create the causal mask
        chunk_causal_mask = get_causal_mask(
            chunk_attention_mask[:, -chunk_size:],
            cache_len,
            min_value=mask_value,
        )
        # Squeeze to [batch, seq_len, cache_len] if needed
        if chunk_causal_mask.dim() == 4:
            chunk_causal_mask = chunk_causal_mask.squeeze(1)

        causal_mask_chunks.append(chunk_causal_mask)

    return input_chunks, causal_mask_chunks, position_ids_chunks, past_key_value_list


# ============================================================================
# Image processing functions for Qwen2.5-VL
# ============================================================================


def resize_image_for_qwen2_5_vl(
    image: Union[Image.Image, torch.Tensor, np.ndarray, str],
    target_width: int = QWEN2_5_VL_RESIZED_WIDTH,
    target_height: int = QWEN2_5_VL_RESIZED_HEIGHT,
) -> Image.Image:
    """Resize image to Qwen2.5-VL expected size.

    Args:
        image: Input image (PIL Image, torch Tensor, numpy array, or file path)
        target_width: Target width (default: 952)
        target_height: Target height (default: 420)

    Returns:
        Resized PIL Image
    """
    # Convert to PIL Image if needed
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, torch.Tensor):
        # Assume [C, H, W] or [H, W, C] format
        if image.dim() == 3:
            if image.shape[0] in [1, 3, 4]:  # [C, H, W]
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Resize to target size
    resized_image = pil_image.resize((target_width, target_height), Image.BILINEAR)
    return resized_image


def resize_image_tensor_for_qwen2_5_vl(
    image_tensor: torch.Tensor,
    target_width: int = QWEN2_5_VL_RESIZED_WIDTH,
    target_height: int = QWEN2_5_VL_RESIZED_HEIGHT,
) -> torch.Tensor:
    """Resize image tensor to Qwen2.5-VL expected size using torch interpolate.

    Args:
        image_tensor: Input tensor [batch, channels, height, width] or [channels, height, width]
        target_width: Target width (default: 952)
        target_height: Target height (default: 420)

    Returns:
        Resized tensor with same batch/channel dimensions
    """  # noqa: E501
    # Handle 3D tensor [C, H, W]
    squeeze_batch = False
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        squeeze_batch = True

    # Resize using bilinear interpolation
    resized = F.interpolate(
        image_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    if squeeze_batch:
        resized = resized.squeeze(0)

    return resized


def preprocess_image_for_qwen2_5_vl(
    image: Union[Image.Image, torch.Tensor, np.ndarray, str],
    target_width: int = QWEN2_5_VL_RESIZED_WIDTH,
    target_height: int = QWEN2_5_VL_RESIZED_HEIGHT,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
) -> torch.Tensor:
    """Preprocess image for Qwen2.5-VL vision model.

    Args:
        image: Input image (PIL Image, torch Tensor, numpy array, or file path)
        target_width: Target width (default: 952)
        target_height: Target height (default: 420)
        normalize: Whether to normalize the image
        mean: Normalization mean
        std: Normalization std

    Returns:
        Preprocessed tensor [1, channels, height, width]
    """
    # Resize image
    pil_image = resize_image_for_qwen2_5_vl(image, target_width, target_height)

    # Convert to tensor [C, H, W] with values in [0, 1]
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(
        2, 0, 1
    )  # [H, W, C] -> [C, H, W]  # noqa: E501

    # Normalize
    if normalize:
        mean_tensor = torch.tensor(mean).view(3, 1, 1)
        std_tensor = torch.tensor(std).view(3, 1, 1)
        image_tensor = (image_tensor - mean_tensor) / std_tensor

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]

    return image_tensor
