from __future__ import annotations

from typing import Any, List, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer

from leap_llm.apis.calibration.calibration import (
    get_pos_ids,
    pad_to_multiple,
    update_causal_mask,
)
from leap_llm.models.internvl3_5.configuration import (
    InternVL3_5LLMConfig,
    InternVL3_5VisionConfig,
)
from leap_llm.models.internvl3_5.model import (
    InternVisionModel,
    InterVL3_5,
    Qwen3ForCausalLM,
)


class InternVL3_5LlmWrapper(nn.Module):
    """Verifier-side wrapper that adapts Qwen3ForCausalLM to the generic interface."""

    def __init__(self, model: Qwen3ForCausalLM, model_args: InternVL3_5LLMConfig):
        super().__init__()
        self.inner = model
        self.inner.compile_mode(False)
        self.model_args = model_args

    @staticmethod
    def load_model(
        input_model_path: str,
        checkpoint: dict | None = None,
        chunk_size: int = 256,
        cache_len: int = 4096,
        kept_tokens_file: str | None = None,
        prebuilt: InterVL3_5 | None = None,
        **kwargs,
    ) -> "InternVL3_5LlmWrapper":
        intervl = prebuilt or InterVL3_5.build(
            input_model_path, chunk_size=chunk_size, cache_len=cache_len
        )
        language_model = intervl.get_language_model()
        return InternVL3_5LlmWrapper(language_model, intervl.model_args.llm_config)

    def get_input_embeddings(self):
        return self.inner.get_input_embeddings()

    def get_model_args(self):
        return self.model_args

    def compile_mode(self, mode: bool = True):
        self.inner.compile_mode(mode)
        return self

    def to(self, device, dtype=None):
        # 先调用父类方法，确保 wrapper 自身的 buffers 和参数都被移动
        if dtype is not None:
            super().to(device, dtype=dtype)
        else:
            super().to(device)
        # 确保内部模型也被移动
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
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value_list: List[torch.Tensor],
    ):
        inputs_embeds = (
            self.get_input_embeddings()(tokens)
            if tokens.dim() == 2
            else tokens  # already embeddings
        )

        num_layers = self.model_args.num_hidden_layers
        cache_keys = past_key_value_list[:num_layers]
        cache_values = past_key_value_list[num_layers:]

        cache_keys_for_call = [
            cache.unsqueeze(0) if cache.dim() == 3 else cache for cache in cache_keys
        ]
        cache_values_for_call = [
            cache.unsqueeze(0) if cache.dim() == 3 else cache for cache in cache_values
        ]

        logits, new_keys, new_values = self.inner.forward(
            inputs_embeds,
            position_ids,
            attention_mask,
            cache_keys=cache_keys_for_call,
            cache_values=cache_values_for_call,
        )

        flattened: list[torch.Tensor] = []
        for tensor in new_keys:
            flattened.append(tensor)
        for tensor in new_values:
            flattened.append(tensor)
        return (logits, *flattened)

    def __getattr__(self, name: str):
        if name in {"inner", "model_args"}:
            return super().__getattr__(name)
        return getattr(self.inner, name)


class InternVL3_5VisionWrapper(nn.Module):
    """Verifier-side wrapper so vision model matches legacy interface."""

    def __init__(self, model: InternVisionModel, model_args: InternVL3_5VisionConfig):
        super().__init__()
        self.inner = model
        self.inner.compile_mode(False)
        self.model_args = model_args

    @staticmethod
    def load_model(
        input_model_path: str,
        checkpoint: dict | None = None,
        prebuilt: InterVL3_5 | None = None,
        **kwargs,
    ) -> "InternVL3_5VisionWrapper":
        intervl = prebuilt or InterVL3_5.build(input_model_path)
        vision_model = intervl.get_vit_model()
        return InternVL3_5VisionWrapper(vision_model, intervl.model_args.vision_config)

    def get_model_args(self):
        return self.model_args

    def compile_mode(self, mode: bool = True):
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

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def __getattr__(self, name: str):
        if name in {"inner", "model_args"}:
            return super().__getattr__(name)
        return getattr(self.inner, name)


def prepare_internvl35_inputs(
    text_input: str,
    tokenizer: AutoTokenizer,
    llm_wrapper: InternVL3_5LlmWrapper,
    chunk_size: int,
    cache_len: int,
    device: str,
    mask_value: float = -512,
    pos_mask_value: int = 1,
    padding_side: str = "left",
) -> Tuple[
    Tuple[torch.Tensor, ...],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """Prepare inputs for InternVL3.5-1b text-only inference.

    This function prepares chunked inputs compatible with Backend's inference loop.

    Args:
        text_input: Text string to process
        tokenizer: Tokenizer instance
        llm_wrapper: InternVL3.5 LLM wrapper
        chunk_size: Chunk size for prefill
        cache_len: KV cache length
        device: Device string
        mask_value: Mask fill value
        pos_mask_value: Position mask value
        padding_side: Padding side ('left' or 'right')

    Returns:
        Tuple of:
            - input_chunks: Tuple of input token tensors
            - causal_mask_chunks: List of attention mask tensors
            - position_ids_chunks: List of position ID tensors
            - past_key_value_list: Initial KV cache tensors
    """
    model_args = llm_wrapper.get_model_args()

    # Tokenize
    tokenizer.padding_side = padding_side
    inputs = tokenizer(text_input, return_tensors="pt")
    input_ids_valid = inputs.input_ids.to(device)
    raw_inputs_len = input_ids_valid.shape[-1]

    # Truncate if too long
    if raw_inputs_len > cache_len:
        inputs = tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=cache_len - 10,
        )
        input_ids_valid = inputs.input_ids.to(device)

    n = input_ids_valid.shape[-1]
    inputs_pad_len, _ = pad_to_multiple(n, chunk_size)

    # Re-tokenize with padding to the aligned length
    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=inputs_pad_len,
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask_final = inputs.attention_mask.to(device)
    valid_seq_len = int(torch.sum(attention_mask_final).item())

    # Initialize KV cache with InternVL3.5-1b specific shape:
    # [1, cache_len, num_kv_heads, head_dim]
    init_kv_data = torch.zeros(
        [1, cache_len, model_args.num_key_value_heads, model_args.head_dim],
        dtype=torch.float32,
    ).to(device)

    past_key_value_list: List[Any] = [init_kv_data] * model_args.num_hidden_layers + [
        init_kv_data
    ] * model_args.num_hidden_layers

    # Prepare input chunks and masks
    input_chunks = input_ids.split(chunk_size, dim=-1)
    total_inputs_len = sum(chunk.shape[-1] for chunk in input_chunks)

    attention_mask_1d = torch.zeros((1, cache_len), dtype=torch.int32, device=device)
    attention_mask_1d[0, -valid_seq_len:] = 1

    cache_position = torch.arange(0, cache_len, dtype=torch.long, device=device)
    dummy_input_ids = torch.zeros((1, valid_seq_len), dtype=torch.long, device=device)

    causal_mask_4d = update_causal_mask(
        attention_mask_1d,
        dummy_input_ids,
        cache_position,
        min_dtype=mask_value,
        sequence_length=cache_len,
        kv_cache_len=cache_len,
        dtype=torch.float32,
        device=device,
        padding_side=padding_side,
    )

    causal_mask_4d = causal_mask_4d[:, :, -total_inputs_len:, :]
    causal_mask_chunks = list(causal_mask_4d.split(chunk_size, dim=2))

    position_ids_chunks = get_pos_ids(
        chunk_size,
        valid_seq_len,
        padding_side,
        pos_mask_value,
    )

    causal_mask_chunks = [item.to(device) for item in causal_mask_chunks]
    position_ids_chunks = [item.to(device) for item in position_ids_chunks]

    if len(causal_mask_chunks) != len(position_ids_chunks) or len(input_chunks) != len(
        position_ids_chunks
    ):
        raise ValueError("mask/position lens error!!!!!!")

    return (
        input_chunks,
        causal_mask_chunks,
        position_ids_chunks,
        past_key_value_list,
    )
