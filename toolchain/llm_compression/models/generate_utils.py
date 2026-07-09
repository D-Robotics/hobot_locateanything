"""Common utilities shared across model-specific generate_func implementations."""

import logging

import torch

from llm_compression.utils.device_manager import get_module_device  # noqa: F401

logger = logging.getLogger(__name__)


def padding_data(data, max_len, left=True, pad_value=0):
    bs, cur_len = data.shape
    pad_len = max_len - cur_len
    if pad_len <= 0:
        return data[:, -max_len:] if left else data[:, :max_len]
    pad = torch.full((bs, pad_len), pad_value, device=data.device, dtype=data.dtype)
    return torch.cat([pad, data], dim=1) if left else torch.cat([data, pad], dim=1)


def align_prefill_length(num_valid_tokens, chunk_size, max_kvcache_len, chunk_prefill):
    """Compute max_lm_input_len for prefill, capped to max_kvcache_len.

    When chunk_prefill is enabled, round up to the nearest multiple of
    chunk_size, then cap to max_kvcache_len so that get_causal_mask's
    pad_tokens (= max_kvcache_len - seq_len) never goes negative.
    """
    if chunk_prefill:
        aligned = ((num_valid_tokens + chunk_size - 1) // chunk_size) * chunk_size
        return min(aligned, max_kvcache_len)
    return chunk_size


def init_prefill_kv_cache(bs, num_layers, num_heads, head_dim, attention_mask, max_kvcache_len, dtype=None):
    cache_keys, cache_values = [], []
    for _ in range(num_layers):
        shape = (bs, max_kvcache_len, num_heads, head_dim)
        cache_keys.append(torch.zeros(shape, device=attention_mask.device, dtype=dtype))
        cache_values.append(torch.zeros(shape, device=attention_mask.device, dtype=dtype))
    return cache_keys, cache_values


def init_kv_cache(cache_keys, cache_values, attention_mask, num_valid_tokens, max_kvcache_len):
    num_layers = len(cache_keys)
    for idx in range(num_layers):
        cache_keys[idx] = cache_keys[idx][:, -num_valid_tokens:]
        cache_values[idx] = cache_values[idx][:, -num_valid_tokens:]
        attention_mask = attention_mask[:, -num_valid_tokens:]
    bs, cur_tokens, num_heads, embed_dim = cache_keys[0].shape
    pad_tokens = max_kvcache_len - cur_tokens
    for idx in range(num_layers):
        pad = torch.zeros(
            bs, pad_tokens, num_heads, embed_dim, device=cache_keys[idx].device, dtype=cache_keys[idx].dtype
        )
        cache_keys[idx] = torch.cat([pad, cache_keys[idx]], dim=1)
        cache_values[idx] = torch.cat([pad.clone(), cache_values[idx]], dim=1)
    return cache_keys, cache_values, attention_mask


def process_kv_cache(cache_keys, cache_values, new_keys, new_values):
    for idx in range(len(cache_keys)):
        refresh_len = new_keys[idx].shape[1]
        cache_keys[idx] = torch.cat([cache_keys[idx], new_keys[idx]], dim=1)[:, refresh_len:]
        cache_values[idx] = torch.cat([cache_values[idx], new_values[idx]], dim=1)[:, refresh_len:]
    return cache_keys, cache_values


def get_causal_mask(attention_mask, max_kvcache_len, min_value=-32768):
    bs, seq_len = attention_mask.shape
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), 1).bool().to(device=attention_mask.device)
    inv_mask = 1 - attention_mask
    q_mask = inv_mask.unsqueeze(1).unsqueeze(3)
    k_mask = inv_mask.unsqueeze(1).unsqueeze(2)
    attention_mask = causal_mask.unsqueeze(0) | (q_mask | k_mask).bool()
    pad_tokens = max_kvcache_len - seq_len
    pad_mask = torch.ones(bs, 1, seq_len, pad_tokens, device=attention_mask.device)
    attention_mask = torch.cat([pad_mask, attention_mask], dim=-1)
    return torch.where(attention_mask == 1, min_value, 0)


def get_causal_mask_chunks(full_mask, max_kvcache_len, chunk_size, min_value=-32768):
    """Split the full causal mask into per-chunk slices, right-aligned to max_kvcache_len.

    Each chunk mask is trimmed so it only attends to KV positions that have
    been filled by the time that chunk is processed, then right-aligned with
    ``min_value`` padding on the left.

    Args:
        attention_mask: 2D attention mask of shape (bs, seq_len).
        max_kvcache_len: Total KV-cache capacity.
        chunk_size: Number of query tokens per chunk.
        min_value: Fill value for masked positions.

    Returns:
        List of mask tensors, each of shape (bs, 1, chunk_size, max_kvcache_len).
    """
    chunks = []
    for chunk in full_mask.split(chunk_size, dim=1):
        last_row = chunk[0, -1, :]
        valid_indices = (last_row != min_value).nonzero(as_tuple=False).squeeze(-1)
        last_valid = valid_indices[-1].item() if valid_indices.numel() > 0 else -1
        valid_part = chunk[:, :, : last_valid + 1]
        pad = torch.full(
            (chunk.shape[0], chunk.shape[1], max_kvcache_len - last_valid - 1),
            min_value,
            dtype=chunk.dtype,
            device=chunk.device,
        )
        chunks.append(torch.cat([pad, valid_part], dim=-1))
    return chunks


def get_decoder_mask(attention_mask, min_value=-32768):
    zeros = torch.zeros((1, 1, 1, 1), device=attention_mask.device)
    new_mask = torch.cat([attention_mask, zeros], dim=-1)[:, :, :, 1:]
    return new_mask, torch.where(new_mask == 1, min_value, 0)


def get_paded_input_ids_attn_mask(input_ids, attention_mask, max_lm_input_len, pad_token_id=0):
    seq_len = input_ids.shape[1]
    if seq_len > max_lm_input_len:
        logger.warning(
            f"input_ids length ({seq_len}) exceeds max_lm_input_len ({max_lm_input_len}), "
            f"left-truncating {seq_len - max_lm_input_len} tokens. "
            f"This will cause information loss and corrupt VLM image embeddings. "
            f"Increase max_lm_input_len in your config to at least {seq_len}."
        )
    attention_mask = padding_data(attention_mask, max_lm_input_len, left=True)
    paded_input_ids = padding_data(input_ids, max_lm_input_len, left=True, pad_value=pad_token_id)
    return paded_input_ids, attention_mask


def is_finished(next_tokens, eos_token_id):
    return next_tokens[0] == eos_token_id


def chunk_visual_forward(vit_input_tensor, vit_seq_len, vit_inference_func, chunk_dim=0):
    """Run visual encoder on each image independently and concatenate results.

    Args:
        vit_input_tensor: All images' patch tokens.
        vit_seq_len: Patch token count per image, used to split along chunk_dim.
        vit_inference_func: Callable that takes one image chunk and returns embeddings.
        chunk_dim: Dimension along which to split vit_input_tensor.

    Returns:
        Concatenated image embeddings along dim-0.
    """
    image_embeds = None
    for chunk in vit_input_tensor.split(vit_seq_len, dim=chunk_dim):
        embeds = vit_inference_func(chunk)
        image_embeds = embeds if image_embeds is None else torch.cat([image_embeds, embeds], dim=0)
    return image_embeds


def chunk_prefill_forward(
    chunks_list,
    prefill_cache_keys,
    prefill_cache_values,
    chunk_size,
    prefill_inference_func,
):
    """Run prefill model chunk-by-chunk with rolling KV-cache update.

    Iterates over pre-split input chunks and updates the KV-cache after each
    step. All model-specific forward logic is delegated to
    ``prefill_inference_func``.

    Args:
        chunks_list: List of dicts, each mapping forward argument names to
            the corresponding chunk tensor (e.g. ``{"input_embeddings": ...,
            "position_ids": ..., "attention_mask": ...}``).
        prefill_cache_keys: List of KV-cache key tensors (one per layer).
        prefill_cache_values: List of KV-cache value tensors (one per layer).
        chunk_size: Number of tokens per chunk, used for rolling the KV-cache.
        prefill_inference_func: Callable invoked as
            ``prefill_inference_func(**chunk_inputs, caches=prefill_caches)``
            where ``chunk_inputs`` is a dict from ``chunks_list``.

    Returns:
        Tuple of (next_token_logits, prefill_cache_keys, prefill_cache_values).
    """
    prefill_caches = prefill_cache_keys + prefill_cache_values
    for chunk_inputs in chunks_list:
        next_token_logits, new_keys, new_values = prefill_inference_func(**chunk_inputs, caches=prefill_caches)

        for i in range(len(prefill_cache_keys)):
            old = prefill_cache_keys[i]
            prefill_cache_keys[i] = torch.concatenate([old[:, chunk_size:], new_keys[i]], dim=1)
        for i in range(len(prefill_cache_values)):
            old = prefill_cache_values[i]
            prefill_cache_values[i] = torch.concatenate([old[:, chunk_size:], new_values[i]], dim=1)
        prefill_caches = prefill_cache_keys + prefill_cache_values

    return next_token_logits, prefill_cache_keys, prefill_cache_values
