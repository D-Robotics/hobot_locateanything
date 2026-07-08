"""Generate logic for Gemma4 - pure LLM."""

import copy

import torch

from llm_compression.models.generate_utils import (
    align_prefill_length,
    chunk_prefill_forward,
    get_causal_mask,
    get_causal_mask_chunks,
    get_decoder_mask,
    get_paded_input_ids_attn_mask,
    is_finished,
    padding_data,
    process_kv_cache,
)
from llm_compression.models.logits_process import (
    temperature_logits_process,
    topk_logits_process,
    topp_logits_process,
)
from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)


def _init_kv_cache_heterogeneous(
    cache_keys,
    cache_values,
    attention_mask,
    num_valid_tokens,
    max_kvcache_len,
    layer_types=None,
    sliding_window=None,
):
    """Per-layer KV cache init for heterogeneous head_dim / num_kv_heads."""
    num_layers = len(cache_keys)
    for idx in range(num_layers):
        is_sliding = layer_types is not None and sliding_window is not None and layer_types[idx] == "sliding_attention"
        keep = min(num_valid_tokens, sliding_window) if is_sliding else num_valid_tokens
        cache_keys[idx] = cache_keys[idx][:, -keep:]
        cache_values[idx] = cache_values[idx][:, -keep:]
    attention_mask = attention_mask[:, -num_valid_tokens:]

    for idx in range(num_layers):
        is_sliding = layer_types is not None and sliding_window is not None and layer_types[idx] == "sliding_attention"
        target_len = sliding_window if is_sliding else max_kvcache_len
        bs, cur_tokens, num_heads, embed_dim = cache_keys[idx].shape
        pad_tokens = target_len - cur_tokens
        if pad_tokens > 0:
            pad = torch.zeros(
                bs,
                pad_tokens,
                num_heads,
                embed_dim,
                device=cache_keys[idx].device,
                dtype=cache_keys[idx].dtype,
            )
            cache_keys[idx] = torch.cat([pad, cache_keys[idx]], dim=1)
            cache_values[idx] = torch.cat([pad.clone(), cache_values[idx]], dim=1)
    return cache_keys, cache_values, attention_mask


def logits_process(scores, temperature, top_k, top_p, min_tokens_to_keep=1, filter_value=float("-inf")):
    scores = temperature_logits_process(scores, temperature)
    scores = topk_logits_process(scores, top_k, filter_value)
    scores = topp_logits_process(scores, top_p, min_tokens_to_keep, filter_value)
    return scores


def build_sliding_attention_mask(attention_mask, sliding_window, position_ids=None):
    """Build sliding-window mask from a full causal mask."""
    if sliding_window is None:
        return attention_mask

    if position_ids is None:
        q_len = attention_mask.shape[-2]
        if q_len <= sliding_window:
            return attention_mask
        min_val = attention_mask.min()
        kv_len = attention_mask.shape[-1]
        window_block = torch.tril(
            torch.ones(q_len, kv_len, device=attention_mask.device, dtype=torch.bool),
            diagonal=-sliding_window,
        ).bool()
        return torch.where(window_block, min_val, attention_mask)

    q_positions = position_ids.to(device=attention_mask.device)
    if q_positions.ndim == 1:
        q_positions = q_positions.unsqueeze(0)

    q_len = attention_mask.shape[-2]
    kv_len = attention_mask.shape[-1]
    min_val = attention_mask.min()
    seen_tokens = q_positions.max(dim=-1).values + 1
    left_pad = kv_len - seen_tokens
    kv_positions = (
        torch.arange(kv_len, device=attention_mask.device)
        .view(1, 1, kv_len)
        .expand(q_positions.shape[0], q_len, kv_len)
    )
    kv_positions = kv_positions - left_pad.view(-1, 1, 1)
    window_block = kv_positions <= (q_positions.unsqueeze(-1) - sliding_window)
    if attention_mask.ndim == 4:
        window_block = window_block.unsqueeze(1)
    return torch.where(window_block, min_val, attention_mask)


def _check_eos(next_tokens, eos_token_id):
    """Check if generation should stop. Handles list-type eos_token_id."""
    if isinstance(eos_token_id, (list, tuple)):
        return any(is_finished(next_tokens, eid) for eid in eos_token_id)
    return is_finished(next_tokens, eos_token_id)


def _build_left_padded_position_ids(attention_mask):
    """Build position ids for left-padded inputs."""
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.clamp_(min=0)


def _build_incremental_causal_mask(
    num_cached,
    num_new,
    max_kvcache_len,
    dtype=torch.float32,
    device=None,
    min_value=-32768.0,
):
    """Causal mask for chunk/incremental prefill with right-aligned KV cache."""
    mask = torch.full(
        (num_new, max_kvcache_len),
        min_value,
        dtype=dtype,
        device=device,
    )
    cache_start = max_kvcache_len - num_cached - num_new
    cache_end = max_kvcache_len - num_new
    if num_cached > 0 and cache_start >= 0:
        mask[:, cache_start:cache_end] = 0

    new_start = max_kvcache_len - num_new
    causal = torch.tril(torch.ones(num_new, num_new, dtype=dtype, device=device))
    mask[:, new_start:] = torch.where(
        causal == 1,
        torch.tensor(0, dtype=dtype, device=device),
        torch.tensor(min_value, dtype=dtype, device=device),
    )
    return mask.unsqueeze(0)


def _build_slide_chunk_masks(
    embeds_chunks,
    pos_chunks,
    attn_mask_chunks,
    sliding_window,
    chunk_size,
    emb_dtype,
    device,
    mask_min_value,
):
    """Build per-chunk sliding-window masks for chunk prefill.

    Each chunk gets a mask whose KV dimension equals sliding_window (or
    chunk_size when chunk >= window). Padding positions in new-token
    columns are masked for valid queries to avoid softmax distortion.
    """
    slide_masks = []
    for ci, ec in enumerate(embeds_chunks):
        cur_cs = ec.shape[1]
        if cur_cs >= sliding_window:
            chunk_kv_len = cur_cs
            visible_old = 0
        else:
            chunk_kv_len = sliding_window
            valid_in_cache = min(ci * chunk_size, sliding_window)
            visible_old = min(valid_in_cache, sliding_window - cur_cs)

        chunk_mask = _build_incremental_causal_mask(
            visible_old,
            cur_cs,
            chunk_kv_len,
            dtype=emb_dtype,
            device=device,
            min_value=mask_min_value,
        )
        if cur_cs > sliding_window:
            chunk_mask = build_sliding_attention_mask(
                chunk_mask,
                sliding_window,
                position_ids=pos_chunks[ci],
            )

        # Mask padding KV positions for valid queries
        chunk_attn = attn_mask_chunks[ci]
        new_start = chunk_kv_len - cur_cs
        kv_is_pad = (1 - chunk_attn).bool()
        q_is_valid = chunk_attn.bool()
        pad_block = q_is_valid.unsqueeze(-1) & kv_is_pad.unsqueeze(1)
        chunk_mask[:, :, new_start:] = torch.where(
            pad_block,
            torch.tensor(mask_min_value, dtype=emb_dtype, device=device),
            chunk_mask[:, :, new_start:],
        )
        slide_masks.append(chunk_mask)
    return slide_masks


def generate_func(
    config,
    prefill_model,
    decode_model,
    input_ids,
    attention_mask,
    do_sample,
    chunk_prefill=False,
):
    with torch.no_grad():
        num_layers = config.num_hidden_layers
        max_kvcache_len = config.max_kvcache_len
        max_lm_input_len = config.max_lm_input_len
        chunk_size = max_lm_input_len
        num_valid_tokens = len(input_ids[0])
        pad_token_id = getattr(config, "pad_token_id", None)
        mask_min_value = -32768.0

        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is not None:
            sliding_window = min(sliding_window, max_kvcache_len)

        prefill_device = next(prefill_model.parameters()).device
        emb_dtype = prefill_model.embed_tokens.weight.dtype

        # ── prefill ──
        max_lm_input_len = align_prefill_length(num_valid_tokens, chunk_size, max_kvcache_len, chunk_prefill)

        paded_input_ids, attention_mask = get_paded_input_ids_attn_mask(
            input_ids, attention_mask, max_lm_input_len, pad_token_id=pad_token_id
        )
        paded_input_ids = paded_input_ids.to(prefill_device)
        attention_mask = attention_mask.to(prefill_device)
        inputs_embeds = prefill_model.embed_tokens(paded_input_ids)
        inputs_embeds[:, :-num_valid_tokens, :] = 0

        bs = inputs_embeds.shape[0]
        layer_types = config.layer_types
        head_dim_sliding = config.head_dim
        head_dim_full = getattr(config, "global_head_dim", config.head_dim)
        num_kv_heads_sliding = config.num_key_value_heads
        num_kv_heads_full = getattr(config, "num_global_key_value_heads", config.num_key_value_heads)

        prefill_cache_keys = []
        prefill_cache_values = []
        for i in range(num_layers):
            if layer_types[i] == "sliding_attention":
                hd, nkv = head_dim_sliding, num_kv_heads_sliding
                cache_len = sliding_window if sliding_window else max_kvcache_len
            else:
                hd, nkv = head_dim_full, num_kv_heads_full
                cache_len = max_kvcache_len
            zeros = torch.zeros(
                bs,
                cache_len,
                nkv,
                hd,
                device=prefill_device,
                dtype=emb_dtype,
            )
            prefill_cache_keys.append(zeros.clone())
            prefill_cache_values.append(zeros.clone())

        prefill_caches = prefill_cache_keys + prefill_cache_values
        position_ids = _build_left_padded_position_ids(attention_mask).to(device=prefill_device)
        causal_mask = (
            get_causal_mask(attention_mask, max_kvcache_len, min_value=mask_min_value).squeeze(0).to(emb_dtype)
        )

        if sliding_window:
            slide_kv_len = max(max_lm_input_len, sliding_window)
            slide_causal_base = (
                get_causal_mask(attention_mask, slide_kv_len, min_value=mask_min_value).squeeze(0).to(emb_dtype)
            )
            slide_attention_mask = build_sliding_attention_mask(
                slide_causal_base,
                sliding_window,
                position_ids=position_ids,
            )
        else:
            slide_attention_mask = causal_mask

        if chunk_prefill:
            embeds_chunks = inputs_embeds.split(chunk_size, dim=1)
            pos_chunks = position_ids.split(chunk_size, dim=1)
            causal_mask_chunks = get_causal_mask_chunks(
                causal_mask, max_kvcache_len, chunk_size, min_value=mask_min_value
            )
            if sliding_window and sliding_window < max_kvcache_len:
                attn_mask_chunks = attention_mask.split(chunk_size, dim=1)
                slide_attention_mask_chunks = _build_slide_chunk_masks(
                    embeds_chunks,
                    pos_chunks,
                    attn_mask_chunks,
                    sliding_window,
                    chunk_size,
                    emb_dtype,
                    prefill_device,
                    mask_min_value,
                )
            else:
                slide_attention_mask_chunks = get_causal_mask_chunks(
                    slide_attention_mask,
                    max_kvcache_len,
                    chunk_size,
                    min_value=mask_min_value,
                )

            chunks_list = [
                {
                    "input_embeddings": embeds_chunks[idx],
                    "position_ids": pos_chunks[idx],
                    "attention_mask": causal_mask_chunks[idx],
                    "slide_attention_mask": slide_attention_mask_chunks[idx],
                }
                for idx in range(len(embeds_chunks))
            ]
            next_token_logits, new_keys, new_values = chunk_prefill_forward(
                chunks_list,
                prefill_cache_keys,
                prefill_cache_values,
                chunk_size,
                prefill_model.forward,
            )
        else:
            next_token_logits, new_keys, new_values = prefill_model.forward(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=causal_mask,
                slide_attention_mask=slide_attention_mask,
                caches=prefill_caches,
            )

        # ── decode ──
        layer_types = config.layer_types
        cache_keys, cache_values, attention_mask = _init_kv_cache_heterogeneous(
            new_keys,
            new_values,
            attention_mask,
            num_valid_tokens,
            max_kvcache_len,
            layer_types=layer_types,
            sliding_window=sliding_window,
        )

        decode_device = next(decode_model.parameters()).device
        cache_keys = [k.to(decode_device) for k in cache_keys]
        cache_values = [v.to(decode_device) for v in cache_values]

        cache_position = torch.tensor([[num_valid_tokens]], device=decode_device, dtype=torch.int32)

        attention_mask = padding_data(attention_mask, max_kvcache_len)
        decoder_mask_full = 1 - attention_mask.view(1, 1, 1, -1)
        if sliding_window:
            slide_valid = min(num_valid_tokens, sliding_window)
            slide_attn = torch.zeros(1, sliding_window, device=attention_mask.device)
            slide_attn[:, sliding_window - slide_valid :] = 1
            decoder_mask_slide = 1 - slide_attn.view(1, 1, 1, -1)
        else:
            decoder_mask_slide = decoder_mask_full

        decode_emb_dtype = decode_model.embed_tokens.weight.dtype
        decode_input_embeddings = decode_model.embed_tokens
        next_token_logits = next_token_logits.to(device=decode_device)
        cache_position = cache_position.to(device=decode_device)
        decoder_mask_full = decoder_mask_full.to(device=decode_device)
        decoder_mask_slide = decoder_mask_slide.to(device=decode_device)

        eos_token_id = config.eos_token_id
        max_new_tokens = max_kvcache_len - num_valid_tokens
        max_generate_tokens = getattr(config, "max_generate_tokens", None)
        if max_generate_tokens is not None and max_generate_tokens < max_new_tokens:
            max_new_tokens = max_generate_tokens
        return_ids = copy.deepcopy(input_ids).to(decode_device)
        idx = 0

        while True:
            if do_sample:
                sampling_kwargs = dict(temperature=1.0, top_k=64, top_p=0.95)
                scores = logits_process(next_token_logits, **sampling_kwargs)
                probs = torch.nn.functional.softmax(scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            idx += 1
            return_ids = torch.cat([return_ids, next_tokens[:, None]], dim=-1)
            if _check_eos(next_tokens, eos_token_id) or idx >= max_new_tokens:
                break

            new_input_id = next_tokens.view(1, 1)
            inputs_embeds = decode_input_embeddings(new_input_id)
            position_ids = cache_position.view(1, 1).long()
            caches = cache_keys + cache_values

            decoder_mask_full, new_decoder_attention_mask = get_decoder_mask(
                decoder_mask_full, min_value=mask_min_value
            )
            new_decoder_attention_mask = new_decoder_attention_mask.squeeze(0).to(decode_emb_dtype)
            decoder_mask_slide, new_slide_attention_mask = get_decoder_mask(
                decoder_mask_slide, min_value=mask_min_value
            )
            new_slide_attention_mask = new_slide_attention_mask.squeeze(0).to(decode_emb_dtype)

            next_token_logits, new_keys, new_values = decode_model.forward(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=new_decoder_attention_mask,
                slide_attention_mask=new_slide_attention_mask,
                caches=caches,
            )
            next_token_logits = next_token_logits.to(device=decode_device)
            cache_keys, cache_values = process_kv_cache(cache_keys, cache_values, new_keys, new_values)
            cache_position = cache_position + 1

        return_ids = return_ids[:, num_valid_tokens:]
        return return_ids
