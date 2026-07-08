import torch

from llm_compression.models.generate_utils import (
    align_prefill_length,
    chunk_prefill_forward,
    get_causal_mask,
    get_causal_mask_chunks,
    get_decoder_mask,
    get_paded_input_ids_attn_mask,
    init_kv_cache,
    init_prefill_kv_cache,
    is_finished,
    process_kv_cache,
)
from llm_compression.models.logits_process import (
    temperature_logits_process,
    topk_logits_process,
    topp_logits_process,
)


def logits_process(scores, temperature, top_k, top_p, min_tokens_to_keep=1, filter_value=float("-inf")):
    scores = temperature_logits_process(scores, temperature)
    scores = topk_logits_process(scores, top_k, filter_value)
    scores = topp_logits_process(scores, top_p, min_tokens_to_keep, filter_value)
    return scores


def prefill_func(config, input_embeddings, prefill_model, decode_model, input_ids, attention_mask, chunk_prefill=False):
    """Run prefill phase: padding, embedding, prefill forward, init KV cache.

    Returns:
        next_token_logits, cache_keys, cache_values, input_ids, num_valid_tokens, input_embeddings
    """
    prefill_device = next(prefill_model.parameters()).device
    decode_device = next(decode_model.parameters()).device

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    max_kvcache_len = config.max_kvcache_len
    chunk_size = config.max_lm_input_len

    num_valid_tokens = len(input_ids[0])
    max_lm_input_len = align_prefill_length(num_valid_tokens, chunk_size, max_kvcache_len, chunk_prefill)

    paded_input_ids, attention_mask = get_paded_input_ids_attn_mask(input_ids, attention_mask, max_lm_input_len)
    paded_input_ids = paded_input_ids.to(prefill_device)
    attention_mask = attention_mask.to(prefill_device)
    input_embeddings = input_embeddings.to(prefill_device)
    inputs_embeds = input_embeddings(paded_input_ids)

    prefill_cache_keys, prefill_cache_values = init_prefill_kv_cache(
        1,
        num_layers,
        num_key_value_heads,
        head_dim,
        attention_mask,
        max_kvcache_len,
        dtype=inputs_embeds.dtype,
    )
    prefill_caches = prefill_cache_keys + prefill_cache_values

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    causal_mask = get_causal_mask(attention_mask, max_kvcache_len).squeeze(0).to(inputs_embeds.dtype)

    if chunk_prefill:
        embeds_chunks = inputs_embeds.split(chunk_size, dim=1)
        pos_chunks = position_ids.split(chunk_size, dim=1)
        causal_mask_chunks = get_causal_mask_chunks(causal_mask, max_kvcache_len, chunk_size)
        chunks_list = [
            {
                "input_embeddings": embeds_chunks[idx],
                "position_ids": pos_chunks[idx],
                "attention_mask": causal_mask_chunks[idx],
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
            caches=prefill_caches,
        )

    cache_keys, cache_values, _ = init_kv_cache(new_keys, new_values, attention_mask, num_valid_tokens, max_kvcache_len)

    next_token_logits = next_token_logits.to(decode_device)
    cache_keys = [t.to(decode_device) for t in cache_keys]
    cache_values = [t.to(decode_device) for t in cache_values]

    return next_token_logits, cache_keys, cache_values, input_ids, num_valid_tokens, input_embeddings


def decode_func(
    config, decode_model, next_token_logits, cache_keys, cache_values, num_valid_tokens, do_sample, chunk_prefill=False
):
    """Run autoregressive decode loop.

    Returns:
        Generated token ids (excluding input).
    """
    decode_device = next(decode_model.parameters()).device
    max_kvcache_len = config.max_kvcache_len
    chunk_size = config.max_lm_input_len
    max_lm_input_len = align_prefill_length(num_valid_tokens, chunk_size, max_kvcache_len, chunk_prefill)

    cache_position = torch.tensor([[num_valid_tokens]], device=decode_device, dtype=torch.int32)
    attention_mask_pad = torch.zeros(1, max_kvcache_len, device=decode_device)
    attention_mask_pad[:, max_kvcache_len - num_valid_tokens :] = 1
    decoder_mask = 1 - attention_mask_pad.view(1, 1, 1, -1)

    decode_input_embeddings = decode_model.get_input_embeddings()

    eos_token_id = config.eos_token_id
    max_decode_len = max_kvcache_len - max_lm_input_len
    max_new_tokens = getattr(config, "max_new_tokens", None)
    if max_new_tokens is not None:
        max_decode_len = min(max_decode_len, max_new_tokens)

    generated_ids = []
    idx = 0

    while True:
        if do_sample:
            scores = logits_process(next_token_logits, temperature=0.6, top_k=20, top_p=0.95)
            probs = torch.nn.functional.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        idx += 1
        generated_ids.append(next_tokens)
        if is_finished(next_tokens, eos_token_id) or idx >= max_decode_len:
            break

        inputs_embeds = decode_input_embeddings(next_tokens.view(1, 1))
        position_ids = cache_position.view(1, 1).long()
        decoder_mask, new_decoder_attention_mask = get_decoder_mask(decoder_mask)
        new_decoder_attention_mask = new_decoder_attention_mask.squeeze(0).squeeze(0).to(inputs_embeds.dtype)

        next_token_logits, new_keys, new_values = decode_model.forward(
            input_embeddings=inputs_embeds,
            position_ids=position_ids,
            attention_mask=new_decoder_attention_mask,
            caches=cache_keys + cache_values,
        )
        cache_keys, cache_values = process_kv_cache(cache_keys, cache_values, new_keys, new_values)
        cache_position = cache_position + 1

    return torch.cat([t[:, None] for t in generated_ids], dim=-1)
