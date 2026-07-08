"""Generate logic for Qwen3Moe - pure LLM, aligned with qwen3 process_utils."""

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
    padding_data,
    process_kv_cache,
)
from llm_compression.models.logits_process import (
    topk_logits_process,
)


def logits_process(scores, top_k, filter_value=float("-inf")):
    scores = topk_logits_process(scores, top_k, filter_value)
    return scores


def generate_func(
    config,
    prefill_model,
    decode_model,
    input_ids,
    attention_mask,
    do_sample,
    chunk_prefill=False,
):
    with torch.inference_mode():
        prefill_device = next(prefill_model.parameters()).device
        decode_device = next(decode_model.parameters()).device

        num_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
        num_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        max_kvcache_len = config.max_kvcache_len
        max_lm_input_len = config.max_lm_input_len
        chunk_size = max_lm_input_len

        num_valid_tokens = len(input_ids[0])

        pad_token_id = getattr(config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(config, "eos_token_id", 0)

        max_lm_input_len = align_prefill_length(num_valid_tokens, chunk_size, max_kvcache_len, chunk_prefill)

        paded_input_ids, attention_mask = get_paded_input_ids_attn_mask(
            input_ids, attention_mask, max_lm_input_len, pad_token_id=pad_token_id
        )
        paded_input_ids = paded_input_ids.to(prefill_device)
        attention_mask = attention_mask.to(prefill_device)
        emb_dtype = prefill_model.embed_tokens.weight.dtype

        bs = paded_input_ids.shape[0]
        prefill_cache_keys, prefill_cache_values = init_prefill_kv_cache(
            bs,
            num_layers,
            num_key_value_heads,
            head_dim,
            attention_mask,
            max_kvcache_len,
            dtype=emb_dtype,
        )
        prefill_caches = prefill_cache_keys + prefill_cache_values

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        causal_mask = get_causal_mask(attention_mask, max_kvcache_len).squeeze(0).to(emb_dtype)
        if chunk_prefill:
            input_ids_chunks = paded_input_ids.split(chunk_size, dim=1)
            pos_chunks = position_ids.split(chunk_size, dim=1)
            causal_mask_chunks = get_causal_mask_chunks(causal_mask, max_kvcache_len, chunk_size)

            chunks_list = [
                {
                    "input_token_ids": input_ids_chunks[idx],
                    "position_ids": pos_chunks[idx],
                    "attention_mask": causal_mask_chunks[idx],
                }
                for idx in range(len(input_ids_chunks))
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
                input_token_ids=paded_input_ids,
                position_ids=position_ids,
                attention_mask=causal_mask,
                caches=prefill_caches,
            )

        # ── decode ──
        cache_keys, cache_values, attention_mask = init_kv_cache(
            new_keys, new_values, attention_mask, num_valid_tokens, max_kvcache_len
        )

        cache_position = torch.tensor([[num_valid_tokens]], device=decode_device, dtype=torch.int32)
        attention_mask = padding_data(attention_mask, max_kvcache_len)
        decoder_mask = 1 - attention_mask.view(1, 1, 1, -1)

        decode_emb_dtype = decode_model.embed_tokens.weight.dtype
        next_token_logits = next_token_logits.to(decode_device)
        cache_position = cache_position.to(decode_device)
        decoder_mask = decoder_mask.to(decode_device)

        eos_token_id = config.eos_token_id
        max_decode_len = max_kvcache_len - max_lm_input_len
        max_new_tokens = getattr(config, "max_new_tokens", None)
        if max_new_tokens is not None:
            max_decode_len = min(max_decode_len, max_new_tokens)
        return_ids = input_ids.clone().to(decode_device)
        idx = 0

        while True:
            if do_sample:
                sampling_kwargs = dict(top_k=50)
                scores = logits_process(next_token_logits, **sampling_kwargs)
                probs = torch.nn.functional.softmax(scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            idx += 1
            return_ids = torch.cat([return_ids, next_tokens[:, None]], dim=-1)
            if is_finished(next_tokens, eos_token_id) or idx >= max_decode_len:
                break

            new_input_id = next_tokens.view(1, 1)
            position_ids = cache_position.view(1, 1).long()
            caches = cache_keys + cache_values
            decoder_mask, new_decoder_attention_mask = get_decoder_mask(decoder_mask)
            new_decoder_attention_mask = new_decoder_attention_mask.squeeze(0).to(decode_emb_dtype)

            next_token_logits, new_keys, new_values = decode_model.forward(
                input_token_ids=new_input_id,
                position_ids=position_ids,
                attention_mask=new_decoder_attention_mask,
                caches=caches,
            )
            next_token_logits = next_token_logits.to(decode_device)
            cache_keys, cache_values = process_kv_cache(cache_keys, cache_values, new_keys, new_values)
            cache_position = cache_position + 1

        return_ids = return_ids[:, num_valid_tokens:]
        return return_ids
