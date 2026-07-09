import torch

from llm_compression.models.generate_utils import (
    align_prefill_length,
    chunk_prefill_forward,
    chunk_visual_forward,
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
    temperature_logits_process,
    topk_logits_process,
    topp_logits_process,
)


def logits_process(scores, temperature, top_k, top_p, min_tokens_to_keep=1, filter_value=float("-inf")):
    scores = temperature_logits_process(scores, temperature)
    scores = topk_logits_process(scores, top_k, filter_value)
    scores = topp_logits_process(scores, top_p, min_tokens_to_keep, filter_value)
    return scores


def gen_inputs_embeds(config, input_ids, inputs_embeds, image_embeds):
    img_context_token_id = config.img_context_token_id
    image_mask = input_ids == img_context_token_id
    image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    return inputs_embeds


def generate_func(
    config,
    input_embeddings,
    visual_model,
    prefill_model,
    decode_model,
    input_ids,
    pixel_values,
    attention_mask,
    do_sample,
    chunk_prefill=False,
):
    with torch.inference_mode():
        visual_device = next(visual_model.parameters()).device
        prefill_device = next(prefill_model.parameters()).device
        decode_device = next(decode_model.parameters()).device

        # Step 1: Extract image features
        chunk_size = config.llm_config.max_lm_input_len
        num_valid_tokens = len(input_ids[0])
        image_embeds = None
        if pixel_values is not None:
            pixel_values = pixel_values.to(visual_device)
            if chunk_prefill:
                image_embeds = chunk_visual_forward(
                    pixel_values,
                    1,  # process 1 tile at a time
                    visual_model.forward,
                    chunk_dim=0,
                )
            else:
                image_embeds = visual_model(pixel_values)

        # Step 2: Prepare inputs for prefill
        max_lm_input_len = align_prefill_length(
            num_valid_tokens, chunk_size, config.llm_config.max_kvcache_len, chunk_prefill
        )

        paded_input_ids, attention_mask = get_paded_input_ids_attn_mask(input_ids, attention_mask, max_lm_input_len)
        paded_input_ids = paded_input_ids.to(prefill_device)
        attention_mask = attention_mask.to(prefill_device)
        input_embeddings = input_embeddings.to(prefill_device)
        inputs_embeds = input_embeddings(paded_input_ids)

        if image_embeds is not None:
            inputs_embeds = gen_inputs_embeds(config, paded_input_ids, inputs_embeds, image_embeds)

        inputs_embeds = inputs_embeds.to(prefill_device)

        # VLM: access config via config.llm_config
        head_dim = getattr(
            config.llm_config, "head_dim", config.llm_config.hidden_size // config.llm_config.num_attention_heads
        )
        num_layers = config.llm_config.num_hidden_layers
        num_key_value_heads = config.llm_config.num_key_value_heads
        max_kvcache_len = config.llm_config.max_kvcache_len
        bs = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        prefill_cache_keys, prefill_cache_values = init_prefill_kv_cache(
            bs,
            num_layers,
            num_key_value_heads,
            head_dim,
            attention_mask,
            max_kvcache_len,
            dtype=inputs_embeds.dtype,
        )
        prefill_caches = prefill_cache_keys + prefill_cache_values

        causal_mask = get_causal_mask(attention_mask, max_kvcache_len).squeeze(0).to(inputs_embeds.dtype)
        pad_len = seq_len - num_valid_tokens
        position_ids = torch.cat([
            torch.zeros(1, pad_len, device=prefill_device, dtype=torch.long),
            torch.arange(num_valid_tokens, device=prefill_device, dtype=torch.long).unsqueeze(0),
        ], dim=1)

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
                attention_mask=causal_mask,
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                caches=prefill_caches,
            )

        # Step 3: Decode loop
        num_valid_tokens = int(attention_mask.sum(dim=1).item())
        cache_keys, cache_values, attention_mask = init_kv_cache(
            new_keys, new_values, attention_mask, num_valid_tokens, max_kvcache_len
        )

        cache_position = num_valid_tokens
        attention_mask = padding_data(attention_mask, max_kvcache_len)
        decoder_mask = 1 - attention_mask.view(1, 1, 1, -1)

        next_token_logits = next_token_logits.to(decode_device)
        input_embeddings = decode_model.get_input_embeddings().to(decode_device)
        decoder_mask = decoder_mask.to(decode_device)
        cache_keys = [t.to(decode_device) for t in cache_keys]
        cache_values = [t.to(decode_device) for t in cache_values]

        eos_token_id = config.llm_config.eos_token_id
        return_ids = input_ids.clone()
        return_ids = return_ids.to(decode_device)

        idx = 0
        max_decode_len = max_kvcache_len - seq_len
        max_new_tokens = getattr(config.llm_config, "max_new_tokens", None)
        if max_new_tokens is not None:
            max_decode_len = min(max_decode_len, max_new_tokens)
        while True:
            if do_sample:
                sampling_kwargs = dict(temperature=1.0, top_k=20, top_p=0.95)
                scores = logits_process(next_token_logits, **sampling_kwargs)
                probs = torch.nn.functional.softmax(scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            idx += 1
            return_ids = torch.cat([return_ids, next_tokens[:, None]], dim=-1)

            if is_finished(next_tokens, eos_token_id) or idx > max_decode_len:
                break

            new_input_id = next_tokens.view(1, 1)
            inputs_embeds = input_embeddings(new_input_id)
            position_ids = torch.tensor([[cache_position]], device=decode_device, dtype=torch.long)
            caches = cache_keys + cache_values
            decoder_mask, new_decoder_attention_mask = get_decoder_mask(decoder_mask)
            new_decoder_attention_mask = new_decoder_attention_mask.squeeze(0).to(inputs_embeds.dtype)

            next_token_logits, new_keys, new_values = decode_model.forward(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=new_decoder_attention_mask,
                caches=caches,
            )
            cache_keys, cache_values = process_kv_cache(cache_keys, cache_values, new_keys, new_values)
            cache_position = cache_position + 1

        return_ids = return_ids[:, num_valid_tokens:]
        return return_ids
