import torch

from llm_compression.models.generate_utils import (
    align_prefill_length,
    chunk_visual_forward,
    get_causal_mask,
    get_causal_mask_chunks,
    get_decoder_mask,
    get_paded_input_ids_attn_mask,
    is_finished,
    padding_data,
)
from llm_compression.models.logits_process import (
    temperature_logits_process,
    topk_logits_process,
    topp_logits_process,
)


def logits_process(
    scores,
    temperature,
    top_k,
    top_p,
    min_tokens_to_keep=1,
    filter_value=float("-inf"),
):
    scores = temperature_logits_process(scores, temperature)
    scores = topk_logits_process(scores, top_k, filter_value)
    scores = topp_logits_process(scores, top_p, min_tokens_to_keep, filter_value)
    return scores


def flatten_caches(cache_keys, cache_values, conv_states, recurrent_states):
    return cache_keys + cache_values + conv_states + recurrent_states


def get_rope_index(
    config,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor = None,
):
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    vision_start_token_id = config.vision_start_token_id

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    bsz, total_len = input_ids.shape
    position_ids = torch.ones(3, bsz, total_len, dtype=input_ids.dtype, device=input_ids.device)
    mrope_position_deltas = []

    image_index = 0
    for i, seq_input_ids in enumerate(input_ids):
        valid_ids = seq_input_ids[attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(valid_ids == vision_start_token_id).squeeze(1)
        vision_tokens = valid_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()

        input_tokens = valid_ids.tolist()
        llm_pos_ids_list = []
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
            image_index += 1
            remain_images -= 1

            text_len = ed_image - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

            llm_grid_t = t.item()
            llm_grid_h = h.item() // spatial_merge_size
            llm_grid_w = w.item() // spatial_merge_size

            t_index = (
                torch.arange(llm_grid_t, device=input_ids.device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, device=input_ids.device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=input_ids.device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed_image + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids[i]))

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def gen_inputs_embeds(config, input_ids, inputs_embeds, image_embeds):
    image_mask = input_ids == config.image_token_id
    image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    return inputs_embeds


def init_prefill_mixed_cache(
    bs,
    config,
    device,
    dtype,
):
    cache_keys, cache_values = [], []
    conv_states, recurrent_states = [], []

    max_kvcache_len = config.max_kvcache_len
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    for layer_type in config.layer_types:
        if layer_type == "full_attention":
            zeros = torch.zeros(
                bs,
                max_kvcache_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            cache_keys.append(zeros.clone())
            cache_values.append(zeros.clone())
            conv_states.append(None)
            recurrent_states.append(None)
        else:
            conv_dim = (
                config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim
            )
            cache_keys.append(None)
            cache_values.append(None)
            conv_states.append(
                torch.zeros(
                    bs,
                    conv_dim,
                    config.linear_conv_kernel_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            recurrent_states.append(
                torch.zeros(
                    bs,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    device=device,
                    dtype=dtype,
                )
            )
    return cache_keys, cache_values, conv_states, recurrent_states


def chunk_prefill_forward_mixed(
    chunks_list,
    config,
    prefill_cache_keys,
    prefill_cache_values,
    prefill_conv_states,
    prefill_recurrent_states,
    prefill_inference_func,
):
    prefill_caches = flatten_caches(
        prefill_cache_keys,
        prefill_cache_values,
        prefill_conv_states,
        prefill_recurrent_states,
    )

    for chunk_inputs in zip(*chunks_list):
        (
            next_token_logits,
            new_keys,
            new_values,
            new_conv_states,
            new_recurrent_states,
        ) = prefill_inference_func(*chunk_inputs, prefill_caches)
        for idx, layer_type in enumerate(config.layer_types):
            if layer_type == "full_attention":
                refresh_len = new_keys[idx].shape[1]
                tgt_dev = new_keys[idx].device
                prefill_cache_keys[idx] = torch.cat(
                    [prefill_cache_keys[idx].to(tgt_dev)[:, refresh_len:], new_keys[idx]], dim=1
                )
                prefill_cache_values[idx] = torch.cat(
                    [prefill_cache_values[idx].to(tgt_dev)[:, refresh_len:], new_values[idx]],
                    dim=1,
                )
            else:
                prefill_conv_states[idx] = new_conv_states[idx]
                prefill_recurrent_states[idx] = new_recurrent_states[idx]

        prefill_caches = flatten_caches(
            prefill_cache_keys,
            prefill_cache_values,
            prefill_conv_states,
            prefill_recurrent_states,
        )

    return (
        next_token_logits,
        prefill_cache_keys,
        prefill_cache_values,
        prefill_conv_states,
        prefill_recurrent_states,
    )


def init_decode_mixed_cache(
    config,
    new_keys,
    new_values,
    new_conv_states,
    new_recurrent_states,
    attention_mask,
    num_valid_tokens,
):
    cache_keys, cache_values = [], []
    conv_states, recurrent_states = [], []
    attention_mask = attention_mask[:, -num_valid_tokens:]
    max_kvcache_len = config.max_kvcache_len

    for idx, layer_type in enumerate(config.layer_types):
        if layer_type == "full_attention":
            key_cache = new_keys[idx][:, -num_valid_tokens:]
            value_cache = new_values[idx][:, -num_valid_tokens:]
            bs, cur_tokens, num_heads, embed_dim = key_cache.shape
            pad_tokens = max_kvcache_len - cur_tokens
            pad = torch.zeros(
                bs,
                pad_tokens,
                num_heads,
                embed_dim,
                device=key_cache.device,
                dtype=key_cache.dtype,
            )
            cache_keys.append(torch.cat([pad, key_cache], dim=1))
            cache_values.append(torch.cat([pad.clone(), value_cache], dim=1))
            conv_states.append(None)
            recurrent_states.append(None)
        else:
            cache_keys.append(None)
            cache_values.append(None)
            conv_states.append(new_conv_states[idx])
            recurrent_states.append(new_recurrent_states[idx])
    return cache_keys, cache_values, conv_states, recurrent_states, attention_mask


def process_mixed_cache(
    config,
    cache_keys,
    cache_values,
    conv_states,
    recurrent_states,
    new_keys,
    new_values,
    new_conv_states,
    new_recurrent_states,
):
    for idx, layer_type in enumerate(config.layer_types):
        if layer_type == "full_attention":
            refresh_len = new_keys[idx].shape[1]
            cache_keys[idx] = torch.cat([cache_keys[idx], new_keys[idx]], dim=1)[:, refresh_len:]
            cache_values[idx] = torch.cat([cache_values[idx], new_values[idx]], dim=1)[:, refresh_len:]
        else:
            conv_states[idx] = new_conv_states[idx]
            recurrent_states[idx] = new_recurrent_states[idx]
    return cache_keys, cache_values, conv_states, recurrent_states


def generate_func(
    config,
    input_embeddings,
    visual_model,
    prefill_model,
    decode_model,
    input_ids,
    pixel_values,
    image_grid_thw,
    attention_mask,
    do_sample,
    chunk_prefill=False,
):
    with torch.inference_mode():
        visual_device = next(visual_model.parameters()).device
        prefill_device = next(prefill_model.parameters()).device
        decode_device = next(decode_model.parameters()).device

        max_kvcache_len = config.text_config.max_kvcache_len
        max_lm_input_len = config.text_config.max_lm_input_len
        max_new_tokens = getattr(config.text_config, "max_new_tokens", None)

        pixel_values = pixel_values.unsqueeze(0).to(visual_device)
        image_grid_thw = image_grid_thw.to(visual_device)
        chunk_size = max_lm_input_len
        num_valid_tokens = len(input_ids[0])

        if chunk_prefill:
            vit_seq_len = image_grid_thw[0][-1] * image_grid_thw[0][-2]
            image_embeds = chunk_visual_forward(
                pixel_values,
                vit_seq_len,
                visual_model.forward,
                chunk_dim=1,
            )
            max_lm_input_len = align_prefill_length(num_valid_tokens, chunk_size, max_kvcache_len, chunk_prefill)
        else:
            image_embeds = visual_model.forward(pixel_values)
            max_lm_input_len = chunk_size

        del pixel_values

        pad_token_id = getattr(
            config,
            "pad_token_id",
            getattr(getattr(config, "text_config", None), "pad_token_id", None),
        )
        if pad_token_id is None:
            pad_token_id = getattr(config, "eos_token_id", None) or getattr(
                getattr(config, "text_config", None), "eos_token_id", 0
            )
        paded_input_ids, attention_mask = get_paded_input_ids_attn_mask(
            input_ids,
            attention_mask,
            max_lm_input_len,
            pad_token_id=pad_token_id,
        )
        paded_input_ids = paded_input_ids.to(prefill_device)
        attention_mask = attention_mask.to(prefill_device)
        input_embeddings = input_embeddings.to(prefill_device)
        inputs_embeds = input_embeddings(paded_input_ids)
        inputs_embeds = gen_inputs_embeds(config, paded_input_ids, inputs_embeds, image_embeds).to(prefill_device)

        bs = inputs_embeds.shape[0]
        (
            prefill_cache_keys,
            prefill_cache_values,
            prefill_conv_states,
            prefill_recurrent_states,
        ) = init_prefill_mixed_cache(bs, config.text_config, attention_mask.device, inputs_embeds.dtype)
        position_ids, mrope_position_deltas = get_rope_index(config, paded_input_ids, image_grid_thw, attention_mask)
        position_ids = position_ids.to(prefill_device, dtype=torch.int32)
        full_causal_mask = get_causal_mask(attention_mask, max_kvcache_len).squeeze(1)
        full_causal_mask = full_causal_mask.to(inputs_embeds.dtype)

        if chunk_prefill:
            embeds_chunks = inputs_embeds.split(chunk_size, dim=1)
            pos_chunks = position_ids.split(chunk_size, dim=2)
            linear_mask_chunks = attention_mask.split(chunk_size, dim=1)
            causal_mask_chunks = [
                chunk.unsqueeze(1) for chunk in get_causal_mask_chunks(full_causal_mask, max_kvcache_len, chunk_size)
            ]
            (
                next_token_logits,
                new_keys,
                new_values,
                new_conv_states,
                new_recurrent_states,
            ) = chunk_prefill_forward_mixed(
                [
                    embeds_chunks,
                    pos_chunks,
                    causal_mask_chunks,
                    linear_mask_chunks,
                ],
                config.text_config,
                prefill_cache_keys,
                prefill_cache_values,
                prefill_conv_states,
                prefill_recurrent_states,
                prefill_model.forward,
            )
        else:
            prefill_caches = flatten_caches(
                prefill_cache_keys,
                prefill_cache_values,
                prefill_conv_states,
                prefill_recurrent_states,
            )
            (
                next_token_logits,
                new_keys,
                new_values,
                new_conv_states,
                new_recurrent_states,
            ) = prefill_model.forward(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=full_causal_mask.unsqueeze(1),
                linear_attention_mask=attention_mask,
                caches=prefill_caches,
            )

        (
            cache_keys,
            cache_values,
            conv_states,
            recurrent_states,
            attention_mask,
        ) = init_decode_mixed_cache(
            config.text_config,
            new_keys,
            new_values,
            new_conv_states,
            new_recurrent_states,
            attention_mask,
            num_valid_tokens,
        )

        cache_position = len(paded_input_ids[0]) + mrope_position_deltas  # (bsz, 1)
        attention_mask = padding_data(attention_mask, max_kvcache_len)
        decoder_mask = 1 - attention_mask.view(1, 1, 1, -1)

        next_token_logits = next_token_logits.to(decode_device)
        decode_input_embeddings = decode_model.get_input_embeddings().to(decode_device)
        cache_position = cache_position.to(decode_device)
        decoder_mask = decoder_mask.to(decode_device)
        cache_keys = [t.to(decode_device) if t is not None else t for t in cache_keys]
        cache_values = [t.to(decode_device) if t is not None else t for t in cache_values]
        conv_states = [t.to(decode_device) if t is not None else t for t in conv_states]
        recurrent_states = [t.to(decode_device) if t is not None else t for t in recurrent_states]

        eos_token_id = config.text_config.eos_token_id
        max_decode_len = max_kvcache_len - num_valid_tokens
        if max_new_tokens is not None:
            max_decode_len = min(max_decode_len, max_new_tokens)
        return_ids = input_ids.clone().to(decode_device)
        idx = 0

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
            if is_finished(next_tokens, eos_token_id) or idx >= max_decode_len:
                break

            new_input_id = next_tokens.view(1, 1)
            inputs_embeds = decode_input_embeddings(new_input_id)
            position_ids = cache_position.view(1, 1, 1).expand(3, -1, -1).long()
            caches = flatten_caches(cache_keys, cache_values, conv_states, recurrent_states)
            decoder_mask, new_decoder_attention_mask = get_decoder_mask(decoder_mask)
            new_decoder_attention_mask = new_decoder_attention_mask.to(inputs_embeds.dtype)
            linear_attention_mask = torch.ones(1, 1, device=decode_device, dtype=attention_mask.dtype)

            (
                next_token_logits,
                new_keys,
                new_values,
                new_conv_states,
                new_recurrent_states,
            ) = decode_model.forward(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=new_decoder_attention_mask,
                linear_attention_mask=linear_attention_mask,
                caches=caches,
            )
            (
                cache_keys,
                cache_values,
                conv_states,
                recurrent_states,
            ) = process_mixed_cache(
                config.text_config,
                cache_keys,
                cache_values,
                conv_states,
                recurrent_states,
                new_keys,
                new_values,
                new_conv_states,
                new_recurrent_states,
            )
            cache_position = cache_position + 1

        generated_ids = return_ids[:, num_valid_tokens:]
        return generated_ids
