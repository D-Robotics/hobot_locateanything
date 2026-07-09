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
    repetition_penalty_logits_process,
)


def get_rope_index(
    config,
    input_ids,
    image_grid_thw,
    attention_mask=None,
):
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
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
    image_index = 0
    attention_mask = attention_mask.to(total_input_ids.device)
    for i, input_ids in enumerate(total_input_ids):
        input_ids = input_ids[attention_mask[i] == 1]
        image_nums = 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
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
            second_per_grid_t = 0
            image_index += 1
            remain_images -= 1
            ed = ed_image
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            range_tensor = torch.arange(llm_grid_t).view(-1, 1)
            expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

            second_per_grid_t = torch.as_tensor(second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device)

            time_tensor = expanded_range * second_per_grid_t * config.vision_config.tokens_per_second

            time_tensor_long = time_tensor.long()
            t_index = time_tensor_long.flatten()

            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
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


def logits_process(
    scores,
    temperature,
    top_k,
    top_p,
    min_tokens_to_keep=1,
    filter_value=float("-inf"),
):
    """
    Process logits with various techniques.
    """
    scores = temperature_logits_process(scores, temperature)
    scores = topk_logits_process(scores, top_k, filter_value)
    scores = topp_logits_process(scores, top_p, min_tokens_to_keep, filter_value)
    return scores


def remove_repeat(config, pixel_values):
    patch_size = config.vision_config.patch_size
    tokens = pixel_values.shape[0]
    pixel_values = pixel_values.reshape([-1, 3, 2, patch_size, patch_size])
    pixel_values = pixel_values[:, :, 0]
    pixel_values = pixel_values.reshape(tokens, -1)
    return pixel_values


def gen_inputs_embeds(config, input_ids, inputs_embeds, image_embeds, window_index):
    reverse_indices = torch.argsort(window_index)
    image_embeds = image_embeds[:, reverse_indices, :]

    image_mask = input_ids == config.image_token_id
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
    image_grid_thw,
    attention_mask,
    do_sample,
    chunk_prefill,
):
    with torch.inference_mode():
        visual_device = next(visual_model.parameters()).device
        prefill_device = next(prefill_model.parameters()).device
        decode_device = next(decode_model.parameters()).device

        # step 1: process visual model forward and get image embeddings
        pixel_values = remove_repeat(config, pixel_values)
        pixel_values = pixel_values.unsqueeze(0).to(visual_device)

        chunk_size = config.text_config.max_lm_input_len
        num_valid_tokens = len(input_ids[0])
        if chunk_prefill:
            vit_seq_len = image_grid_thw[0][-1] * image_grid_thw[0][-2]
            image_embeds = chunk_visual_forward(pixel_values, vit_seq_len, visual_model.forward, chunk_dim=1)
            max_lm_input_len = align_prefill_length(
                num_valid_tokens, chunk_size, config.text_config.max_kvcache_len, chunk_prefill
            )
        else:
            image_embeds = visual_model.forward(pixel_values)
            max_lm_input_len = chunk_size

        # step 2: process prefill model forward and get output logits and kvcache
        paded_input_ids, attention_mask = get_paded_input_ids_attn_mask(input_ids, attention_mask, max_lm_input_len)
        paded_input_ids = paded_input_ids.to(prefill_device)
        attention_mask = attention_mask.to(prefill_device)
        inputs_embeds = input_embeddings(paded_input_ids.to(input_embeddings.weight.device)).to(prefill_device)

        # only need the first image grid
        window_index, _ = visual_model.get_window_index(image_grid_thw[:1])

        inputs_embeds = gen_inputs_embeds(config, paded_input_ids, inputs_embeds, image_embeds, window_index).to(
            prefill_device
        )

        num_heads = config.text_config.num_attention_heads
        head_dim = config.text_config.hidden_size // num_heads
        num_layers = config.text_config.num_hidden_layers
        num_key_value_heads = config.text_config.num_key_value_heads
        max_kvcache_len = config.text_config.max_kvcache_len

        bs = inputs_embeds.shape[0]
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

        causal_mask = get_causal_mask(attention_mask, max_kvcache_len).squeeze(0).to(image_embeds.dtype)
        position_ids, mrope_position_deltas = get_rope_index(config, paded_input_ids, image_grid_thw, attention_mask)
        position_ids = position_ids.squeeze().unsqueeze(0).to(prefill_device)

        if chunk_prefill:
            embeds_chunks = inputs_embeds.split(chunk_size, dim=1)
            pos_chunks = position_ids.split(chunk_size, dim=2)
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

        # step 3: process decode model forward and get final output logits
        cache_keys, cache_values, attention_mask = init_kv_cache(
            new_keys, new_values, attention_mask, num_valid_tokens, max_kvcache_len
        )

        cache_position = len(paded_input_ids[0]) + mrope_position_deltas
        attention_mask = padding_data(attention_mask, max_kvcache_len)
        decoder_mask = 1 - attention_mask.view(1, 1, 1, -1)

        next_token_logits = next_token_logits.to(decode_device)
        cache_position = cache_position.to(decode_device)
        decoder_mask = decoder_mask.to(decode_device)
        cache_keys = [t.to(decode_device) for t in cache_keys]
        cache_values = [t.to(decode_device) for t in cache_values]
        idx = 0

        eos_token_id = config.eos_token_id
        max_decode_len = max_kvcache_len - len(input_ids[0])
        max_new_tokens = getattr(config.text_config, "max_new_tokens", None)
        if max_new_tokens is not None:
            max_decode_len = min(max_decode_len, max_new_tokens)
        return_ids = input_ids.clone()
        return_ids = return_ids.to(decode_device)

        while True:
            if do_sample:
                sampling_kwargs = dict(
                    temperature=0.01,
                    top_k=1,
                    top_p=0.001,
                )
                scores = logits_process(next_token_logits, **sampling_kwargs)
                probs = torch.nn.functional.softmax(scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Align with HF generate(): penalise repeated tokens before argmax
                next_token_logits = repetition_penalty_logits_process(
                    next_token_logits, return_ids, 1.05
                )
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            idx += 1
            return_ids = torch.cat([return_ids, next_tokens[:, None]], dim=-1)
            if is_finished(next_tokens, eos_token_id) or idx >= max_decode_len:
                break
            new_input_id = next_tokens.view(1, 1)
            inputs_embeds = decode_model.get_input_embeddings()(new_input_id)
            position_ids = cache_position
            position_ids = position_ids.view(1, 1, 1)
            caches = cache_keys + cache_values
            decoder_mask, new_decoder_attention_mask = get_decoder_mask(decoder_mask)
            new_decoder_attention_mask = new_decoder_attention_mask.to(image_embeds.dtype)
            new_decoder_attention_mask = new_decoder_attention_mask.squeeze(0).squeeze(0)
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
