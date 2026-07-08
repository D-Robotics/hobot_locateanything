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
    temperature_logits_process,
    topk_logits_process,
    topp_logits_process,
)


def logits_process(scores, temperature, top_k, top_p, min_tokens_to_keep=1, filter_value=float("-inf")):
    scores = temperature_logits_process(scores, temperature)
    scores = topk_logits_process(scores, top_k, filter_value)
    scores = topp_logits_process(scores, top_p, min_tokens_to_keep, filter_value)
    return scores


def get_rope_index(
    config,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor = None,
):
    """Compute MRoPE 3D position IDs for Qwen3-VL.

    Returns:
        position_ids: (3, bsz, seq_len) — T/H/W indices
        mrope_position_deltas: (bsz, 1)
    """
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
            ed = ed_image
            llm_grid_t = t.item()
            llm_grid_h = h.item() // spatial_merge_size
            llm_grid_w = w.item() // spatial_merge_size

            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # Build T/H/W indices for image tokens
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
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
        mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids[i]))

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def gen_inputs_embeds(config, input_ids, inputs_embeds, image_embeds):
    """Replace image token embeddings with ViT output embeddings."""
    image_mask = input_ids == config.image_token_id
    image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    return inputs_embeds


def scatter_deepstack_embeds(config, input_ids, inputs_embeds, deepstack_feature_lists):
    """Scatter deepstack features into full-sequence tensors at image token positions.

    deepstack_feature_lists: list of (1, num_image_tokens, hidden_size)
    Returns: list of (1, seq_len, hidden_size), each aligned to image token positions.
    """
    image_mask = (input_ids == config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    image_mask = image_mask.to(inputs_embeds.device)
    scattered = []
    for ds_feat in deepstack_feature_lists:
        base = torch.zeros_like(inputs_embeds)
        ds_feat = ds_feat.to(inputs_embeds.device, inputs_embeds.dtype)
        base = base.masked_scatter(image_mask, ds_feat)
        scattered.append(base)
    return scattered


def normalize_visual_outputs(visual_outputs):
    """Normalize visual outputs for native and HBM wrappers."""
    if isinstance(visual_outputs, tuple):
        if len(visual_outputs) < 2:
            return visual_outputs[0], None
        return visual_outputs[0], visual_outputs[1]

    if isinstance(visual_outputs, list):
        if len(visual_outputs) == 0:
            return None, None
        if len(visual_outputs) == 1:
            return visual_outputs[0], None
        return visual_outputs[0], visual_outputs[1:]

    return visual_outputs, None


def chunk_visual_forward_with_deepstack(vit_input_tensor, vit_seq_len, vit_inference_func, chunk_dim=0):
    """Chunked visual forward that also merges deepstack outputs."""
    image_embeds = None
    deepstack_feature_lists = None

    for chunk in vit_input_tensor.split(vit_seq_len, dim=chunk_dim):
        chunk_outputs = vit_inference_func(chunk)
        chunk_image_embeds, chunk_deepstack_feature_lists = normalize_visual_outputs(chunk_outputs)

        image_embeds = (
            chunk_image_embeds if image_embeds is None else torch.cat([image_embeds, chunk_image_embeds], dim=1)
        )

        if chunk_deepstack_feature_lists is None:
            continue
        if deepstack_feature_lists is None:
            deepstack_feature_lists = [feat for feat in chunk_deepstack_feature_lists]
        else:
            deepstack_feature_lists = [
                torch.cat([old_feat, new_feat], dim=1)
                for old_feat, new_feat in zip(deepstack_feature_lists, chunk_deepstack_feature_lists, strict=True)
            ]

    return image_embeds, deepstack_feature_lists


def generate_func(
    config,
    visual_model,
    prefill_model,
    decode_model,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor,
    do_sample: bool,
    chunk_prefill: bool = False,
):
    """Full prefill → decode generation loop for Qwen3-VL."""
    with torch.inference_mode():
        visual_device = next(visual_model.parameters()).device
        prefill_device = next(prefill_model.parameters()).device
        decode_device = next(decode_model.parameters()).device

        # ---- Vision forward ----
        visual_dtype = next(visual_model.parameters()).dtype
        pixel_values = pixel_values.unsqueeze(0).to(visual_device, dtype=visual_dtype)
        image_grid_thw = image_grid_thw.to(visual_device)
        chunk_size = config.text_config.max_lm_input_len
        num_valid_tokens = len(input_ids[0])

        if chunk_prefill:
            vit_seq_len = image_grid_thw[0][-1] * image_grid_thw[0][-2]
            image_embeds, deepstack_feature_lists = chunk_visual_forward_with_deepstack(
                pixel_values,
                vit_seq_len,
                visual_model.forward,
                chunk_dim=1,
            )
            max_lm_input_len = align_prefill_length(
                num_valid_tokens, chunk_size, config.text_config.max_kvcache_len, chunk_prefill
            )
        else:
            image_embeds, deepstack_feature_lists = normalize_visual_outputs(visual_model.forward(pixel_values))
            max_lm_input_len = chunk_size
        # image_embeds: (1, num_merged_patches, text_hidden_size)
        # deepstack_feature_lists: list of (1, num_merged_patches, text_hidden_size)

        # ---- Prefill ----
        paded_input_ids, attention_mask = get_paded_input_ids_attn_mask(input_ids, attention_mask, max_lm_input_len)
        paded_input_ids = paded_input_ids.to(prefill_device)
        attention_mask = attention_mask.to(prefill_device)
        prefill_input_embeddings = prefill_model.get_input_embeddings().to(prefill_device)
        inputs_embeds = prefill_input_embeddings(paded_input_ids)
        inputs_embeds = gen_inputs_embeds(config, paded_input_ids, inputs_embeds, image_embeds).to(prefill_device)
        # scatter deepstack features to full-sequence shape (at image token positions)
        scattered_deepstack = (
            scatter_deepstack_embeds(config, paded_input_ids, inputs_embeds, deepstack_feature_lists)
            if deepstack_feature_lists
            else None
        )

        num_heads = config.text_config.num_attention_heads
        head_dim = getattr(config.text_config, "head_dim", config.text_config.hidden_size // num_heads)
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

        # MRoPE position_ids: (3, bsz, seq_len), convert to int64
        position_ids, mrope_position_deltas = get_rope_index(config, paded_input_ids, image_grid_thw, attention_mask)
        position_ids = position_ids.to(prefill_device, dtype=torch.long)

        causal_mask = get_causal_mask(attention_mask, max_kvcache_len).squeeze(0).to(inputs_embeds.dtype)
        if chunk_prefill:
            embeds_chunks = inputs_embeds.split(chunk_size, dim=1)
            pos_chunks = position_ids.split(chunk_size, dim=2)
            causal_mask_chunks = get_causal_mask_chunks(causal_mask, max_kvcache_len, chunk_size)
            ds_split = [ds.split(chunk_size, dim=1) for ds in scattered_deepstack]
            deepstack_chunks_by_step = [list(t) for t in zip(*ds_split, strict=True)]
            chunks_list = [
                {
                    "input_embeddings": embeds_chunks[idx],
                    "position_ids": pos_chunks[idx],
                    "attention_mask": causal_mask_chunks[idx],
                    "deepstack_visual_embeds": deepstack_chunks_by_step[idx],
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
            prefill_caches = prefill_cache_keys + prefill_cache_values
            next_token_logits, new_keys, new_values = prefill_model.forward(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=causal_mask,
                deepstack_visual_embeds=scattered_deepstack,
                caches=prefill_caches,
            )

        # ---- Init decode KV cache ----
        cache_keys, cache_values, attention_mask = init_kv_cache(
            new_keys, new_values, attention_mask, num_valid_tokens, max_kvcache_len
        )

        cache_position = len(paded_input_ids[0]) + mrope_position_deltas  # (bsz, 1)
        attention_mask = padding_data(attention_mask, max_kvcache_len)
        decoder_mask = 1 - attention_mask.view(1, 1, 1, -1)

        next_token_logits = next_token_logits.to(decode_device)
        decode_input_embeddings = decode_model.get_input_embeddings().to(decode_device)
        cache_position = cache_position.to(decode_device)
        decoder_mask = decoder_mask.to(decode_device)
        cache_keys = [t.to(decode_device) for t in cache_keys]
        cache_values = [t.to(decode_device) for t in cache_values]

        eos_token_id = config.eos_token_id
        max_decode_len = max_kvcache_len - max_lm_input_len
        max_new_tokens = getattr(config.text_config, "max_new_tokens", None)
        if max_new_tokens is not None:
            max_decode_len = min(max_decode_len, max_new_tokens)
        return_ids = input_ids.clone().to(decode_device)
        idx = 0

        while True:
            if do_sample:
                sampling_kwargs = dict(temperature=0.7, top_k=20, top_p=0.8)
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

            decode_position_ids = cache_position.view(1, 1, 1).expand(3, -1, -1).long()

            caches = cache_keys + cache_values
            decoder_mask, new_decoder_attention_mask = get_decoder_mask(decoder_mask)
            # Decode / HBM expect (bsz, 1, max_lm_tokens); get_decoder_mask returns 4D (1, 1, 1, max_lm).
            new_decoder_attention_mask = new_decoder_attention_mask.reshape(1, 1, -1).to(inputs_embeds.dtype)
            next_token_logits, new_keys, new_values = decode_model.forward(
                input_embeddings=inputs_embeds,
                position_ids=decode_position_ids,
                attention_mask=new_decoder_attention_mask,
                caches=caches,
            )
            cache_keys, cache_values = process_kv_cache(cache_keys, cache_values, new_keys, new_values)
            cache_position = cache_position + 1

        return_ids = return_ids[:, num_valid_tokens:]
        return return_ids
