"""Eagle3 speculative decoding generation flow.

Eagle3ModelForGeneration wraps a base model and Eagle3 draft model, providing
a generate() interface compatible with llm_compression's calib/compile/eval tools.
"""

import torch
from torch import nn

from .utils import evaluate_posterior, get_llm_config


class Eagle3ModelForGeneration(nn.Module):
    """Composite model that combines base model + Eagle3 draft model.

    Exposes all base model sub-modules (prefill, decode) as attributes
    so calib/compile can access them via getattr(model, model_part).
    """

    def __init__(self, base_model, eagle3_model, prefill_collector, decode_collector, eagle3_config):
        super().__init__()
        self._base_model = base_model
        self.eagle3 = eagle3_model
        self._prefill_collector = prefill_collector
        self._decode_collector = decode_collector
        self._eagle3_config = eagle3_config
        self.config = base_model.config

    @property
    def prefill(self):
        return self._base_model.prefill

    @property
    def decode(self):
        return self._base_model.decode

    def get_input_embeddings(self):
        return self._base_model.get_input_embeddings()

    def get_config(self):
        return self._base_model.get_config()

    def generate(self, inputs, do_sample=False, chunk_prefill=False):
        llm_config = get_llm_config(self.config)
        top_k = self._eagle3_config.get("top_k", 8)
        depth = self._eagle3_config.get("depth", 7)
        total_tokens = self._eagle3_config.get("total_tokens", 60) - 1

        with torch.inference_mode():
            # === Phase 1: Base model prefill ===
            next_token_logits, cache_keys, cache_values, input_ids, num_valid_tokens, input_embeddings = (
                self._base_model.run_prefill(inputs, chunk_prefill)
            )
            hidden_states = self._prefill_collector.get_concat_hidden_states()

            decode_device = next(self.decode.parameters()).device
            eagle3_device = next(self.eagle3.parameters()).device
            embed_device = next(input_embeddings.parameters()).device
            head_dim = getattr(llm_config, "head_dim", llm_config.hidden_size // llm_config.num_attention_heads)
            max_kvcache_len = llm_config.max_kvcache_len
            eos_token_id = llm_config.eos_token_id
            mask_value = torch.finfo(torch.float32).min

            # === Phase 2: Eagle3 draft model prefill (warm up eagle3 KV cache) ===
            eagle3_kv = [
                torch.zeros(
                    1,
                    max_kvcache_len,
                    self.eagle3.config["num_key_value_heads"],
                    head_dim,
                    dtype=next_token_logits.dtype,
                    device=eagle3_device,
                )
                for _ in range(2)
            ]
            eagle3_embed = input_embeddings(input_ids[:, 1:].to(embed_device)).to(eagle3_device)
            _eagle3_prefill(
                self.eagle3,
                eagle3_kv,
                eagle3_embed,
                hidden_states.to(eagle3_device),
                num_valid_tokens - 1,
                max_kvcache_len,
                mask_value,
            )

            # First token from base model argmax
            next_token = torch.argmax(next_token_logits, dim=-1)
            return_ids = torch.cat([input_ids.clone().to(decode_device), next_token[:, None].to(decode_device)], dim=-1)

            # === Phase 3: Speculative decode loop ===
            max_new_tokens = max_kvcache_len - num_valid_tokens
            max_new_tokens_cfg = getattr(llm_config, "max_new_tokens", None)
            if max_new_tokens_cfg is not None:
                max_new_tokens = min(max_new_tokens, max_new_tokens_cfg)

            # Generate first draft tree
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = _topk_generate(
                self.eagle3,
                eagle3_kv,
                hidden_states.to(eagle3_device),
                torch.cat([input_ids, next_token[:, None].to(input_ids.device)], dim=1),
                num_valid_tokens,
                max_kvcache_len,
                mask_value,
                top_k,
                depth,
                total_tokens,
                input_embeddings,
                embed_device,
                eagle3_device,
                is_prefill=True,
            )

            valid_input_len = num_valid_tokens
            base_kv_len = num_valid_tokens

            for _ in range(max_new_tokens):
                # Step A: Base model verifies all candidate tokens in one forward pass
                new_tokens, accept_length, best_candidate, select_indices, verify_hidden, sample_p = _tree_verify(
                    self.decode,
                    self._decode_collector,
                    input_embeddings,
                    draft_tokens,
                    retrieve_indices,
                    tree_mask,
                    tree_position_ids,
                    cache_keys,
                    cache_values,
                    valid_input_len,
                    base_kv_len,
                    max_kvcache_len,
                    mask_value,
                    embed_device,
                    decode_device,
                )
                return_ids = torch.cat([return_ids, new_tokens[None]], dim=-1)

                if eos_token_id in new_tokens.tolist():
                    break
                if return_ids.shape[1] - num_valid_tokens > max_new_tokens:
                    break

                # Step B: Update tracking state
                base_kv_len = min(base_kv_len + select_indices.shape[0], max_kvcache_len)
                valid_input_len += select_indices.shape[0]

                # Step C: Generate next draft tree from accepted hidden states
                # Approximate next_token's hidden by repeating last accepted hidden
                accept_hidden = verify_hidden[:, retrieve_indices[best_candidate, : accept_length + 1]].to(
                    eagle3_device
                )
                accept_hidden = torch.cat([accept_hidden, accept_hidden[:, -1:]], dim=1)

                next_token = torch.argmax(sample_p)[None, None]
                accept_tokens_for_eagle3 = draft_tokens[0, retrieve_indices[best_candidate, 1 : accept_length + 1]]
                eagle3_input_ids = torch.cat([accept_tokens_for_eagle3[None].to(next_token.device), next_token], dim=1)

                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = _topk_generate(
                    self.eagle3,
                    eagle3_kv,
                    accept_hidden,
                    eagle3_input_ids,
                    eagle3_input_ids.shape[1],
                    max_kvcache_len,
                    mask_value,
                    top_k,
                    depth,
                    total_tokens,
                    input_embeddings,
                    embed_device,
                    eagle3_device,
                    is_prefill=False,
                )

            return return_ids[:, num_valid_tokens:]


def _tree_verify(
    decode_model,
    decode_collector,
    input_embeddings,
    draft_tokens,
    retrieve_indices,
    tree_mask,
    tree_position_ids,
    cache_keys,
    cache_values,
    valid_input_len,
    base_kv_len,
    max_kvcache_len,
    mask_value,
    embed_device,
    decode_device,
):
    """Verify draft candidates with base model in one parallel forward pass.

    Returns:
        new_tokens: Accepted tokens + next base model prediction.
        accept_length: Number of accepted draft tokens.
        best_candidate: Index of best candidate path.
        select_indices: Indices of accepted positions in tree (for KV cache update).
        verify_hidden: Hidden states from verification (for next draft round).
        sample_p: Logits at accept boundary (for next token selection).
    """
    draft_tokens_dev = draft_tokens.to(decode_device)
    position_ids = (tree_position_ids.to(decode_device) + valid_input_len).unsqueeze(0)
    seq_len = draft_tokens_dev.shape[-1]

    # Tree attention mask: each candidate can attend to history + its ancestors in tree
    tree_attn_mask = _gen_tree_attn_mask(
        seq_len, base_kv_len, tree_mask, max_kvcache_len, mask_value, decode_device, dims=4
    )
    candidate_embeds = input_embeddings(draft_tokens_dev.to(embed_device)).to(decode_device)

    # Base model forward: return logits for ALL positions (not just last)
    base_logits, new_keys, new_values = decode_model.forward(
        input_embeddings=candidate_embeds,
        position_ids=position_ids.int(),
        attention_mask=tree_attn_mask.squeeze(0).to(candidate_embeds.dtype),
        caches=cache_keys + cache_values,
        return_all_logits=True,
    )

    verify_hidden = decode_collector.get_concat_hidden_states()

    # Extract logits along each candidate path via retrieve_indices
    logits = base_logits[0, retrieve_indices.to(decode_device)]

    # Greedy acceptance: find longest prefix where draft == base argmax
    padding = torch.tensor([[-1]], device=draft_tokens.device)
    candidates = torch.cat([draft_tokens, padding], dim=1)[0, retrieve_indices]
    best_candidate, accept_length, sample_p = evaluate_posterior(logits, candidates)

    # New tokens = accepted draft tokens + base model's prediction at accept boundary
    accept_tokens = candidates[best_candidate, 1 : accept_length + 1]
    next_token_from_base = torch.argmax(sample_p)
    new_tokens = torch.cat([accept_tokens.to(decode_device), next_token_from_base[None].to(decode_device)])

    select_indices = retrieve_indices[best_candidate, : accept_length + 1]

    # Update base KV cache with verified keys/values from accepted path
    for idx in range(len(cache_keys)):
        sel_k = new_keys[idx][:, select_indices]
        sel_v = new_values[idx][:, select_indices]
        n_accept = sel_k.shape[1]
        cache_keys[idx] = torch.cat([cache_keys[idx][:, n_accept:], sel_k], dim=1)
        cache_values[idx] = torch.cat([cache_values[idx][:, n_accept:], sel_v], dim=1)

    return new_tokens, accept_length, best_candidate, select_indices, verify_hidden, sample_p


def _eagle3_prefill(eagle3_model, eagle3_kv, token_embeds, hidden_states, seq_len, cache_len, mask_value):
    """Run Eagle3 draft model on the full prefill sequence to warm up its KV cache."""
    device = token_embeds.device
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    # Causal mask: attend to all previous positions, mask future
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device), 1) * mask_value
    pad = torch.full((seq_len, cache_len - seq_len), mask_value, device=device)
    attn_mask = torch.cat([pad, causal], dim=-1).unsqueeze(0).unsqueeze(0)

    logits, new_k, new_v, out_hidden = eagle3_model(
        token_embeds[:, :seq_len],
        hidden_states[:, :seq_len],
        position_ids,
        attn_mask,
        eagle3_kv,
    )
    # Shift KV cache: drop oldest, append new
    eagle3_kv[0] = torch.cat([eagle3_kv[0][:, seq_len:], new_k], dim=1)
    eagle3_kv[1] = torch.cat([eagle3_kv[1][:, seq_len:], new_v], dim=1)


def _topk_generate(
    eagle3_model,
    eagle3_kv,
    hidden_states,
    input_ids,
    valid_len,
    cache_len,
    mask_value,
    top_k,
    depth,
    total_tokens,
    input_embeddings,
    embed_device,
    eagle3_device,
    is_prefill=False,
):
    """Generate tree-structured candidate tokens using Eagle3 draft model.

    Expands a tree of depth `depth` with branching factor `top_k`, then selects
    the best `total_tokens` candidates by cumulative log-probability.

    Returns:
        draft_tokens: [1, total_tokens+1] candidate token ids (first is sample_token).
        retrieve_indices: [leaf_num, max_depth] indices to extract candidate paths.
        tree_mask: [total_tokens+1, total_tokens+1] attention mask for tree verify.
        tree_position_ids: [total_tokens+1] relative position of each node in tree.
    """
    device = eagle3_device
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    d2t = eagle3_model.d2t.to(device)
    sample_token = input_ids[:, -1].to(device)

    # --- Initial eagle3 forward to get first-level predictions ---
    if is_prefill:
        vl = min(valid_len, top_k)
        eagle3_kv_len = valid_len - 1
        len_posi = valid_len
    else:
        vl = valid_len
        eagle3_kv_len = eagle3_kv[0].shape[1] - (eagle3_kv[0] == 0).all(dim=-1).all(dim=-1).sum().item()
        len_posi = eagle3_kv_len + vl

    # Pad input to top_k width (eagle3 always processes top_k positions)
    pad_ids = torch.full((1, top_k), 2, dtype=torch.long, device=device)
    pad_ids[:, -vl:] = input_ids[:, -vl:].to(device)
    token_embeds = input_embeddings(pad_ids.to(embed_device)).to(device)

    pos_ids = torch.zeros(top_k, dtype=torch.long, device=device)
    base_pos = (valid_len - vl) if is_prefill else eagle3_kv_len
    for i in range(vl):
        pos_ids[-vl + i] = base_pos + i

    # Causal mask for valid positions within the padded window
    attn_mask = torch.ones(top_k, top_k, device=device) * mask_value
    valid_mask = torch.tril(torch.ones(vl, vl, device=device))
    valid_mask[valid_mask == 0] = mask_value
    valid_mask[valid_mask == 1] = 0
    attn_mask[-vl:, -vl:] = valid_mask
    attn_mask = _gen_tree_attn_mask(top_k, eagle3_kv_len, attn_mask, cache_len, mask_value, device, dims=2)

    h_shape = list(hidden_states.shape)
    h_shape[1] = top_k
    pad_hidden = torch.zeros(h_shape, device=device, dtype=hidden_states.dtype)
    pad_hidden[:, -vl:] = hidden_states[:, -vl:]

    logits, new_k, new_v, out_hidden = eagle3_model(
        token_embeds,
        pad_hidden,
        pos_ids[None],
        attn_mask[None, None],
        eagle3_kv,
    )
    eagle3_kv[0] = torch.cat([eagle3_kv[0][:, vl:], new_k[:, -vl:]], dim=1)
    eagle3_kv[1] = torch.cat([eagle3_kv[1][:, vl:], new_v[:, -vl:]], dim=1)
    eagle3_kv_len += vl

    # --- Tree expansion: iteratively expand top_k branches for `depth` levels ---
    kv_with_expand = [cache.clone() for cache in eagle3_kv]
    tree_mask_init = mask_value * torch.ones(top_k, top_k, device=device) - mask_value * torch.eye(top_k, device=device)
    position_ids_init = torch.zeros(top_k, dtype=torch.long, device=device)

    last_hidden = out_hidden[:, -1]
    last_p = logsoftmax(logits[:, -1])
    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values

    scores_list = [topk_p[0][None]]
    parents_list = [torch.zeros(1, dtype=torch.long, device=device)]
    ss_token = [topk_index + d2t[topk_index]]  # Map draft vocab → target vocab

    scores = topk_p[0]
    expand_input_ids = topk_index + d2t[topk_index]
    input_hidden = last_hidden[None].repeat(1, top_k, 1)
    tree_mask = tree_mask_init.clone()
    topk_cs_index = torch.arange(top_k, device=device)

    for i in range(depth):
        position_ids = len_posi + position_ids_init
        token_embeds = input_embeddings(expand_input_ids.to(embed_device)).to(device)
        tree_attn_mask = _gen_tree_attn_mask(top_k, eagle3_kv_len, tree_mask, cache_len, mask_value, device, dims=2)

        logits, new_k, new_v, out_hidden = eagle3_model(
            token_embeds,
            input_hidden,
            position_ids[None],
            tree_attn_mask[None, None],
            kv_with_expand,
        )
        kv_with_expand[0] = torch.cat([kv_with_expand[0][:, top_k:], new_k], dim=1)
        kv_with_expand[1] = torch.cat([kv_with_expand[1][:, top_k:], new_v], dim=1)
        len_posi += 1

        # Track parent indices for tree structure reconstruction
        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + top_k**2 * bias2 + bias1
        parents = topk_cs_index + bias
        parents_list.append(parents)

        # Score each branch by cumulative log-prob, keep top_k globally
        last_p = logsoftmax(logits[0])
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        cu_scores = topk_p + scores[:, None]

        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        scores = topk_cs_p

        out_ids = topk_cs_index // top_k
        input_hidden = out_hidden[:, out_ids]
        expand_input_ids = topk_index.view(-1)[topk_cs_index][None]
        expand_input_ids = expand_input_ids + d2t[expand_input_ids]
        tree_mask = torch.cat((tree_mask[out_ids], tree_mask_init), dim=1)

        ss_token.append(topk_index + d2t[topk_index])
        scores_list.append(cu_scores)

    # --- Assemble final candidate tree from all levels ---
    scores_list = torch.cat(scores_list, dim=0).view(-1)
    ss_token_list = torch.cat(ss_token, dim=0).view(-1)

    # Select top total_tokens candidates by score
    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    top_scores_index = torch.sort(top_scores.indices).values

    draft_tokens = ss_token_list[top_scores_index]
    draft_tokens = torch.cat((sample_token.view(-1), draft_tokens), dim=0)

    # Reconstruct tree structure: build parent pointers and tree mask
    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
    mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
    mask_index[draft_parents == 0] = -1
    mask_index = mask_index + 1
    mask_index_list = mask_index.tolist()

    # Build boolean tree mask: node i can attend to all its ancestors
    final_tree_mask = torch.eye(total_tokens + 1, device=device).bool()
    final_tree_mask[:, 0] = True
    for i in range(total_tokens):
        final_tree_mask[i + 1].add_(final_tree_mask[mask_index_list[i]])

    tree_position_ids = torch.sum(final_tree_mask, dim=1) - 1
    final_tree_mask = final_tree_mask.float()
    final_tree_mask[final_tree_mask == 0.0] = mask_value
    final_tree_mask[final_tree_mask == 1.0] = 0
    draft_tokens = draft_tokens[None]

    # Build retrieve_indices: for each leaf, trace path back to root
    max_depth = torch.max(tree_position_ids) + 1
    noleaf_index = torch.unique(mask_index).tolist()
    leaf_num = total_tokens - (len(noleaf_index) - 1)

    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.int64) - 1
    retrieve_indices = retrieve_indices.tolist()

    rid = 0
    position_ids_list = tree_position_ids.tolist()
    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            dep = position_ids_list[i]
            for j in reversed(range(dep + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    return draft_tokens, retrieve_indices, final_tree_mask, tree_position_ids


def _gen_tree_attn_mask(seq_len, history_len, tree_mask, cache_len, mask_value, device, dims=2):
    """Generate attention mask combining KV cache history + tree structure.

    Args:
        dims: 2 for Eagle3 model (seq_len, cache_len), 4 for base model (1, 1, seq_len, cache_len).
    """
    mask_len = tree_mask.shape[-1]
    if mask_len + history_len > cache_len:
        history_len = cache_len - mask_len
    begin_idx = cache_len - (mask_len + history_len)

    if dims == 4:
        attn_mask = torch.full((1, 1, seq_len, cache_len), mask_value, device=device)
        attn_mask[:, :, :, begin_idx : begin_idx + history_len] = 0
        attn_mask[:, :, :, -mask_len:] = tree_mask.unsqueeze(0).unsqueeze(0)
    else:
        attn_mask = torch.full((seq_len, cache_len), mask_value, device=device)
        attn_mask[:, begin_idx : begin_idx + history_len] = 0
        attn_mask[:, -mask_len:] = tree_mask

    return attn_mask
