"""Sampling utilities for token generation."""

import torch


def repetition_penalty_logits_process(scores, input_ids, penalty, prompt_length=None):
    """Apply repetition penalty to logits.

    Args:
        scores: Logits tensor of shape (batch_size, vocab_size).
        input_ids: Input token ids of shape (batch_size, seq_len).
        penalty: Repetition penalty value (>1.0 penalises repeated tokens).
        prompt_length: If provided, penalty is applied only to generation tokens
            (tokens after prompt_length), aligning with Transformers v4.41.0+
            behaviour and the original repetition-penalty paper.
            If None, penalty is applied to all tokens (legacy behaviour).
    """
    if prompt_length is None:
        unique_ids = torch.unique(input_ids, dim=1)
        score = torch.gather(scores, 1, unique_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        scores_processed = scores.scatter(1, unique_ids, score)
    else:
        generation_ids = input_ids[:, prompt_length:]
        if generation_ids.shape[1] > 0:
            unique_generation_ids = torch.unique(generation_ids, dim=1)
            score = torch.gather(scores, 1, unique_generation_ids)
            score = torch.where(score < 0, score * penalty, score / penalty)
            scores_processed = scores.scatter(1, unique_generation_ids, score)
        else:
            scores_processed = scores
    return scores_processed


def temperature_logits_process(scores, temperature):
    """Divide logits by temperature."""
    return scores / temperature


def topk_logits_process(scores, top_k, filter_value=float("-inf")):
    """Keep only the top-k logits; mask the rest with filter_value."""
    top_k = min(top_k, scores.size(-1))
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    return scores.masked_fill(indices_to_remove, filter_value)


def topp_logits_process(scores, top_p, min_tokens_to_keep=1, filter_value=float("-inf")):
    """Nucleus (top-p) filtering: keep the smallest set of tokens whose
    cumulative probability exceeds top_p, masking the rest with filter_value."""
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return scores.masked_fill(indices_to_remove, filter_value)
