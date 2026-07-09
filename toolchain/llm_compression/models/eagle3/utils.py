import torch


def get_llm_config(config):
    """Extract LLM config from a model config object.

    Probes for nested config structures:
        config.llm_config -> config.text_config -> config itself
    """
    for attr in ("llm_config", "text_config"):
        if hasattr(config, attr):
            return getattr(config, attr)
    return config


def evaluate_posterior(logits, candidates):
    """Evaluate candidate sequences against base model predictions (greedy).

    For each candidate path, computes the longest prefix where the candidate
    tokens match the base model's argmax predictions.

    Args:
        logits: Base model logits [leaf_num, seq_len, vocab_size].
        candidates: Candidate token sequences [leaf_num, max_depth].

    Returns:
        best_candidate: Index of the best candidate.
        accept_length: Number of accepted tokens.
        sample_p: Logits at the acceptance boundary for next token generation.
    """
    posterior_mask = (candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)).int()
    candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
    accept_length = candidates_accept_length.max()

    if accept_length == 0:
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

    return best_candidate, accept_length, logits[best_candidate, accept_length]


class HiddenStateCollector:
    """Collect hidden states from base model layers via forward hooks.

    Registers hooks on specified layers to capture their output hidden states,
    then concatenates them along the last dimension for Eagle3's fc layer input.
    """

    def __init__(self):
        self.collected = {}
        self._handles = []

    def make_hook(self, layer_idx):
        def hook(module, input, output):
            # output is (hidden_states, new_key, new_value)
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.collected[layer_idx] = hidden_states.detach()

        return hook

    def register(self, layers, layer_indices):
        """Register hooks on specified decoder layers.

        Args:
            layers: nn.ModuleList of decoder layers.
            layer_indices: List of layer indices to hook.
        """
        self.remove()
        for idx in layer_indices:
            handle = layers[idx].register_forward_hook(self.make_hook(idx))
            self._handles.append(handle)

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.collected.clear()

    def get_concat_hidden_states(self):
        """Get concatenated hidden states from hooked layers.

        Returns:
            Tensor of shape [batch, seq_len, hidden_size * num_layers].
        """
        sorted_states = [self.collected[k] for k in sorted(self.collected.keys())]
        result = torch.cat(sorted_states, dim=-1)
        self.collected.clear()
        return result
