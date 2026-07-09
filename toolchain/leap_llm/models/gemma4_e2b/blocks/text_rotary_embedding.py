from typing import Callable, Optional

import torch

from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4TextConfig
from leap_llm.nn.utils import Module


class Gemma4TextRotaryEmbedding(Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma4TextConfig, cache_len: int = 4096, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_init_fns: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}
        self.rope_type: dict[str, str] = {}

        max_len_cached = torch.arange(self.max_seq_len_cached, device=device)
        max_len_cached = max_len_cached.unsqueeze(0)

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            if (rope_type := rope_params["rope_type"]) != "default":
                rope_init_fn = self.compute_proportional_rope_parameters
            else:
                rope_init_fn = self.compute_default_rope_parameters

            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            rope_init_fn_kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, **rope_init_fn_kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)
            # print(f"self.{layer_type}_attention_scaling: {curr_attention_scaling}")
            cos, sin = self._set_cos_sin_cache(max_len_cached, layer_type, device)
            cos_copy = cos[:, : config.cache_len, :].clone()
            sin_copy = sin[:, : config.cache_len, :].clone()
            del cos
            del sin
            # print(f"cos_copy.shape: {cos_copy.shape}")
            # print(f"sin_copy.shape: {sin_copy.shape}")
            self.register_buffer(f"{layer_type}_cos", cos_copy, persistent=False)
            self.register_buffer(f"{layer_type}_sin", sin_copy, persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Gemma4TextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @staticmethod
    def compute_proportional_rope_parameters(
        config: Gemma4TextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
        head_dim_key: str | None = None,
    ):
        rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

        head_dim = getattr(config, head_dim_key, None) or config.hidden_size // config.num_attention_heads

        base = rope_parameters_dict["rope_theta"]
        factor = rope_parameters_dict.get("factor", 1.0)
        rope_proportion = rope_parameters_dict.get("partial_rotary_factor", 1.0)

        attention_factor = 1.0  # Unused in this type of RoPE

        rope_angles = int(rope_proportion * head_dim // 2)

        inv_freq_rotated = 1.0 / (
            base
            ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
        )

        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                (
                    inv_freq_rotated,
                    torch.zeros(nope_angles, dtype=torch.float32, device=device),
                ),
                dim=0,
            )
        else:
            inv_freq = inv_freq_rotated

        inv_freq /= factor
        return inv_freq, attention_factor

    @torch.no_grad()
    def _set_cos_sin_cache(self, position_ids, layer_type=None, device=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        return cos.to(device), sin.to(device)
