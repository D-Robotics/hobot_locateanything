from typing import Callable

import torch

from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4VisionConfig
from leap_llm.nn.utils import Module


class Gemma4VisionRotaryEmbedding(Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        config: Gemma4VisionConfig,
        cache_len: int = 4096,
        device=None,
    ):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.device = device
        self.rope_type = "default"
        rope_init_fn: Callable = self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        cos, sin = self._set_cos_sin_cache()
        # unsqueeze in #head dim = -2
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        self.register_buffer("vision_pe_cos", cos, persistent=False)
        self.register_buffer("vision_pe_sin", sin, persistent=False)

    def compute_position_id(self, max_patches=2520):
        patch_height, patch_width = self.config.image_thw

        patch_grid = torch.meshgrid(
            torch.arange(patch_width, device=self.device),
            torch.arange(patch_height, device=self.device),
            indexing="xy",
        )
        stacked_grid = torch.stack(patch_grid, dim=-1)
        real_positions = stacked_grid.reshape(1, -1, 2)
        position_ids = real_positions
        return position_ids

    @staticmethod
    def compute_default_rope_parameters(
        config: Gemma4VisionConfig,
        device=None,
        seq_len=None,
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
        base = (
            config.rope_parameters["rope_theta"]
            if isinstance(getattr(config, "rope_parameters", None), dict)
            else config.rope_theta
        )
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        spatial_dim = dim // 2
        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / spatial_dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def _set_cos_sin_cache(self):
        inv_freq = self.inv_freq
        position_ids = self.compute_position_id()
        # ``self.device`` may be ``None`` (the default in ``__init__``), so use
        # the buffer's device to be safe.
        device = self.device or inv_freq.device
        position_ids = position_ids.to(device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # Multidimensional positions: [batch, num_patches, ndim]. Apply rotations to each spatial dim separately
        all_cos, all_sin = [], []
        for i in range(2):
            dim_position_ids = position_ids[:, :, i]
            dim_position_ids_expanded = dim_position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ dim_position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            all_cos.append(cos)
            all_sin.append(sin)

        cos = torch.cat(all_cos, dim=-1).to(dtype=inv_freq.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=inv_freq.dtype)
        return cos, sin
