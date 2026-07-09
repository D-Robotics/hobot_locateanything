import torch
import torch.nn as nn
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4VisionConfig
from leap_llm.nn.modules import (
    DynamicQuantLinear,
)
from leap_llm.nn.utils import Module


class Gemma4VisionPatchEmbedder(Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.image_thw = config.image_thw
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = DynamicQuantLinear(
            3 * self.patch_size**2,
            self.hidden_size,
            bias=False,
        )
        self.position_embedding_table = nn.Parameter(torch.ones(2, self.position_embedding_size, self.hidden_size))
        ph, pw = self.image_thw
        self.register_buffer("vision_pe", torch.zeros(1, ph * pw, self.hidden_size), persistent=False)

    def _refresh_vision_pe(self):
        """Recompute the cached ``vision_pe`` from the current table.

        Called from ``_load_from_state_dict`` after a checkpoint is loaded
        so the cache always tracks the live ``position_embedding_table``.
        """
        ph, pw = self.image_thw
        self.vision_pe = self._position_embeddings(ph, pw).to(self.vision_pe.dtype)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Refresh the cached ``vision_pe`` after the table is loaded from
        # the checkpoint; the buffer itself is not part of the state dict.
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        table_key = prefix + "position_embedding_table"
        if table_key in state_dict:
            self._refresh_vision_pe()

    def _position_embeddings(self, patch_height, patch_width):
        patch_grid = torch.meshgrid(
            torch.arange(patch_width),
            torch.arange(patch_height),
            indexing="xy",
        )
        stacked_grid = torch.stack(patch_grid, dim=-1)
        real_positions = stacked_grid.reshape(1, -1, 2)
        one_hot = F.one_hot(real_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        return position_embeddings

    def forward(self, pixel_values):
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values)
        hidden_states = hidden_states + self.vision_pe
        return hidden_states

    def build(self, pixel_values):
        pixel_values = leap.mul(leap.sub(pixel_values, 0.5), 2.0)
        hidden_states = self.input_proj(pixel_values)
        hidden_states = leap.add(hidden_states, self.vision_pe)
        return hidden_states
