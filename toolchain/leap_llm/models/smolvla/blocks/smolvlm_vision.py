"""SmolVLM2 vision embeddings matching transformers SmolVLMVisionEmbeddings."""

from __future__ import annotations

import torch
from hbdk4.compiler import leap

from leap_llm.models.smolvla.blocks.configuration_smolvlm import SmolVLMVisionConfig
from leap_llm.models.smolvla.blocks.fake_quant_embedding import SmolVLMFakeQuantEmbedding
from leap_llm.nn.modules.vision_embedding import FakeQuantPatchEmbedding
from leap_llm.nn.utils import Module


def compute_smolvlm_position_ids(
    patch_attention_mask: torch.BoolTensor,
    num_patches_per_side: int,
    coord_dtype: torch.dtype | None = None,
) -> torch.LongTensor:
    """Match transformers.models.smolvlm.modeling_smolvlm.SmolVLMVisionEmbeddings."""
    batch_size = patch_attention_mask.shape[0]
    max_nb_patches_h = patch_attention_mask.shape[1]
    max_nb_patches_w = patch_attention_mask.shape[2]
    device = patch_attention_mask.device
    if coord_dtype is None:
        coord_dtype = torch.float32

    boundaries = torch.arange(
        1 / num_patches_per_side,
        1.0,
        1 / num_patches_per_side,
        device=device,
    )
    position_ids = torch.full(
        size=(batch_size, max_nb_patches_h * max_nb_patches_w),
        fill_value=0,
        dtype=torch.long,
        device=device,
    )

    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        nb_patches_h = p_attn_mask[:, 0].sum()
        nb_patches_w = p_attn_mask[0].sum()

        # HF uses pixel_values.dtype for coordinate arange (float16 at inference).
        h_indices = torch.arange(nb_patches_h, device=device, dtype=coord_dtype)
        w_indices = torch.arange(nb_patches_w, device=device, dtype=coord_dtype)

        fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
        fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (
            bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w
        ).flatten()
        position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

    return position_ids


def full_image_patch_mask(
    batch_size: int,
    image_size: int,
    patch_size: int,
    device: torch.device,
) -> torch.BoolTensor:
    side = image_size // patch_size
    return torch.ones(batch_size, side, side, dtype=torch.bool, device=device)


class SmolVLMVisionEmbeddings(Module):
    """HF-compatible vision token embedding (variable-resolution position ids)."""

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2

        self.patch_embedding = FakeQuantPatchEmbedding(
            config.hidden_size, config.num_channels, config.patch_size
        )
        self.position_embedding = SmolVLMFakeQuantEmbedding(
            self.num_patches, self.embed_dim
        )

        # Precompute position ids for fixed full square images (calibration / HBM compile).
        # Match HF inference: coordinates computed in float16 when pixel_values are float16.
        mask = full_image_patch_mask(1, self.image_size, self.patch_size, torch.device("cpu"))
        pos = compute_smolvlm_position_ids(
            mask, self.num_patches_per_side, coord_dtype=torch.float16
        )
        self.register_buffer("full_image_position_ids", pos, persistent=False)

    def _embed(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        position_ids = compute_smolvlm_position_ids(
            patch_attention_mask,
            self.num_patches_per_side,
            coord_dtype=pixel_values.dtype,
        )
        return embeddings + self.position_embedding(position_ids)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        if patch_attention_mask is None:
            batch_size = pixel_values.shape[0]
            patch_attention_mask = full_image_patch_mask(
                batch_size,
                pixel_values.shape[2],
                self.patch_size,
                pixel_values.device,
            )
        return self._embed(pixel_values, patch_attention_mask)

    def build(self, pixel_values, position_ids=None):
        del position_ids  # use HF-compatible ids for fixed square input
        batch = pixel_values.type.shape[0]
        side = self.num_patches_per_side
        pixel_nhwc = leap.transpose(pixel_values, [0, 2, 3, 1])
        self.patch_embedding.to("cpu", dtype=torch.float32)
        patch_embeds = self.patch_embedding(pixel_nhwc)
        embeddings = leap.reshape(patch_embeds, [batch, side * side, self.embed_dim])
        pos = self.full_image_position_ids.to(torch.int64)
        pos = leap.reshape(pos, [side * side, 1])
        pos_emb = self.position_embedding(pos)
        return leap.add(embeddings, pos_emb)
