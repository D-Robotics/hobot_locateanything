import torch

from leap_llm.models.gemma4_e2b.blocks.rmsnorm import Gemma4RMSNorm
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4VisionConfig,
)
from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class Gemma4MultimodalEmbedder(Module):
    def __init__(
        self,
        multimodal_config: Gemma4AudioConfig | Gemma4VisionConfig,
        text_hidden_size: int = 1536,
    ):
        super().__init__()

        self.multimodal_hidden_size = getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size)
        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_hidden_size
        self.embedding_projection = DynamicQuantLinear(self.multimodal_hidden_size, self.text_hidden_size, bias=False)
        self.embedding_pre_projection_norm = Gemma4RMSNorm(self.multimodal_hidden_size, eps=self.eps, with_scale=False)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.

        Args:
            inputs_embeds (torch.Tensor): Multimodal hidden states from
                the tower's pooler/output. Shape:
                ``(batch_size, num_tokens, multimodal_hidden_size)``.
                For vision (post-pool): ``(batch_size, 256, 768)``.
                For audio (post-output_proj): ``(batch_size, num_tokens, 2048)``.

        Returns:
            torch.Tensor: Projected embeddings in the language model
                space. Shape:
                ``(batch_size, num_tokens, text_hidden_size)``,
                e.g. ``(batch_size, 256, 1536)`` for vision or
                ``(batch_size, num_tokens, 1536)`` for audio.
        """
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)

    def build(self, inputs_embeds):
        """Leap export path. See :meth:`forward` for shapes."""
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        embs = self.embedding_projection(embs_normed)
        return embs
