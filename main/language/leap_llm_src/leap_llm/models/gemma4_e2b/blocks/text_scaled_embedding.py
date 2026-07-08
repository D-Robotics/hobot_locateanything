import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules.embedding import FakeQuantEmbedding


class Gemma4TextScaledWordEmbedding(FakeQuantEmbedding):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(vocab_size, hidden_size)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

    def build(self, input_ids):
        """build flow for scaled word embeddings - gather + scale"""
        embedded = super().build(input_ids)
        return leap.mul(self.scalar_embed_scale, embedded)
