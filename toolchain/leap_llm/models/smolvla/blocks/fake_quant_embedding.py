"""Fake-quant position embedding for SmolVLM vision (no pi0 dependency)."""

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules.const_fake_quant import ConstFakeQuant
from leap_llm.nn.utils import Module


class SmolVLMFakeQuantEmbedding(Module):
    """Embedding lookup with ConstFakeQuant(8) on weight for BPU compile."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.weight_fake_quant = ConstFakeQuant(8)

    def build(self, x):
        weight_data = self.weight.data.to(torch.float16)
        weight_data = self.weight_fake_quant(weight_data)
        return leap.gather_nd(weight_data, x, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_data = self.weight_fake_quant(self.weight.data)
        return weight_data[x]
