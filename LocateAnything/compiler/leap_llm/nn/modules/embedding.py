import torch
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant


class FakeQuantEmbedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.weight_fake_quant = ConstFakeQuant(8)
        self.absmax_weight = None

    def build(self, x):
        weight_data = self.weight.data.to(torch.float32)
        weight_data = self.weight_fake_quant(weight_data)
        # print("Embedding build absmax:", self.weight_fake_quant.absmax)
        return leap.gather_nd(weight_data, x, 0)

    def forward(self, x: torch.Tensor):
        if self.absmax_weight is None:
            weight_data = self.weight_fake_quant(self.weight.data)
        inputs_embeds = weight_data[x]
        return inputs_embeds


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def build(self, x):
        return leap.gather_nd(self.weight.data, x, 0)

    def forward(self, x: torch.Tensor):
        inputs_embeds = self.weight.data[x]
        return inputs_embeds
