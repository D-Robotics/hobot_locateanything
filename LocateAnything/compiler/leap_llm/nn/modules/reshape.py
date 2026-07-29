from typing import Sequence, Union

import torch
from hbdk4.compiler import leap
from torch import SymInt
from torch.types import _int

from leap_llm.nn.utils import Module


class Reshape(Module):
    def __init__(self) -> None:
        super().__init__()

    def build(self, x, shape):
        return leap.reshape(x, shape)

    def forward(self, x: torch.Tensor, shape: Sequence[Union[_int, SymInt]]):
        return x.reshape(shape)
