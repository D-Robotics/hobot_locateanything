# -----------------------------------------------------------------------------
# This file includes modifications to the original code from PyTorch
# Source: https://github.com/pytorch/pytorch
#
# Copyright (c) 2016-     Facebook, Inc. and other contributors
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# -----------------------------------------------------------------------------
# Modifications:
# - [2025-09-09] Modified by junjun.zhao
#   Changes: AvgPool1d Added static quantization processing and
#            incorporated build function to construct the leap model.
# -----------------------------------------------------------------------------

from typing import Optional

from hbdk4.compiler import leap
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import (
    _size_1_t,
    _size_any_t,
)
from torch.nn.modules.utils import _single

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant

__all__ = ["AvgPool1d"]
# __all__ = ["MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d"]


class _MaxPoolNd(Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ]
    return_indices: bool
    ceil_mode: bool

    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", dilation={dilation}, ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class _AvgPoolNd(Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
    ]

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"  # noqa: E501


class AvgPool1d(_AvgPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.x_fake_quant = ConstFakeQuant(16)
        self.out_fake_quant = ConstFakeQuant(16)

    def build(self, input):
        input = self.x_fake_quant(input)
        output = leap.avg_pool(
            input,
            kernel=[self.kernel_size[0]],
            stride=[self.stride[0]],
            pad=[self.padding[0], self.padding[0]],
            dilation=[1],
            ceilMode=self.ceil_mode,
        )
        output = self.out_fake_quant(output)
        return output

    def forward(self, input: Tensor) -> Tensor:
        input = self.x_fake_quant(input)
        output = F.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
        output = self.out_fake_quant(output)
        return output
