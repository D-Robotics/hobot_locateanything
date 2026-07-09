# mypy: allow-untyped-defs
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
#   Changes: Conv1d & Conv3d Added static quantization processing and
#            incorporated build function to construct the leap model.
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

import torch
from hbdk4.compiler import leap
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant

__all__ = ["Conv1d", "Conv3d"]


class _ConvNd(Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    # type: ignore[empty-body]
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor: ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"  # noqa: E501
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"  # noqa: E501
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.bias = None

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        quant_bits = 16
        self.x_fake_quant = ConstFakeQuant(quant_bits)
        self.out_fake_quant = ConstFakeQuant(quant_bits)
        self.absmax_weight = None

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        out = F.conv1d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def build(self, input):
        input = self.x_fake_quant(input)

        weight = self.weight.data.to(torch.float32)
        weight = weight.transpose(1, 2).contiguous()

        bias = self.bias.data.to(torch.float32) if self.bias is not None else None

        weight_min = (-self.absmax_weight).tolist()
        weight_max = self.absmax_weight.tolist()

        weight_quant = leap.const_fake_quant(
            weight,
            weight_min,
            weight_max,
            8,
            True,
            axis=0,
        )

        output = leap.conv(
            input=input,
            weight=weight_quant,
            bias=bias,
            stride=self.stride,
            pad=[self.padding[0], self.padding[0]],
            dilation=self.dilation,
            groupNum=self.groups,
        )
        output = self.out_fake_quant(output)
        return output

    def forward(self, input: Tensor) -> Tensor:
        input = self.x_fake_quant(input)

        if self.absmax_weight is None:
            weight_reshape = torch.reshape(self.weight, [self.out_channels, -1])
            per_channel_max, _ = torch.max(weight_reshape.abs(), dim=1)
            self.absmax_weight = per_channel_max

        output = self._conv_forward(input, self.weight, self.bias)
        output = self.out_fake_quant(output)

        return output


class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        quant_output=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        self.quant_output = quant_output
        quant_bits = 8
        self.x_fake_quant = ConstFakeQuant(quant_bits)
        absmax_weight = torch.zeros(self.out_channels)
        self.register_buffer("absmax_weight", absmax_weight, persistent=False)
        self.visited = False
        if self.quant_output:
            self.out_fake_quant = ConstFakeQuant(16)

    def build(self, input):
        x_quant = self.x_fake_quant(input)
        weight = self.weight.data
        weight = weight.permute(0, 2, 3, 1).contiguous()

        bias = self.bias.data if self.bias is not None else None
        weight_min = (-self.absmax_weight).tolist()
        weight_max = self.absmax_weight.tolist()
        weight = leap.cast_type(weight, output_type=leap.float32)
        weight_quant = leap.const_fake_quant(
            weight,
            weight_min,
            weight_max,
            8,
            True,
            axis=0,
        )

        # assert self.padding == (0, 0), "padding not supported yet"

        output = leap.conv2d(
            input=x_quant,
            weight=weight_quant,
            bias=bias,
            pad=[self.padding[0], self.padding[0], self.padding[0], self.padding[0]],
            stride=self.stride,
            dilation=self.dilation,
            groupNum=self.groups,
        )
        if self.quant_output:
            output = self.out_fake_quant(output)
        return output

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        input = self.x_fake_quant(input)
        if self.visited is False:
            weight_reshape = torch.reshape(self.weight, [self.out_channels, -1])
            per_channel_max, _ = torch.max(weight_reshape.abs(), dim=1)
            self.absmax_weight = per_channel_max
            self.visited = True
        output = self._conv_forward(input, self.weight, self.bias)
        if self.quant_output:
            output = self.out_fake_quant(output)
        return output


class Conv3d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        quant_output=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        self.quant_output = quant_output
        quant_bits = 8
        self.x_fake_quant = ConstFakeQuant(quant_bits)
        if self.quant_output:
            self.out_fake_quant = ConstFakeQuant(16)
        absmax_weight = torch.zeros(self.out_channels)
        self.register_buffer("absmax_weight", absmax_weight, persistent=False)
        self.visited = False

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def build(self, input):
        input = self.x_fake_quant(input)
        weight = self.weight.data.to(torch.float32)
        weight = weight.permute(0, 2, 3, 4, 1).contiguous()

        bias = self.bias.data.to(torch.float32) if self.bias is not None else None

        weight_min = (-self.absmax_weight).tolist()
        weight_max = self.absmax_weight.tolist()

        weight_quant = leap.const_fake_quant(
            weight,
            weight_min,
            weight_max,
            8,
            True,
            axis=0,
        )
        # TODO: @junjun.zhao
        assert self.padding == (0, 0, 0), "padding not supported yet"

        output = leap.conv3d(
            input=input,
            weight=weight_quant,
            bias=bias,
            stride=self.stride,
            dilation=self.dilation,
            groupNum=self.groups,
        )
        if self.quant_output:
            output = self.out_fake_quant(output)
        return output

    def forward(self, input: Tensor) -> Tensor:
        input = self.x_fake_quant(input)
        if self.visited is False:
            weight_reshape = torch.reshape(self.weight, [self.out_channels, -1])
            per_channel_max, _ = torch.max(weight_reshape.abs(), dim=1)
            self.absmax_weight = per_channel_max
            self.visited = True
        output = self._conv_forward(input, self.weight, self.bias)
        if self.quant_output:
            output = self.out_fake_quant(output)
        return output
