# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import device

from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["IrModule"]


class IrModule(nn.Module):
    """Basic class of ir module.

    Args:
         model_path: Path of ir model file.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
    """

    def __init__(
        self,
        model_path: str,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.model_path = model_path
        # assert os.path.exists(self.model_path), "model_path not exists."
        logger.info(f"Load ir module from {self.model_path}!")

        self.reformat_input_func = reformat_input_func
        self.reformat_output_func = reformat_output_func

    @torch.no_grad()
    def forward(self, data):
        if self.reformat_input_func is not None:
            data = self.reformat_input_func(data)

        data = self.check_input_impl(data)
        result = self.forward_impl(data)
        result = self.check_output_impl(result)

        if self.reformat_output_func is not None:
            result = self.reformat_output_func(result)
        return result

    def forward_impl(self, data):
        pass

    def check_input_impl(self, data):
        return data

    def check_output_impl(self, data):
        return data

    def check_type_shape(self, name_info, shape_info, dtype_info, data):
        logger.debug(f"check: {name_info}, {data[name_info].shape}, {shape_info}")
        if name_info in data:
            if not torch.Size(shape_info) == data[name_info].shape:
                raise ValueError(
                    f"Shape of {name_info} is not matched, \
                    expect {shape_info} but get \
                    {data[name_info].shape}."
                )

            if str(dtype_info) not in str(data[name_info].dtype):
                if str(dtype_info) == "torch.int32":
                    data[name_info] = data[name_info].to(dtype=torch.int32)
                elif str(dtype_info) == "torch.int64":
                    data[name_info] = data[name_info].to(dtype=torch.int64)
                elif str(dtype_info) == "torch.float16":
                    data[name_info] = data[name_info].to(dtype=torch.float16)
                else:
                    raise TypeError(
                        f"Dtype of {name_info} is not matched, \
                        excepted {dtype_info} but get \
                        {data[name_info].dtype}."
                    )
        else:
            raise KeyError(f"Cannot find {name_info} in {list(data.keys())}.")

    def cpu(self):
        pass

    def cuda(self, device: Optional[Union[int, device]] = None):
        pass
