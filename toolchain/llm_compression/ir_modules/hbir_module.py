# Copyright (c) Horizon Robotics. All rights reserved.
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from hbdk4.compiler import Hbm, load
from hbdk4.compiler.overlay import Value
from torch import device

from llm_compression.utils.logger import get_logger

try:
    from torch._six import string_classes
except ImportError:
    string_classes = (str, bytes)

from .ir_module import IrModule  # noqa: E402

logger = get_logger(__name__)

__all__ = ["HbirModule"]

np2torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}
convert_tensor = torch.utils.data._utils.collate.default_convert


def convert_numpy(
    data: Any,
    to_list: bool = False,
    dtype: Optional[str] = None,
) -> Any:
    r"""Convert each Tensor array data field into a numpy, recursively."""
    elem_type = type(data)
    if elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if dtype:
            data = data.astype(dtype)
        data = data.tolist() if to_list else data
        return data
    elif isinstance(data, torch.Tensor):
        scale = None
        data = data.detach().cpu().numpy()
        if dtype:
            data = data.astype(dtype)
        if to_list:
            data = data.tolist()
        if scale is not None:
            return (data, scale)
        else:
            return data
    elif isinstance(data, Mapping):
        return {key: convert_numpy(data[key], to_list=to_list, dtype=dtype) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(convert_numpy(d, to_list=to_list, dtype=dtype) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, string_classes):
        return [convert_numpy(d, to_list=to_list, dtype=dtype) for d in data]
    else:
        return data


class HbirModule(IrModule):
    """Inference module of hbir.

    Args:
         model_path: Path of ir model file.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
    """

    def __init__(
        self,
        model_path: str,
        return_tensor: bool = True,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
    ):
        super().__init__(
            model_path=model_path,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )
        if model_path.endswith(".hbm"):
            self.model = Hbm(model_path)
        else:
            self.model = load(model_path)
        self.return_tensor = return_tensor
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        # self.model.functions[0].register_callback(self._bc_callback)
        self.bc_layers_outputs = {}

    def get_layers_outputs(self):
        return self.bc_layers_outputs

    def _bc_callback(self, op, results, operands):
        if op.type == "func.func":
            return True
        if len(results) > 0 and isinstance(results[0], (torch.Tensor, np.ndarray, Value)):
            # print(op.name, results[0].max())
            op_name = op.name.split(",")[-1].split('"')[1]
            if op_name in self.bc_layers_outputs:
                for idx in range(100):
                    name = f"{op_name}_{idx}"
                    if name not in self.bc_layers_outputs:
                        break
            else:
                name = op_name

            self.bc_layers_outputs[name] = results[0]
        return True

    def check_input_impl(self, data):
        assert isinstance(data, Dict)
        format_data = {}
        for inp in self.model[0].inputs:
            if inp.quant_info is None:
                dtype = np2torch_dtype_dict[inp.type.np_dtype]
                self.check_type_shape(inp.name, inp.type.shape, dtype, data)
                format_data[inp.name] = data[inp.name]
                self.device = data[inp.name].device
            else:
                self.device = data[inp.name].device
                _scale = torch.tensor(inp.quant_info.scales[0], device=self.device)
                _zero_point = torch.tensor(inp.quant_info.zeros[0], device=self.device)
                data[inp.name] = torch.clamp(
                    torch.round(data[inp.name] / _scale) + _zero_point, -128, 127
                )  # kvcache convert_bc is int8
                data[inp.name] = data[inp.name].to(torch.int8)
                format_data[inp.name] = data[inp.name]

        return convert_numpy(format_data)

    def check_output_impl(self, data):
        return_data = data
        for outp in self.model[0].outputs:
            if outp.quant_info is not None:
                _scale = np.array(outp.quant_info.scales[0])
                _zero_point = np.array(outp.quant_info.zeros[0])
                return_data[outp.name] = (return_data[outp.name] - _zero_point) * _scale
        if self.return_tensor:
            return_data = convert_tensor(return_data)
            for k, v in return_data.items():
                return_data[k] = v.to(device=self.device)
        return return_data

    def forward_impl(self, data):
        output = self.model.functions[0].feed(data)
        return output

    def cpu(self):
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        logger.warning("Hbir does not support GPU now. Use CPU instead.")
        return self


hbir_module = None


def hbir_module_init(model_path, return_tensor, input_func, output_func):
    global hbir_module
    hbir_module = HbirModule(
        model_path=model_path,
        return_tensor=return_tensor,
        reformat_input_func=input_func,
        reformat_output_func=output_func,
    )


def hbir_module_task(data):
    return hbir_module(data)
