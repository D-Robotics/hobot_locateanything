# Copyright (c) Horizon Robotics. All rights reserved.
import os
from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Union

import numpy as np
import torch
from hbm_infer.hbm_rpc_session_flexible import HbmHandle, HbmRpcServer, HTensor
from hbm_infer.hbm_rpc_session_flexible import HbmRpcSession as HbmRpcSessionFlexible
from torch.utils._pytree import tree_flatten

from llm_compression.ir_modules.ir_module import IrModule
from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)


def _env_enabled(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


class PipelineHbmServer:
    def __init__(
        self,
        host,
        hbm_path,
        username: str = "root",
        password: Optional[str] = None,
        remote_root: str = "/map/hbm_infer",
        core_id: Optional[Sequence[int]] = None,
        remote_environment: Optional[Mapping[str, str]] = None,
        frame_timeout: Optional[int] = None,
    ):
        self.hbm_server = HbmRpcServer(host=host, username=username, password=password, remote_root=remote_root)
        self.hbm_handle = HbmHandle(local_hbm_path=hbm_path, hbm_rpc_server=self.hbm_server)
        session_kwargs = {}
        if core_id is not None:
            session_kwargs["core_id"] = list(core_id)
        if remote_environment is not None:
            session_kwargs["remote_environment"] = {str(k): str(v) for k, v in remote_environment.items()}
        if frame_timeout is not None:
            session_kwargs["frame_timeout"] = frame_timeout
        self.session = HbmRpcSessionFlexible(
            hbm_handle=self.hbm_handle,
            hbm_rpc_server=self.hbm_server,
            **session_kwargs,
        )

    def __del__(self):
        self.close_server()

    def close_server(self) -> None:
        logger.info("close server")
        self.hbm_handle.deinit()
        self.hbm_server.deinit()

    def get_session(self):
        return self.session


class PipelineHbmModule(IrModule):
    hbm_server = None

    def __init__(
        self,
        host=None,
        hbm_path=None,
        hbm_name="default",
        hbm_server=None,
        username: str = "root",
        password: Optional[str] = None,
        remote_root: str = "/map/hbm_infer",
        core_id: Optional[Sequence[int]] = None,
        remote_environment: Optional[Mapping[str, str]] = None,
        frame_timeout: Optional[int] = None,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
        quant_input: bool = True,
        dequant_output: bool = True,
        output_config=None,
    ):
        super().__init__(
            model_path=None,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )
        self.host = host
        self.username = username
        self.password = password
        self.remote_root = remote_root
        self.core_id = core_id
        self.remote_environment = remote_environment
        self.enable_tensor_dump = _env_enabled("HBM_DUMP_INFERENCE_TENSORS")
        self.dump_root = os.path.join(os.getcwd(), "hbm_tensor_dumps", hbm_name)
        self.dump_index = 0

        if hbm_server is None:
            if PipelineHbmModule.hbm_server is None:
                logger.info(f"init hbm server, {hbm_name}")
                choosen_one = self.get_host()
                logger.info(f"choosen: {choosen_one}")
                PipelineHbmModule.hbm_server = PipelineHbmServer(
                    choosen_one,
                    hbm_path,
                    username=username,
                    password=password,
                    remote_root=remote_root,
                    core_id=core_id,
                    remote_environment=remote_environment,
                    frame_timeout=frame_timeout,
                )
            self.initialized = True
        else:
            PipelineHbmModule.hbm_server = hbm_server
            self.initialized = True

        self.session = self.get_session()
        self.hbm_name = hbm_name
        self.output_config = output_config

        self.quant_input = quant_input
        self.dequant_output = dequant_output
        for model_name in self.session.get_model_names():
            logger.info(model_name)
        self.input_info = self.session.get_input_info(self.hbm_name)
        self.output_info = self.session.get_output_info(self.hbm_name)
        for k, v in self.input_info.items():
            logger.debug(f"input_{k}, {v}")
        for k, v in self.output_info.items():
            logger.debug(f"output_{k}, {v}")

        if not all(["quantizeAxis" in v for k, v in self.input_info.items()]):
            logger.warning(
                f"Not all inputs has QuantiScaleInfo"
                f"(quantizeAxis, scale, zero_point), "
                f"while dequant_input is True. "
                f"{['quantizeAxis' in v for k, v in self.input_info.items()]}"
            )

        if not all(["quantizeAxis" in v for k, v in self.output_info.items()]):
            logger.warning(
                f"Not all outputs has QuantiScaleInfo"
                f"(quantizeAxis, scale, zero_point), "
                f"while dequant_output is True. "
                f"{['quantizeAxis' in v for k, v in self.output_info.items()]}"
            )

    def get_host(self):
        if isinstance(self.host, list):
            ranks = torch.distributed.get_rank()
            host_idx = ranks % len(self.host)
            host = self.host[host_idx]
            logger.info(f"Choose host {host}.")
            return host
        else:
            return self.host

    def show_input_output_info(self):
        return self.session.show_input_output_info(self.hbm_name)

    def sanitize_dump_name(self, name):
        return str(name).replace("/", "_").replace(" ", "_")

    def tensor_to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous().numpy()
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value)
        return None

    def dump_tensor_tree(self, prefix, name, value, dump_dir, file_index):
        if isinstance(value, dict):
            for child_name, child_value in value.items():
                self.dump_tensor_tree(
                    prefix,
                    f"{name}_{self.sanitize_dump_name(child_name)}",
                    child_value,
                    dump_dir,
                    file_index,
                )
            return

        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                self.dump_tensor_tree(prefix, f"{name}_{idx}", item, dump_dir, file_index)
            return

        array = self.tensor_to_numpy(value)
        if array is None:
            return

        current_index = file_index[0]
        file_index[0] += 1
        shape_str = "x".join(str(dim) for dim in array.shape) or "scalar"
        file_name = f"{prefix}_{current_index:06d}_{self.sanitize_dump_name(name)}_{str(array.dtype)}_{shape_str}.bin"
        array.tofile(os.path.join(dump_dir, file_name))

    def dump_inference_tensors(self, stage, tensors):
        dump_dir = os.path.join(self.dump_root, f"{self.dump_index:06d}_{stage}")
        os.makedirs(dump_dir, exist_ok=True)
        self.dump_tensor_tree(stage, self.hbm_name, tensors, dump_dir, [0])

    def forward_impl(self, data):
        if self.enable_tensor_dump:
            self.dump_inference_tensors("input", data)
        output = self.session(
            data=data,
            output_config=self.output_config,
            model_name=self.hbm_name,
        )
        if self.enable_tensor_dump:
            self.dump_inference_tensors("output", output)
        self.dump_index += 1
        return output

    def get_session(self):
        return PipelineHbmModule.hbm_server.get_session() if self.initialized is True else None

    def is_initialized(self):
        return self.initialized

    def cpu(self):
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        logger.warning("Hbm does not support GPU now. Use CPU instead.")
        return self

    def get_top_device(self, data):
        top_device = torch.device("cpu")
        flat_inputs, _ = tree_flatten(data)
        for d in flat_inputs:
            if isinstance(d, torch.Tensor) and "cuda" in d.device.type:
                top_device = d.device
        return top_device

    def up_to_top_device(self, data, top_device):
        # 0. peel off list & tuple with length == 1
        for k in data:
            if isinstance(data[k], (list, tuple)) and len(data[k]) == 1:
                data[k] = data[k][0]

        for k in data:
            # 1. to torch.Tensor
            if isinstance(data[k], np.ndarray):
                data[k] = torch.from_numpy(data[k])

            # 2. to top_device
            if isinstance(data[k], torch.Tensor):
                if "cuda" in top_device.type:
                    data[k] = data[k].to(device=top_device)
            elif isinstance(data[k], HTensor):
                continue
            else:
                raise TypeError(
                    "Input data not matched the hbm inputs, check the "
                    "input & output info with show_input_output_info()"
                )

        return data

    def check_input_impl(self, data):
        self.device = self.get_top_device(data)
        data = self.filter_input_data(data)
        data = self.up_to_top_device(data, self.device)
        if self.quant_input:
            data = self.quant_input_data(data)

        return data

    def check_output_impl(self, data):
        return_data = data

        if self.dequant_output:
            return_data = self.dequant_output_data(return_data)
        return_data = self.up_to_top_device(return_data, self.device)
        return return_data

    def filter_input_data(self, data):
        # 1. All inputs is ready.
        for name in self.input_info:
            if name not in data:
                logger.warning(f"Missing input: {name}")
        if all(name in data for name in self.input_info):
            return {name: data[name] for name in self.input_info}
        raise TypeError(
            "Input data not matched the hbm inputs, check " "the input & output info with show_input_output_info()"
        )

    def quant_input_data(self, data):
        for k in self.input_info:
            if self.input_info[k]["tensor_type"] == "DATA_TYPE_S32":
                data[k] = data[k].to(dtype=torch.int32)
            if self.input_info[k]["tensor_type"] == "DATA_TYPE_S64":
                data[k] = data[k].to(dtype=torch.int64)
            if self.input_info[k]["tensor_type"] == "DATA_TYPE_F16" and data[k].dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                data[k] = data[k].to(dtype=torch.float16)
            if "quantizeAxis" not in self.input_info[k]:
                continue
            if isinstance(data[k], HTensor):
                continue
            axis = self.input_info[k]["quantizeAxis"]
            new_shape = [1] * len(data[k].shape)
            new_shape[axis] = len(self.input_info[k]["scale_data"])

            _min, _max = {
                "DATA_TYPE_S8": [-128, 127],
                "DATA_TYPE_S16": [-32768, 32767],
            }[self.input_info[k]["tensor_type"]]
            if isinstance(data[k], torch.Tensor):
                _scale = torch.tensor(self.input_info[k]["scale_data"], device=data[k].device).reshape(new_shape)
                _zero_point = torch.tensor(self.input_info[k]["zero_point_data"], device=data[k].device).reshape(
                    new_shape
                )
                data[k] = torch.clamp(torch.round(data[k] / _scale) + _zero_point, _min, _max)
                data[k] = data[k].to(self.session.hbm_map_torch[self.input_info[k]["tensor_type"]][0])
            elif isinstance(data[k], np.ndarray):
                _scale = np.array(self.input_info[k]["scale_data"]).reshape(new_shape)
                _zero_point = np.array(self.input_info[k]["zero_point_data"]).reshape(new_shape)
                data[k] = np.clip(np.round(data[k] / _scale) + _zero_point, _min, _max)
                data[k] = data[k].astype(self.session.hbm_map_numpy[self.input_info[k]["tensor_type"]][0])
            else:
                raise TypeError(f"data cannot support {type(data[k])}")
        return data

    def dequant_output_data(self, data):
        for k in self.output_info:
            if k not in data:
                continue
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device=self.device)
            if "quantizeAxis" not in self.output_info[k]:
                continue
            if isinstance(data[k], HTensor):
                continue
            axis = self.output_info[k]["quantizeAxis"]
            new_shape = [1] * len(data[k].shape)
            new_shape[axis] = len(self.output_info[k]["scale_data"])

            if isinstance(data[k], torch.Tensor):
                _scale = torch.tensor(self.output_info[k]["scale_data"], device=data[k].device).reshape(new_shape)
                _zero_point = torch.tensor(self.output_info[k]["zero_point_data"], device=data[k].device).reshape(
                    new_shape
                )
                data[k] = (data[k] - _zero_point) * _scale
            elif isinstance(data[k], np.ndarray):
                _scale = np.array(self.output_info[k]["scale_data"]).reshape(new_shape)
                _zero_point = np.array(self.output_info[k]["zero_point_data"]).reshape(new_shape)
                data[k] = (data[k] - _zero_point) * _scale
            else:
                raise TypeError(f"data cannot support {type(data[k])}")
        return data
