from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import torch
from hbdk4.compiler.overlay import Value


@dataclass
class TensorInfo:
    """TensorInfo class for storing tensor information."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    data: np.ndarray[Any, np.dtype[Any]] | None

    def __init__(
        self,
        obj: Union[
            torch.Tensor,
            Value,
            np.ndarray[Any, np.dtype[Any]],
            np.generic,
            float,
            int,
        ],
        name: str = "",
    ):
        """Initialize the TensorInfo."""
        self.name = name
        if isinstance(obj, torch.Tensor):
            self.data = obj.detach().cpu().numpy()
            self.shape = self.data.shape
            self.dtype = self.data.dtype

        elif isinstance(obj, np.ndarray):
            self.data = obj
            self.shape = self.data.shape
            self.dtype = self.data.dtype

        elif isinstance(obj, Value):
            self.shape = tuple(obj.type.shape)
            self.dtype = np.dtype(obj.type.np_dtype)
            self.data = None

        else:
            if isinstance(obj, (np.generic, float, int)):
                arr = np.asarray(obj)
                self.data = arr
                self.shape = arr.shape
                self.dtype = arr.dtype
            else:
                raise TypeError(f"Unsupported type for TensorInfo: {type(obj)}")

    def dump(self, output_dir: str, prefix: str = "") -> None:
        """Dump tensor metadata and, when available, data to disk."""
        os.makedirs(output_dir, exist_ok=True)

        sanitized_name = self.name.replace("/", "_").replace(".", "_")
        base_filename = f"{prefix}_{sanitized_name}" if prefix else sanitized_name
        base_filepath = os.path.join(output_dir, base_filename)

        meta_path = f"{base_filepath}.json"
        meta_data = {
            "name": self.name,
            "shape": self.shape,
            "dtype": str(self.dtype),
        }
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=4)

        if self.data is not None:
            data_path = f"{base_filepath}.npy"
            np.save(data_path, self.data)


TensorDict = OrderedDict[str, TensorInfo]


@dataclass
class VerifierArgs:
    """Arguments for the model verifier."""

    model_name: str
    model_dir: str
    compare_mode: str | None = None
    transpose_cache: bool = True

    # Input data paths
    input_text_path: str | None = None
    input_image_path: str | None = None  # keep original naming for image input
    input_json_path: str | None = None

    # Model configuration
    chunk_size: int = 256
    cache_len: int = 4096
    device: str = "cpu"

    # BC-specific paths
    quant_llm_model_path: str | None = None
    quant_vlm_model_path: str | None = None

    # HBM-specific paths and connection info
    hbm_llm_model_path: str | None = None
    hbm_vlm_model_path: str | None = None
    remote_ip: str | None = None
    username: str = "root"
    password: str = ""
    port: int = 22
    remote_path: str | None = None
    kept_tokens_file: str | None = None

    # Vision model configuration
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class CompareResult:
    """CompareResult class for storing comparison results."""

    name: str
    torch_shape: tuple[int, ...] | None
    torch_dtype: np.dtype[Any] | None
    torch_info: TensorInfo | None
    bc_shape: tuple[int, ...] | None
    bc_dtype: np.dtype[Any] | None = None
    bc_info: TensorInfo | None = None
    cosine: float | None = None

    def __init__(self, name: str, torch_info: TensorInfo):
        """Initialize the CompareResult."""
        self.name = name
        self.torch_shape = torch_info.shape
        self.torch_dtype = torch_info.dtype
        self.torch_info = torch_info
        self.bc_shape = None
        self.bc_dtype = None
        self.bc_info = None
        self.cosine = None

    def set_bc_info(self, bc_info: TensorInfo | None):
        """Set the bc info."""
        if bc_info is None:
            return

        self.bc_info = bc_info
        self.bc_shape = bc_info.shape
        self.bc_dtype = bc_info.dtype

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dictionary for table/Excel."""
        cosine_str = f"{self.cosine:.6f}" if isinstance(self.cosine, float) else "N/A"
        return {
            "layer": self.name,
            "shape_torch": str(self.torch_shape),
            "shape_bc": str(self.bc_shape) if self.bc_shape is not None else "N/A",
            "dtype_torch": str(self.torch_dtype),
            "dtype_bc": str(self.bc_dtype) if self.bc_dtype is not None else "N/A",
            "cosine": cosine_str,
        }

    def to_detailed_dict(self) -> dict[str, Any]:
        """Convert to detailed dictionary used by Excel detailed sheet."""
        return {
            "layer": self.name,
            "shape_torch": str(self.torch_shape),
            "shape_bc": str(self.bc_shape) if self.bc_shape is not None else "N/A",
            "dtype_torch": str(self.torch_dtype),
            "dtype_bc": str(self.bc_dtype) if self.bc_dtype is not None else "N/A",
            "cosine": self.cosine,
        }


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
