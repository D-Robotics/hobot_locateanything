# Copyright (c) Horizon Robotics. All rights reserved.

import os
from collections import deque

import torch
from torch import nn

from llm_compression.converters.compile_converter import get_hbm_name
from llm_compression.ir_modules.hbir_module import HbirModule
from llm_compression.ir_modules.hbm_module import PipelineHbmModule

__all__ = [
    "HbmWrapper",
    "load_bc_modules",
    "load_hbm_modules",
    "torch2hbm",
]


class HbmWrapper(nn.Module):
    """Generic wrapper for model with HBM support."""

    def __init__(self, hbm, model_part, q_model, original_model=None):
        """Initialize HBM wrapper.

        Args:
            hbm: HBM module for inference
            model_part: Name of the model part (e.g., 'visual', 'prefill', 'decode')
            q_model: Quantized model instance providing HBM helper functions
            original_model: Original model instance to proxy attributes from
        """
        super().__init__()
        self.hbm = hbm
        self.model_part = model_part
        self.q_model = q_model
        self.input_names, self.output_names = q_model.get_model_input_output_name(model_part)
        object.__setattr__(self, "_original_model", original_model)

    def _build_hbm_inputs(self, *args, **kwargs):
        """Build HBM input dict from `args` / `kwargs`.

        Rules:
        - Flatten `args` recursively: any list/tuple inside `args` is expanded depth-first.
        - Fill `self.input_names` in order:
          - If there are still flattened positional values, consume from `args` first.
          - Else if `name` exists in `kwargs`, take `kwargs[name]` (and pop it).
        - After the loop, if `kwargs` still has remaining items, flatten their values
          (list/tuple expanded recursively) and append to the remaining slots.

        Returns:
        - A dict mapping `input_names[i] -> values[i]`.
        """

        def flatten(x):
            if isinstance(x, (list, tuple)):
                result = []
                for item in x:
                    result.extend(flatten(item))
                return result
            else:
                return [x]

        values = []
        arg_queue = deque(v for a in args for v in flatten(a))
        caches = kwargs.pop("caches", None)
        kwargs.pop("return_all_logits", None)
        for name in self.input_names:
            if arg_queue:
                values.append(arg_queue.popleft())
            elif name in kwargs:
                values.append(kwargs.pop(name))
        if caches is not None:
            values.extend(flatten(caches))
        if kwargs:
            values.extend(flatten(list(kwargs.values())))

        return dict(zip(self.input_names, values, strict=False))

    def _parse_hbm_outputs(self, hbm_outputs):
        """Parse HBM output dict into a Python object.

        Rules:
        - If there is only one output name, return that tensor directly.
        - Otherwise iterate `self.output_names` in order:
          - Names containing substring `"key"` are collected into `new_keys` (preserving order).
          - Names containing substring `"value"` are collected into `new_values` (preserving order).
          - All other outputs are put into `parsed` by their original name.
        - If any keys/values were collected, add:
          - `parsed["new_keys"] = new_keys`
          - `parsed["new_values"] = new_values`
        - Return `list(parsed.values())` (in insertion order).
        """
        parsed = {}

        keys = []
        values = []
        if len(self.output_names) == 1:
            return hbm_outputs[self.output_names[0]]

        for name in self.output_names:
            out = hbm_outputs[name]
            if "key" in name:
                keys.append(out)
            elif "value" in name:
                values.append(out)
            else:
                parsed[name] = out

        if keys or values:
            parsed["new_keys"] = keys
            parsed["new_values"] = values

        return list(parsed.values())

    def _slice_prefill_logits_to_last_token(self, outputs):
        """Keep generate compatible when prefill HBM exports full-sequence logits."""
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() == 3 and outputs.shape[1] > 1:
                return outputs[:, -1:, :]
            return outputs
        if (
            isinstance(outputs, list)
            and outputs
            and isinstance(outputs[0], torch.Tensor)
            and outputs[0].dim() == 3
            and outputs[0].shape[1] > 1
        ):
            outputs = list(outputs)
            outputs[0] = outputs[0][:, -1:, :]
        return outputs

    def forward(self, *args, **kwargs):
        """Forward pass through HBM or original model."""
        return_all_logits = kwargs.get("return_all_logits", False)
        hbm_inputs = self._build_hbm_inputs(*args, **kwargs)
        hbm_outputs = self.hbm(hbm_inputs)
        outputs = self._parse_hbm_outputs(hbm_outputs)
        if self.model_part == "prefill" and not return_all_logits:
            outputs = self._slice_prefill_logits_to_last_token(outputs)
        return outputs

    def parameters(self, recurse=True):
        """Proxy to original model's parameters for dtype/device queries."""
        original = object.__getattribute__(self, "_original_model")
        if original is not None:
            return original.parameters(recurse=recurse)
        return super().parameters(recurse=recurse)

    def __getattr__(self, name):
        """Proxy missing attributes to the wrapped original module.

        This wrapper keeps HBM-specific members on itself, and forwards any
        unknown attribute access to `_original_model` (if provided), so that
        callers can transparently access the original module's methods and
        state (e.g. helper functions, cached tensors, etc.).
        """
        if name in ("hbm", "model_part", "q_model", "_original_model"):
            if name in self.__dict__:
                return self.__dict__[name]
            if name != "_original_model":
                _modules = object.__getattribute__(self, "_modules")
                if name in _modules:
                    return _modules[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        original_model = self.__dict__.get("_original_model")
        if original_model is not None and hasattr(original_model, name):
            return getattr(original_model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def load_hbm_modules(hbm_config, hbm_load_path, model_list, hbm_names):
    """load HBM modules based on configuration and model_list.

    Args:
        hbm_config: Configuration dict for HBM settings
        hbm_load_path: Path to HBM files
        model_list: List of model parts from config
        hbm_names: Mapping from model part to resolved HBM filename
    """
    hbm_modules = {}
    hbm_path_list = []
    host = getattr(hbm_config, "host", None)
    if host is None:
        raise ValueError("Host is not set for hbm_rpc_eval. ")
    username = getattr(hbm_config, "username", "root")
    password = getattr(hbm_config, "password", None)
    remote_root = getattr(hbm_config, "remote_root", "/map/hbm_infer")
    core_id = getattr(hbm_config, "core_id", None)
    remote_environment = getattr(hbm_config, "remote_environment", None)
    frame_timeout = getattr(hbm_config, "frame_timeout", None)
    # Normalize remote_environment from EasyDict/dict to plain str-keyed dict
    if remote_environment is not None:
        remote_environment = {str(k): str(v) for k, v in remote_environment.items()}
    seen_hbm_paths = set()
    for model_part in model_list:
        if model_part not in hbm_names:
            raise ValueError(f"HBM name not found for model_part='{model_part}'")
        hbm_file = hbm_names[model_part]
        hbm_path = os.path.join(hbm_load_path, hbm_file)
        if hbm_path in seen_hbm_paths:
            continue  # prefill and decode share lm.hbm; avoid loading it twice
        seen_hbm_paths.add(hbm_path)
        if os.path.exists(hbm_path):
            hbm_path_list.append(hbm_path)
        else:
            raise FileNotFoundError(f"HBM file not found for model_part='{model_part}' at '{hbm_path}'. ")

    for model_part in model_list:
        hbm_modules[model_part] = PipelineHbmModule(
            host=host,
            hbm_path=hbm_path_list,
            hbm_name=model_part,
            username=username,
            password=password,
            remote_root=remote_root,
            core_id=core_id,
            remote_environment=remote_environment,
            frame_timeout=frame_timeout,
        )

    return hbm_modules


def load_bc_modules(bc_load_path, model_list, stage="export"):
    """Load BC modules using HbirModule for local inference.

    Args:
        bc_load_path: Path to BC files directory.
        model_list: List of model parts from config.
        stage: Which compilation stage to load.
            'export' -> {model_part}.bc
            'convert' -> {model_part}_convert.bc
    """
    stage_suffix = {"export": "", "convert": "_convert"}
    suffix = stage_suffix.get(stage)
    if suffix is None:
        raise ValueError(f"Unknown stage='{stage}', expected one of {list(stage_suffix.keys())}")

    bc_modules = {}
    bc_path_to_module = {}
    for model_part in model_list:
        bc_file = f"{model_part}{suffix}.bc"
        bc_path = os.path.join(bc_load_path, bc_file)
        if not os.path.exists(bc_path):
            raise FileNotFoundError(f"BC file not found for model_part='{model_part}' at '{bc_path}'.")
        bc_path_to_module[bc_path] = HbirModule(model_path=bc_path)
        bc_modules[model_part] = bc_path_to_module[bc_path]

    return bc_modules


def setup_model_with_hbm(q_model, hbm_modules, model_list):
    """Setup model with HBM wrappers (internal)."""
    model = q_model.get_generated_model()
    model.eval()
    for model_part in model_list:
        original_model = getattr(model, model_part)
        hbm_module = hbm_modules.get(model_part)
        wrapped = HbmWrapper(
            hbm=hbm_module,
            model_part=model_part,
            q_model=q_model,
            original_model=original_model,
        )
        setattr(model, model_part, wrapped)
    return model


def torch2hbm(q_model, custom_config):
    """Convert torch model to HBM RPC model (attach HBM to q_model's generated model).

    Builds hbm_modules from config and attaches HBM wrappers to the model from
    q_model.get_generated_model(). Caller is responsible for creating q_model.

    Args:
        q_model: Quantized model instance (already loaded).
        custom_config: Config with model.model_list and hbm_rpc_eval.*.

    Returns:
        model: Model with HBM wrappers attached, for model.generate().
    """
    hbm_config = getattr(custom_config, "hbm_rpc_eval", None)
    if not hbm_config:
        raise ValueError("hbm_rpc_eval configuration not found in config file")
    hbm_load_path = hbm_config.hbm_load_path
    model_list = getattr(custom_config.model, "model_list", [])
    if not model_list:
        raise ValueError("model_list is empty in configuration")
    if "lm" in model_list:
        model_list = [p for p in model_list if p != "lm"] + ["prefill", "decode"]

    stage = getattr(hbm_config, "stage", "compile")
    if stage in ("export", "convert"):
        hbm_modules = load_bc_modules(hbm_load_path, model_list, stage)
    elif stage == "compile":
        hbm_names = {model_part: get_hbm_name(q_model, model_part, custom_config) for model_part in model_list}
        hbm_modules = load_hbm_modules(hbm_config, hbm_load_path, model_list, hbm_names)
    else:
        raise ValueError(f"Unknown stage='{stage}', expected one of 'export', 'convert', 'compile'")

    model = setup_model_with_hbm(q_model, hbm_modules, model_list)
    return model
