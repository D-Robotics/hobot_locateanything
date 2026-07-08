# Copyright (c) Horizon Robotics. All rights reserved.

import gc
import os

import torch
from safetensors import safe_open

import horizon_plugin_pytorch as horizon
from horizon_plugin_pytorch.dtype import qint4, qint8
from horizon_plugin_pytorch.march import set_march
from horizon_plugin_pytorch.nn import Softmax
from horizon_plugin_pytorch.nn.layer_norm import SplitLayerNorm
from horizon_plugin_pytorch.quantization import get_qconfig
from horizon_plugin_pytorch.quantization.observer_v2 import HistogramObserver
from horizon_plugin_pytorch.quantization.qconfig_setter import ModuleNameTemplate, QconfigSetter
from llm_compression.utils.logger import get_logger
from llm_compression.utils.trace_utils import trace_all_branches

Softmax.reduce_rcp_float_precision = torch.float32
SplitLayerNorm.head_float_precision = torch.float32


logger = get_logger(__name__)

__all__ = [
    "Float2Calibration",
]


def sync_kvcache_scales(q_model, prefill_model, decode_model):
    """Unify KV-cache int8 scales between prefill and decode after calibration.

    In runtime, the KV-cache tensors produced by the prefill model are
    directly reused by the decode model, and they share the same memory.
    Therefore, their quantization scales must be consistent to avoid
    numerical mismatch between prefill and decode."""
    logger.info("unifying KV-cache int8 scales between prefill and decode...")
    fq_suffix = ".activation_post_process"
    fq_names = set(n + fq_suffix for n in q_model.get_kvcache_names("prefill"))
    prefill_fqs = {n: m for n, m in prefill_model.named_modules() if n in fq_names and hasattr(m, "scale")}
    decode_fqs = {n: m for n, m in decode_model.named_modules() if n in fq_names and hasattr(m, "scale")}
    for name in prefill_fqs:
        mod_p, mod_d = prefill_fqs[name], decode_fqs[name]
        scale_p, scale_d = mod_p.scale.detach().cpu(), mod_d.scale.detach().cpu()
        unified = scale_p if scale_p.abs().max() >= scale_d.abs().max() else scale_d
        mod_p.scale.data.copy_(unified.to(mod_p.scale.device))
        mod_d.scale.data.copy_(unified.to(mod_d.scale.device))
    logger.info(f"KV-cache scale sync done for {len(prefill_fqs)} modules.")


def load_weight_qparams(
    model_dir: str,
):
    """
    parse .safetensors file and extract weight qparams.
    Args:
        model_dir (str): directory containing .safetensors files
    Returns:
        dict: a dictionary of the form
        {module_name: {"threshold": {"weight": tensor}, "dtype": {"weight": dtype}}}
    """

    shard_files = sorted(
        os.path.join(model_dir, filename) for filename in os.listdir(model_dir) if filename.endswith(".safetensors")
    )

    suffix_to_param = {
        "buf_scales": "scale",
        "buf_zeros": "zero_point",
        "buf_qmin": "qmin",
        "buf_qmax": "qmax",
    }
    strip_prefixes = ("model.",)

    qparams_raw = {}
    for file in shard_files:
        with safe_open(file, framework="pt") as tensors:
            for key in tensors.keys():  # noqa
                mod_name, _, suffix = key.rpartition(".")
                param_name = suffix_to_param.get(suffix)
                if param_name is None:
                    continue

                if mod_name.startswith(strip_prefixes):
                    mod_name = mod_name.split(".", 1)[1]
                qparams_raw.setdefault(mod_name, {})[param_name] = tensors.get_tensor(key)

    qparams_dict = {}
    for mod_name, params in qparams_raw.items():
        qmin = int(params["qmin"].item())
        qmax = int(params["qmax"].item())

        dtype = next((dt for dt in (qint4, qint8) if qmin == dt.min and qmax == dt.max), None)
        if dtype is None:
            raise ValueError(f"Unsupported qmin/qmax ({qmin}, {qmax}) for module {mod_name}")

        zero_point = params["zero_point"]
        if not torch.all(zero_point == 0):
            raise ValueError(f"Non-zero zero_point is not supported for module {mod_name}")

        qparams_dict[mod_name] = {
            "dtype": {"weight": dtype},
            "threshold": {"weight": params["scale"] * (-qmin)},
        }

    return qparams_dict


class Float2Calibration:
    """Convert float model to calibration model for quantization.

    This class wraps the quantization preparation process using JIT_STRIP method,
    applying quantization templates and dynamic quantization to the model.

    Args:
        q_model: The quantized model wrapper (BaseQModel instance).
            Quantization settings will be automatically retrieved from q_model.
        model_part: Name of the model part (visual, prefill, decode, etc.).
        custom_config: Configuration object containing calibration settings.
        observer: Observer class for quantization. Defaults to HistogramObserver.
    """

    def __init__(
        self,
        q_model,
        model_part,
        custom_config,
        observer=HistogramObserver,
    ):
        if custom_config.model.march in ("nash-e", "nash-m", "nash-p", "nash-starry-p"):
            set_march(custom_config.model.march)
        else:
            raise ValueError(f"Unsupported march type: {custom_config.model.march}")

        horizon.qat_mode.set_qat_mode("fuse_bn")

        # Automatically retrieve settings from q_model and custom_config
        # TODO: dynamic quant api in plugin will be refactored, currently a demo.
        # In shared LM mode (model_list contains "lm"), compile stage splits lm
        # into prefill+decode but qconfig must stay "lm" to match the calibration checkpoint.
        model_list = getattr(custom_config.model, "model_list", [])
        qconfig_part = "lm" if model_part in ("prefill", "decode") and "lm" in model_list else model_part
        q_templates = q_model.get_qconfig_setting(qconfig_part)
        weight_qparams = load_weight_qparams(custom_config.model.model_path)
        if weight_qparams:
            logger.info(
                f"Loaded weight qparams for {len(weight_qparams)} modules from {custom_config.model.model_path}"
            )
            q_templates = [ModuleNameTemplate(weight_qparams, freeze=True)] + q_templates
        example_inputs = q_model.get_model_trace_dummy_input(model_part)
        log_path = getattr(getattr(custom_config, "calibration", None), "log_path", None) or "."
        log_save_dir = os.path.join(log_path, f"{model_part}_qconfig_setting")

        self.qconfig_setter = QconfigSetter(
            reference_qconfig=get_qconfig(observer=observer),
            templates=q_templates,
            save_dir=log_save_dir,
            enable_optimize=False,
        )

        self.example_inputs = example_inputs

    def __call__(self, model, calib_ckpt_path=None):
        """Convert model to calibration model.

        Args:
            model: The original model to convert
            calib_ckpt_path: Optional path to calibration checkpoint file.
                If provided and file exists, loaded after calibration.

        Returns:
            Prepared calibration model
        """
        model.eval()
        # Enable trace_all_branches so that modules with prefill/decode
        # if-else branches execute both paths during JIT trace, ensuring
        # all operators (e.g. torch.cat in decode path) are recorded.
        with trace_all_branches():
            model = horizon.quantization.prepare(
                model,
                example_inputs=self.example_inputs,
                qconfig_setter=self.qconfig_setter,
                method=horizon.quantization.PrepareMethod.JIT_STRIP,
                inplace=True,
            )

        # Load calibration checkpoint if provided
        if calib_ckpt_path is not None:
            if not os.path.exists(calib_ckpt_path):
                raise FileNotFoundError(f"Calibration checkpoint not found: {calib_ckpt_path}")
            logger.info(f"Loading calibration checkpoint from {calib_ckpt_path}")
            state_dict = torch.load(calib_ckpt_path, map_location="cpu")
            miss_key, unexpected_key = model.load_state_dict(state_dict, False)
            logger.info(f"miss_key: {miss_key}")
            logger.info(f"unexpected_key: {unexpected_key}")
            horizon.quantization.set_fake_quantize(model, horizon.quantization.FakeQuantState.VALIDATION)
        else:
            horizon.quantization.set_fake_quantize(model, horizon.quantization.FakeQuantState.CALIBRATION)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return model
