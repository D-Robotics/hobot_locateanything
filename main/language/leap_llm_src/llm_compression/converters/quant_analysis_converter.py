import copy
import inspect
import os

import torch
from hbdk4.compiler import load
from horizon_plugin_profiler import QuantAnalysis
from horizon_plugin_pytorch.quantization.hbdk4 import pre_export
from tqdm import tqdm

from llm_compression.converters import Float2Calibration
from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)

# float->plugin(fake_quant), plugin->pre_export, pre_export->export_bc, export_bc->convert_bc
STAGES = ("fake_quant", "pre_export", "export", "convert")


def get_module_by_name(model, module_name):
    if hasattr(model, module_name):
        return getattr(model, module_name)
    raise ValueError(f"Model does not have attribute '{module_name}'")


def to_device(data, device):
    """Recursively move tensors to device; supports nested list/tuple/dict."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return [to_device(x, device) for x in data]
    if isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data


class CaptureInputContext:
    def __init__(self, module):
        self.module = module
        self.captured_inputs = []
        self.original_forward = None

    def __enter__(self):
        self.original_forward = self.module.forward
        sig = inspect.signature(self.original_forward)

        def new_forward(*args, **kwargs):
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                self.captured_inputs.append(bound.args)
            except Exception as e:
                logger.warning(f"Failed to capture inputs: {e}")
            return self.original_forward(*args, **kwargs)

        self.module.forward = new_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_forward:
            self.module.forward = self.original_forward


def capture_inputs_for_part(float_model_wrapper, model_part, dataloader, q_model=None, stage=None, chunk_prefill=False):
    """Run the model and capture inputs for a specific submodule."""
    logger.info(f"Capturing inputs for {model_part}...")
    float_module = get_module_by_name(float_model_wrapper, model_part)

    with CaptureInputContext(float_module) as ctx:
        with torch.no_grad():
            for sample in tqdm(dataloader):
                float_model_wrapper.generate(sample, do_sample=False, chunk_prefill=chunk_prefill)
        captured_inputs = ctx.captured_inputs

    logger.info(f"Captured {len(captured_inputs)} inputs for {model_part}")
    if len(captured_inputs) == 0:
        raise RuntimeError(f"No inputs captured for {model_part}. Check dataloader and model_part.")

    if stage in ("export", "convert") and q_model is not None:
        check_inputs_valid(captured_inputs, q_model, model_part)

    return captured_inputs


def _prepare_plugin_model(q_model, float_module, model_part, config, calib_load_path):
    float_module_for_plugin = copy.deepcopy(float_module)
    ckpt_name = f"{model_part}_calibration.pth.tar"
    ckpt_file = os.path.join(calib_load_path, ckpt_name)
    assert os.path.exists(ckpt_file), f"Calibration checkpoint not found in {calib_load_path}"
    calibrator = Float2Calibration(q_model=q_model, model_part=model_part, custom_config=config)
    return calibrator(float_module_for_plugin, calib_ckpt_path=ckpt_file)


def _prepare_pre_export_model(q_model, float_module, model_part, config, calib_load_path):
    plugin_model = _prepare_plugin_model(q_model, float_module, model_part, config, calib_load_path)
    return pre_export(plugin_model)


def _prepare_export_bc_model(q_model, float_module, model_part, config, bc_path):
    export_bc_name = f"{model_part}.bc"
    export_bc_file = os.path.join(bc_path, export_bc_name)
    assert os.path.exists(export_bc_file), f"Export BC file not found in {bc_path}"
    return load(export_bc_file)


def _prepare_convert_bc_model(q_model, float_module, model_part, config, bc_path):
    convert_bc_name = f"{model_part}_convert.bc"
    convert_bc_file = os.path.join(bc_path, convert_bc_name)
    assert os.path.exists(convert_bc_file), f"Convert BC file not found in {bc_path}"
    return load(convert_bc_file)


def prepare_baseline_model(stage, q_model, float_module, model_part, config, qa_config):
    if stage == "fake_quant":
        return float_module
    if stage == "pre_export":
        return _prepare_plugin_model(q_model, float_module, model_part, config, qa_config.baseline_model_load_path)
    if stage == "export":
        return _prepare_pre_export_model(q_model, float_module, model_part, config, qa_config.baseline_model_load_path)
    if stage == "convert":
        return _prepare_export_bc_model(q_model, float_module, model_part, config, qa_config.baseline_model_load_path)
    raise ValueError(f"Unknown stage: {stage}, must be one of {STAGES}")


def prepare_analysis_model(stage, q_model, float_module, model_part, config, qa_config):
    if stage == "fake_quant":
        return _prepare_plugin_model(q_model, float_module, model_part, config, qa_config.analysis_model_load_path)
    if stage == "pre_export":
        return _prepare_pre_export_model(q_model, float_module, model_part, config, qa_config.analysis_model_load_path)
    if stage == "export":
        return _prepare_export_bc_model(q_model, float_module, model_part, config, qa_config.analysis_model_load_path)
    if stage == "convert":
        return _prepare_convert_bc_model(q_model, float_module, model_part, config, qa_config.analysis_model_load_path)
    raise ValueError(f"Unknown stage: {stage}, must be one of {STAGES}")


def check_inputs_valid(captured_inputs, q_model, model_part):
    """Align tensor dtypes in captured_inputs with example_inputs.

    Convert to example's dtype when inconsistent, due to BC input constraints.
    """
    if not captured_inputs:
        return
    example_inputs = q_model.get_model_trace_dummy_input(model_part)

    def _align_dtype(c, e):
        if isinstance(e, torch.Tensor) and isinstance(c, torch.Tensor) and c.dtype != e.dtype:
            return c.to(e.dtype)
        if isinstance(e, list) and isinstance(c, list):
            return [_align_dtype(ci, ei) for ci, ei in zip(c, e)]
        return c

    for i, sample in enumerate(captured_inputs):
        s = list(sample) if isinstance(sample, (tuple, list)) else [sample]
        aligned = [_align_dtype(sc, ex) for sc, ex in zip(s, example_inputs)]
        captured_inputs[i] = (
            tuple(aligned)
            if isinstance(sample, tuple)
            else (aligned[0] if not isinstance(sample, (tuple, list)) else aligned)
        )


def run_quant_analysis(baseline_model, analysis_model, stage, captured_inputs, output_dir, metrics, device_ids):
    """Run the actual quantization analysis."""
    logger.info(f"Running QuantAnalysis (stage={stage}) in {output_dir}...")
    qa = QuantAnalysis(
        baseline_model,
        analysis_model,
        analysis_model_type=stage,
        device_ids=device_ids,
        out_dir=output_dir,
    )
    qa.auto_find_bad_case(captured_inputs)
    qa.run(run_baseline_model=True, run_analysis_model=True)
    qa.compare_per_layer()
    if stage != "convert":
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            m = metric.upper()
            logger.info(f"Calculating sensitivity ({m})...")
            qa.sensitivity(metric=m, output_names=("0",))
    else:
        logger.info("Convert stage, not supporting sensitivity calculation, skip...")
    logger.info(f"Analysis completed. Results saved to {output_dir}")
