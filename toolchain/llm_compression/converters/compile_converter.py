# Copyright (c) Horizon Robotics. All rights reserved.

import inspect
import json
import os
import shutil

import horizon_plugin_pytorch as horizon
import torch
from hbdk4.compiler import Hbm, compile, convert, link, save, statistics
from hbdk4.compiler.extra_apis import llm_convert
from horizon_plugin_pytorch.quantization.hbdk4 import export
from horizon_plugin_pytorch.quantization.qconfig_setter import ConvDtypeTemplate
from torch import nn

from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "Calibration2Hbm",
    "hbo2hbm",
    "get_embed_tokens_filename",
    "get_hbm_desc",
    "get_hbm_name",
    "resolve_embed_dtype",
    "save_embed_tokens",
    "save_tokenizer_files",
]

_VALID_EMBED_DTYPES = ("fp32", "fp16")


def resolve_embed_dtype(compile_config: object) -> str:
    """Resolve external embed_tokens file dtype from compile config."""
    dtype = getattr(compile_config, "embed_dtype", "fp32")
    if dtype not in _VALID_EMBED_DTYPES:
        raise ValueError(f"compile.embed_dtype must be one of {_VALID_EMBED_DTYPES}, got {dtype!r}")
    return dtype


def get_embed_tokens_filename(model_name: str, embed_dtype: str = "fp32") -> str:
    """Return the external embed_tokens filename for the given dtype."""
    if embed_dtype == "fp16":
        return f"{model_name}_embed_tokens_fp16.bin"
    return f"{model_name}_embed_tokens.bin"


def save_embed_tokens(model_name: str, embedding: torch.Tensor, output_path: str, embed_dtype: str = "fp32") -> str:
    """Save embedding weights for external runtime lookup.

    Args:
        model_name: The base name of the model.
        embedding: The embedding weight tensor.
        output_path: The directory to save the file in.
        embed_dtype: The target dtype ("fp32" or "fp16").

    Returns:
        The absolute path to the saved token embeddings file.
    """
    if embed_dtype not in _VALID_EMBED_DTYPES:
        raise ValueError(f"embed_dtype must be one of {_VALID_EMBED_DTYPES}, got {embed_dtype!r}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    emb = embedding.detach().cpu()
    emb = emb.half() if embed_dtype == "fp16" else emb.float()
    out = os.path.join(output_path, get_embed_tokens_filename(model_name, embed_dtype))
    if not os.path.exists(out):
        emb.numpy().tofile(out)
    return out


def save_tokenizer_files(model_path, output_path):
    """Copy tokenizer files to the HBM output directory.

    If tokenizer.json exists in model_path, copy both tokenizer_config.json
    and tokenizer.json directly. Otherwise, generate them via AutoTokenizer.
    """
    os.makedirs(output_path, exist_ok=True)

    tokenizer_json_src = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_src):
        shutil.copy2(os.path.join(model_path, "tokenizer_config.json"), output_path)
        shutil.copy2(tokenizer_json_src, output_path)
        logger.info(f"Copied tokenizer files from {model_path} to {output_path}")
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Generated tokenizer files via AutoTokenizer to {output_path}")


def get_hbm_desc(q_model):
    """Build HBM description metadata.

    The desc follows the new format:
    - Hugging face config sections: all JSON files from model_path are loaded as-is,
      keyed by filename without .json extension.
    - Horizon section: fields that cannot be directly mapped from hugging face configs,
      including user yml overrides, derived values, and token strings resolved
      from tokenizer_config's added_tokens_decoder.

    Only one desc is generated and written to lm.hbm.
    """

    def set_if_not_none(target, key, value):
        if value is not None:
            target[key] = value

    base_cfg = q_model.generated_model.config
    model_list = q_model.custom_config.model.model_list
    horizon_desc = {}

    # Load all hf configs from model_path
    model_path = q_model.custom_config.model.model_path
    hf_configs = _load_hf_configs(model_path)

    # Override temporal_patch_size to 1 for qwen25vl
    if "Qwen2.5-VL" in model_path:
        _set_nested_value(hf_configs, "temporal_patch_size", 1)

    # Visual horizon fields
    get_cfg = getattr(q_model, "get_generated_model_cfg", None)
    model_part = next((p for p in ["visual", "vision_model"] if p in model_list), None)
    if model_part is not None:
        cfg = get_cfg(model_part) if callable(get_cfg) else None
        image_height = getattr(cfg, "image_height", None) or getattr(cfg, "image_size", None)
        image_width = getattr(cfg, "image_width", None) or getattr(cfg, "image_size", None)
        set_if_not_none(horizon_desc, "image_width", image_width)
        set_if_not_none(horizon_desc, "image_height", image_height)

    # LM (prefill/decode) horizon fields
    model_part = "prefill" if "prefill" in model_list else "decode"
    cfg = get_cfg(model_part) if callable(get_cfg) else None
    set_if_not_none(horizon_desc, "prefill_chunk_size", getattr(cfg, "max_lm_input_len", None))
    set_if_not_none(horizon_desc, "prefill_cache_len", getattr(cfg, "max_kvcache_len", None))
    set_if_not_none(horizon_desc, "decode_cache_len", getattr(cfg, "max_kvcache_len", None))
    set_if_not_none(horizon_desc, "decode_chunk_size", 1)
    set_if_not_none(horizon_desc, "vocab_size", getattr(cfg, "vocab_size", None))
    set_if_not_none(horizon_desc, "template", getattr(base_cfg, "template", None))
    horizon_desc["mask_pad_value"] = -32768  # get_decoder_mask function uses -32768
    horizon_desc["vocab_compression"] = False

    compile_config = getattr(q_model.custom_config, "compile", None)
    if compile_config is not None:
        embed_dtype = resolve_embed_dtype(compile_config)
        horizon_desc["embed_dtype"] = embed_dtype
        base_name = os.path.basename(model_path.rstrip("/"))
        horizon_desc["embed_weight_name"] = get_embed_tokens_filename(base_name, embed_dtype)

    # Token strings: resolved from tokenizer_config's added_tokens_decoder by token ID.
    # These are shared across visual, audio, and LM — needed for chat template rendering.
    # e.g. config.image_token_id=151655 → tokenizer_config.added_tokens_decoder["151655"].content
    tokenizer_config = hf_configs.get("tokenizer_config", {})
    for token_name in ("image_token", "video_token", "audio_token"):
        token_str = _resolve_token_str(token_name, base_cfg, tokenizer_config)
        set_if_not_none(horizon_desc, token_name, token_str)

    # Build final desc with public configs + horizon
    hbm_desc = {}
    hbm_desc.update(hf_configs)
    hbm_desc["horizon"] = horizon_desc

    return hbm_desc


def get_hbm_name(q_model, model_name, custom_config):
    march = custom_config.model.march
    base_name = os.path.basename(custom_config.model.model_path.rstrip("/"))
    # TODO: weight bits should be configurable and accessible from custom_config
    w_bits = _resolve_weight_bits(q_model, model_name, custom_config)

    if model_name in ("visual", "vision_model"):
        cfg = _get_model_cfg(q_model, model_name)
        image_height = getattr(cfg, "image_height", None) or getattr(cfg, "image_size", None)
        image_width = getattr(cfg, "image_width", None) or getattr(cfg, "image_size", None)
        core_num = _get_part_core_num(custom_config, model_name)
        return f"{base_name}_vision_{image_width}x{image_height}_w{w_bits}_{march}_corenum_{core_num}.hbm"

    elif model_name in ("prefill", "decode"):
        model_list = custom_config.model.model_list
        fuse_prefill_and_decode = "lm" in model_list or ("prefill" in model_list and "decode" in model_list)
        # In both lm and [prefill, decode] modes, prefill/decode share the same
        # text_config (lm mode: same object; pd mode: deepcopy with identical values),
        # so reading chunk_size/cache_len from prefill config is always correct.
        if fuse_prefill_and_decode:
            model_name = "prefill"
        cfg = _get_model_cfg(q_model, model_name)
        chunk_size = getattr(cfg, "max_lm_input_len", None)
        cache_len = getattr(cfg, "max_kvcache_len", None)
        core_num = _get_part_core_num(custom_config, model_name)
        if fuse_prefill_and_decode:
            core_num_decode = _get_part_core_num(custom_config, "decode")
            core_num = f"{core_num}_{core_num_decode}"

        return f"{base_name}_language_chunk_{chunk_size}_cache_{cache_len}_w{w_bits}_{march}_corenum_{core_num}.hbm"

    elif model_name == "eagle3":
        cfg = _get_model_cfg(q_model, "prefill")
        cache_len = getattr(cfg, "max_kvcache_len", None)
        core_num = _get_part_core_num(custom_config, model_name)
        return f"{base_name}_eagle3_cache_{cache_len}_w{w_bits}_{march}_corenum_{core_num}.hbm"

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def _transform_inputs(inputs, func: callable):
    """Recursively apply func to tensors in nested list/tuple."""
    if isinstance(inputs, torch.Tensor):
        return func(inputs)
    if isinstance(inputs, (list, tuple)):
        transformed = [_transform_inputs(x, func) for x in inputs]
        return type(inputs)(transformed)
    return inputs


def _forward_supports_return_all_logits(model: nn.Module) -> bool:
    return "return_all_logits" in inspect.signature(model.forward).parameters


class PrefillAllLogitsExportWrapper(nn.Module):
    """Export wrapper that forces prefill to emit full-sequence logits.

    HBIR export traces a single forward path. This wrapper pins
    ``return_all_logits=True`` so the compiled prefill graph outputs
    ``[batch, seq_len, vocab]`` logits for PPL evaluation.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        if not _forward_supports_return_all_logits(model):
            raise ValueError(
                "compile.return_all_logits is enabled for prefill, but the model "
                f"forward() does not accept return_all_logits: {type(model).__name__}"
            )
        self.model = model
        # Keep wrapper mode consistent with wrapped model to avoid export/eval behavior mismatch.
        self.train(model.training)

    def get_input_embeddings(self):
        inner = self.model
        if hasattr(inner, "get_input_embeddings"):
            return inner.get_input_embeddings()
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        return None

    def forward(self, *args, **kwargs):
        kwargs.pop("return_all_logits", None)
        return self.model(*args, return_all_logits=True, **kwargs)


def _resolve_return_all_logits_for_prefill(compile_config) -> bool:
    """Resolve prefill ``return_all_logits`` from compile or compile.prefill.

    ``compile.prefill.return_all_logits`` overrides the top-level
    ``compile.return_all_logits`` when present, matching Calibration2Hbm's
    model-specific config merge semantics.
    """
    value = getattr(compile_config, "return_all_logits", None)
    prefill_cfg = getattr(compile_config, "prefill", None)
    if prefill_cfg is not None:
        prefill_value = getattr(prefill_cfg, "return_all_logits", None)
        if prefill_value is not None:
            value = prefill_value
    return bool(value)


def _maybe_wrap_prefill_all_logits(model: nn.Module, model_part: str, compile_config) -> nn.Module:
    if model_part != "prefill" or not _resolve_return_all_logits_for_prefill(compile_config):
        return model
    logger.info(f"Exporting {model_part} with return_all_logits=True for full-sequence logits")
    return PrefillAllLogitsExportWrapper(model)


class Calibration2Hbm:
    """Convert calibration model to HBM (Horizon Binary Model).

    This class handles the conversion from calibration model to HBM format,
    including export, convert, and compile steps. Compilation settings are
    automatically retrieved from q_model and custom_config.

    Args:
        q_model: The quantized model wrapper (BaseQModel instance). Model-specific
            settings (example_inputs, input_names, output_names) will be automatically
            retrieved from q_model.
        model_part: Name of the model part (visual, prefill, decode, etc.).
        custom_config: Configuration object containing compilation settings.
    """

    def __init__(
        self,
        q_model,
        model_part,
        custom_config,
    ):
        from easydict import EasyDict

        if q_model is None or model_part is None or custom_config is None:
            raise ValueError("q_model, model_part, and custom_config are required")

        # Automatically retrieve settings from q_model
        self.example_inputs = q_model.get_model_trace_dummy_input(model_part)
        self.input_names, self.output_names = q_model.get_model_input_output_name(model_part)

        # Get configuration values
        self.march = custom_config.model.march
        self.save_path = custom_config.compile.hbm_save_path
        self.model_part = model_part

        compile_config = custom_config.compile
        self.compile_config = EasyDict(compile_config.__dict__)

        # Update with model-specific config if exists (model-specific config overrides general config)
        model_specific_config = getattr(compile_config, model_part, None)
        if model_specific_config is not None:
            model_specific_dict = {k: v for k, v in model_specific_config.items()}
            logger.info(f"{model_part} Model-specific config: {model_specific_dict}")
            self.compile_config.update(model_specific_dict)

    def _warn_cpu_ops(self, op_stats):
        for func_stats in op_stats:
            cpu_ops = {k: v for k, v in func_stats.items() if k.startswith("hbtl.call")}
            if cpu_ops:
                sep = "!" * 60
                lines = [f"  {op}: {count}" for op, count in cpu_ops.items()]
                logger.warning(
                    f"\n{sep}\n"
                    f"[{self.model_part}] CPU ops detected in converted model:\n"
                    f"{chr(10).join(lines)}\n"
                    f"These ops will run on CPU instead of BPU in HBM!\n"
                    f"{sep}"
                )

    def __call__(self, model):
        """Convert calibration model to HBM.

        This method performs the full compilation pipeline:
        1. Export fakequant HBIR
        2. Convert to quantized model
        3. Compile to HBO format

        Args:
            model: The calibration model to convert (already prepared with calibration checkpoint)

        Returns:
            hbo: Compiled HBO object
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        model.eval()

        # Set fake quantize to validation mode
        horizon.quantization.set_fake_quantize(model, horizon.quantization.FakeQuantState.VALIDATION)
        model = _maybe_wrap_prefill_all_logits(model, self.model_part, self.compile_config)
        # Step 1: Export fakequant HBIR
        export_kwargs = dict(
            name=self.model_part,
            input_names=self.input_names,
            output_names=self.output_names,
            native_pytree=False,
        )
        try:
            model = model.float()
            example_inputs = _transform_inputs(
                self.example_inputs,
                lambda x: x.float() if (isinstance(x, torch.Tensor) and torch.is_floating_point(x)) else x,
            )
            model = export(model, example_inputs, **export_kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"GPU OOM during export for {self.model_part}, falling back to CPU")
            torch.cuda.empty_cache()
            model = model.cpu().float()
            example_inputs = _transform_inputs(
                self.example_inputs,
                lambda x: (
                    x.cpu().float()
                    if (isinstance(x, torch.Tensor) and torch.is_floating_point(x))
                    else (x.cpu() if isinstance(x, torch.Tensor) else x)
                ),
            )
            model = export(model, example_inputs, **export_kwargs)
        model._llm_extra = True
        model._high_precision_qpp = False
        model._skip_move_cpu_ops_pass = True
        save(model, os.path.join(self.save_path, f"{self.model_part}.bc"))
        # Step 2: Convert to quantized model
        rmsnorm_version = getattr(self.compile_config, "rmsnorm_version", "cuda")
        enable_vpu = getattr(self.compile_config, "enable_vpu", True)
        enable_spu = getattr(self.compile_config, "enable_spu", True)  # noqa
        softmax_version = getattr(self.compile_config, "softmax_version", "skip")

        # dynamic_quant is only True for nash-p/nash-starry-p architecture
        if self.march in ("nash-p", "nash-starry-p"):
            model = llm_convert(
                model, march=self.march, rmsnorm_version=rmsnorm_version, softmax_version=softmax_version
            )
        model._use_f16_quant_dequant_on_vae_always = getattr(
            self.compile_config, "use_f16_quant_dequant_on_vae_always", True
        )
        quantized_model = convert(model, march=self.march, enable_vpu=enable_vpu)
        save(quantized_model, os.path.join(self.save_path, f"{self.model_part}_convert.bc"))

        quantized_model[0].remove_io_op(["Quantize", "Cast", "Dequantize"])
        op_stats = statistics(quantized_model)
        self._warn_cpu_ops(op_stats)
        save(quantized_model, os.path.join(self.save_path, f"{self.model_part}_convert_rm.bc"))

        # Step 3: Compile to HBO
        opt_level = getattr(self.compile_config, "opt_level", 2)
        jobs = getattr(self.compile_config, "jobs", 120)
        cache_path = getattr(self.compile_config, "cache_path", "./llm_cache")
        debug = getattr(self.compile_config, "debug", False)
        progress_bar = getattr(self.compile_config, "progress_bar", True)
        input_no_padding = getattr(self.compile_config, "input_no_padding", True)
        output_no_padding = getattr(self.compile_config, "output_no_padding", True)
        cache_mode = getattr(self.compile_config, "cache_mode", "enable")
        enable_hpc = getattr(self.compile_config, "enable_hpc", True)
        core_num = getattr(self.compile_config, "core_num", 1)
        max_l2m_size = getattr(self.compile_config, "max_l2m_size", 0)

        hbo_save_path = os.path.join(self.save_path, f"{self.model_part}.hbo")

        # Prepare compile arguments
        compile_kwargs = {
            "march": self.march,
            "path": hbo_save_path,
            "debug": debug,
            "opt": opt_level,
            "jobs": jobs,
            "progress_bar": progress_bar,
            "input_no_padding": input_no_padding,
            "output_no_padding": output_no_padding,
            "cache_mode": cache_mode,
            "cache_path": cache_path,
            "enable_hpc": enable_hpc,
            "core_num": core_num,
            "max_l2m_size": max_l2m_size,
        }

        hbo = compile(quantized_model, **compile_kwargs)

        return hbo


def hbo2hbm(compiled_hbos, save_path, hbm_desc=None, hbm_names=None):
    """Link HBOs to HBM files.

    This function handles linking compiled HBO objects to HBM files:
    - prefill and decode are linked together as one language HBM file
    - Other models (visual, etc.) are each linked separately
    - If hbm_names are provided, use hbm_names.

    Args:
        compiled_hbos: Dictionary of compiled HBO objects, keyed by model part name
        save_path: Directory path to save HBM files
        hbm_desc: Optional dictionary containing HBM description metadata.
            This is a single desc (not per-model-part), written only to lm.hbm.
        hbm_names: Optional dictionary mapping model part name to output HBM filename.

    Returns:
        None (HBM files are saved to disk)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    hbm_desc = hbm_desc or {}
    hbm_names = hbm_names or {}

    # Link prefill and decode together as one language HBM file.
    language_parts = ["prefill", "decode"]
    present_language_parts = [k for k in language_parts if k in compiled_hbos]
    language_hbos = [compiled_hbos.pop(k) for k in present_language_parts]
    if language_hbos:
        language_name_from = "prefill" if "prefill" in present_language_parts else "decode"
        hbm_filename = hbm_names.get(language_name_from, "lm.hbm")

        hbm_save_path = os.path.join(save_path, hbm_filename)
        link(language_hbos, hbm_save_path)
        if hbm_desc:
            m = Hbm(hbm_save_path)
            m.staged_desc = json.dumps(hbm_desc)
            save_path_tmp = os.path.join(save_path, f"{os.path.splitext(hbm_filename)[0]}_tmp.hbm")
            m.save_by_staged_info(save_path_tmp)
            os.replace(save_path_tmp, hbm_save_path)
        logger.info(f"Linked language model (prefill+decode) to {hbm_save_path}")

    # Link other models (visual, etc.) separately
    for model_part, hbo in list(compiled_hbos.items()):
        hbm_filename = hbm_names.get(model_part, f"{model_part}.hbm")

        hbm_save_path = os.path.join(save_path, hbm_filename)
        link([hbo], hbm_save_path)
        logger.info(f"Linked {model_part} model to {hbm_save_path}")


def _get_model_cfg(q_model, model_name):
    get_cfg = getattr(q_model, "get_generated_model_cfg", None)
    if callable(get_cfg):
        cfg = get_cfg(model_name)
        if cfg is not None:
            return cfg
    raise ValueError(f"Cannot get config for model part '{model_name}' from q_model")


def _get_part_core_num(custom_config, model_name):
    core_num = getattr(custom_config.compile, "core_num", 1)
    model_specific_config = getattr(custom_config.compile, model_name, None)
    if model_specific_config is not None:
        core_num = getattr(model_specific_config, "core_num", core_num)
    return core_num


def _resolve_weight_bits(q_model, model_name, custom_config=None):
    """Resolve weight bit-width for HBM naming.

    Prefer LightCompress fake_quant weight qparams (e.g. W4 from safetensors) over
    default ConvDtypeTemplate (qint8) in nashp_default_qconfig_template.
    """
    model_path = None
    if custom_config is not None:
        model_path = getattr(getattr(custom_config, "model", None), "model_path", None)
    if not model_path:
        model_path = getattr(getattr(getattr(q_model, "custom_config", None), "model", None), "model_path", None)
    if isinstance(model_path, str) and os.path.isdir(model_path):
        from llm_compression.converters.calib_converter import load_weight_qparams

        weight_qparams = load_weight_qparams(model_path)
        if weight_qparams:
            bits_set = {
                params.get("dtype", {}).get("weight").bits
                for params in weight_qparams.values()
                if params.get("dtype", {}).get("weight") is not None and hasattr(params["dtype"]["weight"], "bits")
            }
            if len(bits_set) == 1:
                return bits_set.pop()

    get_qconfig_setting = getattr(q_model, "get_qconfig_setting", None)
    if not callable(get_qconfig_setting):
        return 8
    q_templates = get_qconfig_setting(model_name)
    for q_template in q_templates:
        if isinstance(q_template, ConvDtypeTemplate):
            weight_dtype = q_template.weight_dtype
            w_bits = weight_dtype.bits
            return w_bits
    return 8


def _load_hf_configs(model_path):
    """Load all JSON config files from model_path into a dict.

    Scans the model directory for *.json files and loads each one.
    Returns a dict keyed by filename without .json extension.
    """
    result = {}
    for filename in sorted(os.listdir(model_path)):
        if filename.endswith(".json"):
            filepath = os.path.join(model_path, filename)
            key = filename[:-5]  # strip .json
            with open(filepath, encoding="utf-8") as f:
                result[key] = json.load(f)
    return result


def _resolve_token_str(token_name, base_cfg, tokenizer_config):
    """Resolve a token string from tokenizer_config.json's added_tokens_decoder.

    Strategy: find {token_name}_id in model config (e.g. image_token_id),
    then look up the corresponding content in tokenizer_config["added_tokens_decoder"].

    Returns the token string (e.g. "<|image_pad|>") or None.
    """
    token_id_attr = f"{token_name}_id"
    token_id = None
    for cfg in [base_cfg, getattr(base_cfg, "text_config", None)]:
        if cfg is not None:
            token_id = getattr(cfg, token_id_attr, None)
            if token_id is not None:
                break
    if token_id is None or not tokenizer_config:
        return None
    added_tokens = tokenizer_config.get("added_tokens_decoder", {})
    token_entry = added_tokens.get(str(token_id))
    if token_entry and isinstance(token_entry, dict):
        return token_entry.get("content")

    return None


def _set_nested_value(d, key, value):
    """Recursively set all occurrences of `key` in nested dicts to `value`."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                d[k] = value
            else:
                _set_nested_value(v, key, value)
    elif isinstance(d, list):
        for item in d:
            _set_nested_value(item, key, value)
