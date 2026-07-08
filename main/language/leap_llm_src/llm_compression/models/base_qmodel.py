import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    PropagateTemplate,
    SetDynamicQuantTemplate,
)
from torch import nn

from llm_compression.utils import AttentionManager
from llm_compression.utils.device_manager import DeviceManager


def load_state_dict_with_metadata(model, state_dict, strict=False):
    """Load state_dict into model, injecting module version metadata.

    Populates state_dict._metadata with each module's version so that
    horizon_plugin_pytorch does not emit per-module version warnings.
    """
    state_dict = OrderedDict(state_dict)
    state_dict._metadata = OrderedDict(
        (name, {"version": getattr(mod, "_version", 1)}) for name, mod in model.named_modules()
    )
    return model.load_state_dict(state_dict, strict=strict)


def update_config_from_custom_config(target_config, custom_config_dict):
    """Update target_config with attributes from custom_config_dict (EasyDict).

    Args:
        target_config: The config object to be updated
            (e.g., model_config.vision_config)
        custom_config_dict: EasyDict object containing new values
            (e.g., custom_config.model.vision_config)
    """
    if custom_config_dict is None:
        return

    for key, value in custom_config_dict.__dict__.items():
        if not key.startswith("_"):
            setattr(target_config, key, value)


def nashp_default_qconfig_template():
    default_tae_int8_vae_fp16_qconfig_template = [
        ModuleNameTemplate({"": torch.float16}),
        MatmulDtypeTemplate(
            input_dtypes=[qint8, qint8],
        ),
        ConvDtypeTemplate(
            input_dtype=qint8,
            weight_dtype=qint8,
        ),
        PropagateTemplate(),
    ]
    return default_tae_int8_vae_fp16_qconfig_template


def nashe_default_qconfig_template():
    default_tae_int8_vae_int16_qconfig_template = [
        ModuleNameTemplate({"": qint16}),
        MatmulDtypeTemplate(
            input_dtypes=[qint8, qint8],
        ),
        ConvDtypeTemplate(
            input_dtype=qint8,
            weight_dtype=qint8,
        ),
        PropagateTemplate(),
    ]
    return default_tae_int8_vae_int16_qconfig_template


class BaseQModel(ABC):
    """Base class for quantized model wrappers.

    This abstract base class defines the interface for model quantization,
    providing methods for model building, quantization configuration, and
    model tracing. Subclasses should implement the abstract methods to
    support specific model architectures.
    """

    def __init__(self, model_dir, custom_config=None) -> None:
        """Initialize the quantized model wrapper.

        Args:
            model_dir: Directory path containing the model files
            custom_config: Optional custom configuration object (EasyDict)
        """
        self.custom_config = custom_config
        AttentionManager.set(custom_config)
        self.generated_model = self.build_model(model_dir)
        self._apply_vocab_compression()
        self.device_manager = DeviceManager(self.generated_model, custom_config.model.model_list)

    def _apply_vocab_compression(self):
        """Compress lm_head by keeping only tokens specified in kept_tokens_file.

        When kept_tokens_file is configured in YAML, this method slices
        lm_head.weight to retain only the specified token rows, reducing
        the output vocabulary size.
        """
        kept_tokens_file = getattr(self.custom_config.model, "kept_tokens_file", None)
        if kept_tokens_file is None:
            return

        with open(kept_tokens_file, encoding="utf-8") as f:
            kept_tokens = json.load(f)

        token_ids = kept_tokens["token_ids"] + kept_tokens["added_tokens_ids"]

        for name, module in self.generated_model.named_modules():
            if name.endswith("lm_head") and hasattr(module, "weight"):
                original_vocab_size = module.weight.shape[0]
                module.weight = nn.Parameter(module.weight.data[token_ids])
                module.out_features = len(token_ids)
                logging.info(
                    f"Vocab compression applied to {name}: " f"{original_vocab_size} -> {len(token_ids)} tokens"
                )

    def is_shared_lm_mode(self):
        """Check if shared LM mode is enabled based on model_list.

        In shared LM mode, model_list contains 'lm' instead of 'prefill' and 'decode'.
        The prefill and decode share the same model instance during calibration,
        and deepcopy is deferred to compile stage.

        Returns:
            bool: True if model_list contains 'lm', False otherwise.
        """
        if self.custom_config is None:
            return False
        model_list = getattr(self.custom_config.model, "model_list", [])
        return "lm" in model_list

    def setup_decode_model(self, decode_model):  # noqa: B027
        """Apply decode-specific forward wrapper after deepcopy in compile stage.

        In shared-LM mode, compile.py deepcopies the LM model for decode.
        Subclasses that need a different forward signature for decode
        (e.g. to exclude deepstack_visual_embeds) should override this method.
        """
        pass

    @abstractmethod
    def build_model(self, model_dir):
        """Build the original model from model directory.

        Args:
            model_dir: Directory path containing the model files

        Returns:
            Model instance with generate method for inference
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_trace_dummy_input(self, model_part):
        """Get dummy input for tracing a specific model part.

        Args:
            model_part: Name of the model part
                (e.g., 'visual', 'prefill', 'decode')

        Returns:
            List of example inputs for model tracing
        """
        raise NotImplementedError

    @abstractmethod
    def get_generated_model(self):
        """Get the VLM model with generate(inputs) function for calibration.

        Returns:
            Model instance with generate method for inference
        """
        raise NotImplementedError

    def get_model_input_output_name(self, model_part):
        """Get input and output names for a specific model part.

        Args:
            model_part: Name of the model part
                (e.g., 'visual', 'prefill', 'decode')

        Returns:
            tuple: (input_names, output_names) lists of input/output names
        """
        input_names = []
        output_names = []
        return input_names, output_names

    def input_preprocess(self, message):
        """
        Convert a message into a format compatible with model.generate.
        Processing may vary across models and should be customized.
        """
        return message

    def output_postprocess(self, output):
        """
        Postprocess the output of the model.

        """
        raise output

    def get_kvcache_names(self, model_name):
        """Return the module names of KV-cache fake-quant stubs (cache_k_fq / cache_v_fq).

        These names serve two purposes:
        - In get_qconfig_setting: configure their output dtype as int8.
        - In sync_kvcache_scales: locate the activation_post_process submodule
          and unify scales between prefill and decode after calibration.
        """
        raise NotImplementedError("get_kvcache_names not implemented")

    def get_qconfig_setting(self, model_part):
        """Get quantization configuration settings for a specific model part.

        Args:
            model_part: Name of the model part
                (e.g., 'visual', 'prefill', 'decode')

        Returns:
            q_template: List of quantization templates
        """
        if get_march() in ("nash-p", "nash-starry-p"):
            q_template = nashp_default_qconfig_template()
            q_template.append(
                SetDynamicQuantTemplate(
                    op_kwargs={
                        nn.Linear: {"block_size": "full", "dim": -1},
                    }
                )
            )
        elif get_march() in ("nash-e", "nash-m"):
            q_template = nashe_default_qconfig_template()
        else:
            raise ValueError(f"Unsupported march {get_march()}")
        return q_template
