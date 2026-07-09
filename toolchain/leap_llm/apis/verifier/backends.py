from __future__ import annotations

import os
import re
from collections import OrderedDict
from itertools import islice

import numpy as np
import torch
from hbdk4.compiler import Hbm, load
from hbdk4.compiler.hbm import Graph
from hbdk4.compiler.overlay import Module, Value
from transformers import AutoModelForCausalLM, AutoTokenizer

from leap_llm.apis.calibration import CalibrationDataPreparer
from leap_llm.apis.calibration.data_loader import (
    load_image_data,
    load_message_data,
    load_text_data,
)
from leap_llm.apis.verifier.internvl3_5_wrappers import (
    InternVL3_5LlmWrapper,
    InternVL3_5VisionWrapper,
    prepare_internvl35_inputs,
)
from leap_llm.apis.verifier.qwen2_5_vl_wrappers import (
    Qwen2_5VLLlmWrapper,
    Qwen2_5VLVisionWrapper,
    prepare_qwen2_5_vl_inputs,
    preprocess_image_for_qwen2_5_vl,
)
from leap_llm.apis.verifier.types import TensorDict, TensorInfo, VerifierArgs
from leap_llm.apis.verifier.utils import cast_to_tensor_info
from leap_llm.models.deepseek.model import DeepSeek
from leap_llm.models.internvl3_5.model import InterVL3_5
from leap_llm.models.internvl_1b.model import Internlm1b, Internvl1bVision
from leap_llm.models.internvl_2b.model import Internlm2b, Internvl2bVision
from leap_llm.models.qwen2_5_vl.model import Qwen2_5_VL
from leap_llm.nn.modules.linear import DynamicQuantLinear, FakeQuantLinear
from leap_llm.nn.modules.matmul import DynamicQuantMatmul, FakeQuantMatmul

DEFAULT_CONVERSATION_NUM = 2  # 默认输入的 message 消息数量
DEEPSEEK_MODELS = ["deepseek-qwen-1_5b", "deepseek-qwen-7b"]
INTERNVL_MODELS = [
    "internvl2-1b",
    "internvl2-2b",
    "internvl2_5-1b",
    "internvl2_5-2b",
    "internvl3_5-1b",
]
QWEN2_5_VL_MODELS = ["qwen2_5-vl-3b", "qwen2_5-vl-7b"]
padding_side_dict = {
    "internvl2-1b": "left",
    "internvl2-2b": "left",
    "internvl2_5-1b": "left",
    "internvl2_5-2b": "left",
    "internvl3_5-1b": "left",
    "deepseek-qwen-1_5b": "left",
    "deepseek-qwen-7b": "left",
    "qwen2_5-vl-3b": "left",
    "qwen2_5-vl-7b": "left",
}

mask_value_dict = {
    "internvl2-1b": -8192,
    "internvl2-2b": -8192,
    "internvl2_5-1b": -8192,
    "internvl2_5-2b": -8192,
    "internvl3_5-1b": -512,
    "deepseek-qwen-1_5b": -512,
    "deepseek-qwen-7b": -512,
    "qwen2_5-vl-3b": -512,
    "qwen2_5-vl-7b": -32768,
}

pos_mask_value_dict = {
    "internvl2-1b": 0,
    "internvl2-2b": 0,
    "internvl2_5-1b": 0,
    "internvl2_5-2b": 0,
    "internvl3_5-1b": 1,
    "deepseek-qwen-1.5b": 1,
    "deepseek-qwen-7b": 1,
    "qwen2_5-vl-3b": 1,
    "qwen2_5-vl-7b": 1,
}

data_type_dict = {
    "internvl2-1b": torch.float16,
    "internvl2-2b": torch.float16,
    "internvl2_5-1b": torch.float16,
    "internvl2_5-2b": torch.float16,
    "internvl3_5-1b": torch.float16,
    "deepseek-qwen-1.5b": torch.float32,
    "deepseek-qwen-7b": torch.float16,
    "qwen2_5-vl-3b": torch.float16,
    "qwen2_5-vl-7b": torch.float16,
}

internvl_model_class_dict = {
    "llm": {
        "internvl2-1b": Internlm1b,
        "internvl2-2b": Internlm2b,
        "internvl2_5-1b": Internlm1b,
        "internvl2_5-2b": Internlm2b,
        "internvl3_5-1b": InternVL3_5LlmWrapper,
    },
    "vlm": {
        "internvl2-1b": Internvl1bVision,
        "internvl2-2b": Internvl2bVision,
        "internvl2_5-1b": Internvl1bVision,
        "internvl2_5-2b": Internvl2bVision,
        "internvl3_5-1b": InternVL3_5VisionWrapper,
    },
}


class Backend:
    """Unified backend that handles inference for Torch, BC, and HBM models."""

    def __init__(self, args: VerifierArgs):
        self.args = args
        self.device = self.args.device

        self.torch_llm_model = None
        self.torch_vlm_model = None
        self.torch_llm_model_core = None
        self.torch_vlm_model_core = None
        self.tokenizer = None
        self.calib_data_preparer: CalibrationDataPreparer | None = None
        self.torch_layers_outputs: TensorDict = OrderedDict()
        self.torch_vlm_layers_outputs: TensorDict = OrderedDict()
        self.num_hidden_layers = None

        self.bc_model: Module | None = None
        self.bc_vlm_model: Module | None = None
        self.bc_layers_outputs: TensorDict = OrderedDict()
        self.bc_vlm_layers_outputs: TensorDict = OrderedDict()
        self._hbm_llm_module = None
        self._hbm_vlm_module = None
        self.key_value_groups = 1

        self._load_torch_model()
        self.bc_model = self._load_bc_model(self.args.quant_llm_model_path)
        self.bc_vlm_model = self._load_bc_model(self.args.quant_vlm_model_path)
        self._hbm_llm_module = self._load_hbm_module(self.args.hbm_llm_model_path)
        self._hbm_vlm_module = self._load_hbm_module(self.args.hbm_vlm_model_path)

        self.calib_data_preparer = CalibrationDataPreparer(
            self.args.model_name,
            self.args.model_dir,
            seq_len=self.args.chunk_size,
            kv_cache_len=self.args.cache_len,
            device=self.device,
            transpose_cache=self.args.transpose_cache,
            mask_value=mask_value_dict[self.args.model_name],
            pos_mask_value=pos_mask_value_dict[self.args.model_name],
            data_type=data_type_dict[self.args.model_name],
            padding_side=padding_side_dict[self.args.model_name],
        )

        self.special_name = []

    def _load_text_data(
        self,
        model_type: str = "",
        input_json_path: str = "",
        input_text_path: str = "",
    ):
        if model_type in QWEN2_5_VL_MODELS:
            message_data = load_message_data(input_json_path, model_type)
            if not input_json_path:
                # defalut: use only the first 10 messages in mmstar
                message_data = islice(message_data, DEFAULT_CONVERSATION_NUM)
            for messages in message_data:
                for message in messages:
                    content = message.get("content", [])
                    for item in content:
                        if item["type"] == "text":
                            if not isinstance(item["text"], str):
                                raise ValueError("Invalid prompt entry, 'text' must be a string.")
                            yield item["text"]
        else:
            yield from load_text_data(input_text_path)

    def _load_image_data(
        self,
        model_type: str = "",
        input_json_path: str = "",
        input_image_path: str = "",
        image_width: int = 448,
        image_height: int = 448,
        max_num: int = 1,
    ):
        if model_type in QWEN2_5_VL_MODELS:
            message_data = load_message_data(input_json_path, model_type)
            image_paths = []
            if not input_json_path:
                message_data = islice(message_data, DEFAULT_CONVERSATION_NUM)
            for messages in message_data:
                for message in messages:
                    content = message.get("content", [])
                    for item in content:
                        if item["type"] == "image":
                            image_paths.append(item["image"])
            for image_path in image_paths:
                yield preprocess_image_for_qwen2_5_vl(
                    image=image_path,
                    target_height=image_height,
                    target_width=image_width,
                )
        else:
            yield from load_image_data(input_image_path, max_num=max_num)

    def _load_torch_model(self):
        """Load the torch model."""
        if not os.path.isdir(self.args.model_dir):
            raise ValueError(f"Model directory not found: {self.args.model_dir}")

        is_internvl35 = self.args.model_name == "internvl3_5-1b"
        is_qwen2_5_vl = self.args.model_name in QWEN2_5_VL_MODELS
        model = None
        checkpoint = None
        if not is_internvl35 and not is_qwen2_5_vl:
            model = AutoModelForCausalLM.from_pretrained(self.args.model_dir, trust_remote_code=True)
            checkpoint = model.state_dict()

        try:
            if self.args.model_name in DEEPSEEK_MODELS:
                self.torch_llm_model = DeepSeek.build(
                    input_model_path=self.args.model_dir,
                    chunk_size=self.args.chunk_size,
                    cache_len=self.args.cache_len,
                    preserve_precision=False,
                )
                self.torch_llm_model.model.compile_mode(False)
                self.torch_llm_model.model.to(self.device)
            elif self.args.model_name in INTERNVL_MODELS:
                if self.args.model_name == "internvl3_5-1b":
                    intervl = InterVL3_5.build(
                        self.args.model_dir,
                        chunk_size=self.args.chunk_size,
                        cache_len=self.args.cache_len,
                    )
                    self.torch_llm_model = internvl_model_class_dict["llm"][self.args.model_name].load_model(
                        input_model_path=self.args.model_dir,
                        checkpoint=None,
                        chunk_size=self.args.chunk_size,
                        cache_len=self.args.cache_len,
                        kept_tokens_file=self.args.kept_tokens_file,
                        prebuilt=intervl,
                    )
                    self.torch_vlm_model = internvl_model_class_dict["vlm"][self.args.model_name].load_model(
                        input_model_path=self.args.model_dir,
                        checkpoint=None,
                        prebuilt=intervl,
                    )
                else:
                    self.torch_llm_model = internvl_model_class_dict["llm"][self.args.model_name].load_model(
                        input_model_path=self.args.model_dir,
                        checkpoint=checkpoint,
                        chunk_size=self.args.chunk_size,
                        cache_len=self.args.cache_len,
                        kept_tokens_file=self.args.kept_tokens_file,
                    )
                    self.torch_vlm_model = internvl_model_class_dict["vlm"][self.args.model_name].load_model(
                        input_model_path=self.args.model_dir,
                        checkpoint=checkpoint,
                    )

                self._set_compile_and_device(self.torch_llm_model, self.device)
                self._set_compile_and_device(self.torch_vlm_model, self.device)

            elif self.args.model_name in QWEN2_5_VL_MODELS:
                # Use provided image dimensions or default vision config values
                from leap_llm.models.qwen2_5_vl.model import Qwen2_5_VLVisionConfig

                default_vision_config = Qwen2_5_VLVisionConfig()
                image_width = (
                    self.args.image_width if self.args.image_width is not None else default_vision_config.image_width
                )
                image_height = (
                    self.args.image_height if self.args.image_height is not None else default_vision_config.image_height
                )
                qwen_wrapper = Qwen2_5_VL.build(
                    model_dir=self.args.model_dir,
                    chunk_size=self.args.chunk_size,
                    cache_len=self.args.cache_len,
                    input_model_format="llmc",
                    image_width=image_width,
                    image_height=image_height,
                )
                self.torch_llm_model = Qwen2_5VLLlmWrapper(
                    qwen_wrapper.get_text_model(),
                    qwen_wrapper.model_args,
                    chunk_size=self.args.chunk_size,
                    cache_len=self.args.cache_len,
                )
                self.torch_vlm_model = Qwen2_5VLVisionWrapper(
                    qwen_wrapper.get_visual_model(),
                    qwen_wrapper.model_args,
                )
                self._set_compile_and_device(self.torch_llm_model, self.device)
                self._set_compile_and_device(self.torch_vlm_model, self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_dir, trust_remote_code=True)
        except Exception as err:
            raise Exception(f"Failed to load {self.args.model_name} model: {err}") from err

        if self.args.model_name == "internvl3_5-1b":
            self.torch_llm_model_core = self.torch_llm_model
            self.torch_vlm_model_core = self.torch_vlm_model
        elif self.args.model_name in QWEN2_5_VL_MODELS:
            # Qwen2.5-VL wrappers are the core models
            self.torch_llm_model_core = self.torch_llm_model
            self.torch_vlm_model_core = self.torch_vlm_model
        else:
            self.torch_llm_model_core = (
                self.torch_llm_model.model
                if self.torch_llm_model is not None and hasattr(self.torch_llm_model, "model")
                else self.torch_llm_model
            )
            self.torch_vlm_model_core = (
                self.torch_vlm_model.model
                if self.torch_vlm_model is not None and hasattr(self.torch_vlm_model, "model")
                else self.torch_vlm_model
            )

        model_args = self.torch_llm_model.get_model_args()
        self.key_value_groups = model_args.num_attention_heads // model_args.num_key_value_heads

    def _load_bc_model(self, model_path: str) -> Module | None:
        """Load BC model from path."""
        if not model_path:
            return None
        return load(model_path)

    def _load_hbm_module(self, path: str) -> Graph | None:
        """Load HBM module from path."""
        if not path:
            return None

        return Hbm(path)[0]

    def _set_compile_and_device(self, module, device: str):
        if module is None:
            return
        target = module.model if hasattr(module, "model") else module
        if hasattr(target, "compile_mode"):
            target.compile_mode(False)
        if hasattr(target, "to"):
            target.to(device)

    def _extract_named_cache_tensor(self, outputs: dict, cache_index: int, num_layers: int | None):
        if (
            num_layers is None
            or not isinstance(outputs, dict)
            or ("out_key_cache_0" not in outputs and "out_value_cache_0" not in outputs)
        ):
            return None

        if cache_index < num_layers:
            return outputs.get(f"out_key_cache_{cache_index}")
        else:
            value_index = cache_index - num_layers
            return outputs.get(f"out_value_cache_{value_index}")

    def _extract_llm_output_tensor(self, outputs):
        if isinstance(outputs, dict):
            if "_output_0" in outputs:
                return outputs["_output_0"]
            if "out_hidden_states" in outputs:
                return outputs["out_hidden_states"]
            first_key = next(iter(outputs), None)
            if first_key is not None:
                return outputs[first_key]
        return outputs

    def compute_last_valid_step_index(self, text_input: str) -> int:
        """Return the compare index within the last chunk (0-based), minimal cost."""
        padding_side = padding_side_dict[self.args.model_name]
        if padding_side == "left":
            return self.args.chunk_size - 1

        if self.calib_data_preparer is None:
            raise ValueError("CalibrationDataPreparer is not initialized")

        (
            _,
            _,
            position_ids_chunks,
            _,
        ) = self.calib_data_preparer.prepare_inputs(text_input)
        return int(torch.argmax(position_ids_chunks[-1])) % self.args.chunk_size

    def _get_compare_step_index(self, text_input: str) -> int:
        if self.calib_data_preparer is None:
            raise ValueError("CalibrationDataPreparer is not initialized")

        if not self.calib_data_preparer.full_logits:
            return self.compute_last_valid_step_index(text_input)

        return self.calib_data_preparer.vaild_idx

    def _register_torch_hooks(self, model, model_type: str):
        """Register hooks for Torch model."""
        hooks: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            is_special_op = False
            if isinstance(module, (FakeQuantLinear, FakeQuantMatmul, DynamicQuantMatmul, DynamicQuantLinear)):
                is_special_op = True
                self.special_name.append(name)
            if len(list(module.children())) == 0 or is_special_op:

                def hook_fn(module_, input_, output, layer_name=name):
                    target_dict = self.torch_vlm_layers_outputs if model_type == "vlm" else self.torch_layers_outputs
                    if isinstance(output, torch.Tensor):
                        target_dict[layer_name] = TensorInfo(output, layer_name)
                    elif hasattr(output, "last_hidden_state"):
                        target_dict[layer_name] = TensorInfo(output.last_hidden_state, layer_name)
                    elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                        target_dict[layer_name] = TensorInfo(output[0], layer_name)

                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        return hooks

    def update_kv_cache(
        self,
        past_key_value_list: list[torch.Tensor],
        outputs: dict,
        chunk_size: int,
        transpose_cache: bool = True,
        num_layers: int | None = None,
        convert_from_numpy: bool = False,
    ) -> None:
        """Update KV cache with new outputs.

        Args:
            past_key_value_list: List of past KV tensors to update in-place
            outputs: Dict containing new cache values
            chunk_size: Size of current chunk
            transpose_cache: Whether cache is transposed
            num_layers: Number of model layers (if None, use len(past_key_value_list))
            convert_from_numpy: Whether to convert outputs from numpy
        """
        cache_count = num_layers * 2 if num_layers else len(past_key_value_list)

        for z in range(cache_count):
            if isinstance(outputs, dict):
                output_key = f"_output_{z + 1}"
                if output_key in outputs:
                    new_cache = outputs[output_key]
                else:
                    new_cache = self._extract_named_cache_tensor(outputs, z, num_layers)
                    if new_cache is None:
                        continue
            else:
                new_cache = outputs[z + 1]

            if convert_from_numpy and isinstance(new_cache, np.ndarray):
                new_cache = torch.from_numpy(new_cache)

            past = past_key_value_list[z]

            if isinstance(new_cache, torch.Tensor) and new_cache.dtype != past.dtype:
                new_cache = new_cache.to(past.dtype)

            if hasattr(new_cache, "device") and new_cache.device != past.device:
                new_cache = new_cache.to(past.device)

            if (
                isinstance(new_cache, torch.Tensor)
                and isinstance(past, torch.Tensor)
                and new_cache.dim() == 4
                and past.dim() == 4
            ):
                slice_past = past[:, chunk_size:, :, :]
                updated_cache = torch.cat([slice_past, new_cache], dim=1)
            elif transpose_cache:
                slice_past = past[chunk_size:]
                updated_cache = torch.cat([slice_past, new_cache], dim=0)
            else:
                slice_past = past[:, chunk_size:]
                updated_cache = torch.cat([slice_past, new_cache], dim=-1)

            past_key_value_list[z] = updated_cache

    def prepare_llm_chunk_inputs(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_value_list: list[torch.Tensor],
        model_meta: list,
        model_name: str,
        torch_llm_model=None,
    ) -> dict:
        """Prepare inputs for LLM inference in a unified way.

        Args:
            input_ids: Token IDs tensor
            position_ids: Position IDs tensor
            attn_mask: Attention mask tensor
            past_key_value_list: List of past KV cache tensors
            model_meta: List of (name, shape, dtype) tuples for model inputs
            model_name: Name of the model
            torch_llm_model: Torch LLM model instance (needed for INTERNVL embeddings)

        """
        input_dict = {}

        if model_name in INTERNVL_MODELS:
            embedding_layer = None
            if hasattr(torch_llm_model, "get_input_embeddings"):
                embedding_layer = torch_llm_model.get_input_embeddings()
            else:
                embedding_layer = getattr(torch_llm_model.model, "embed_tokens", None) or getattr(
                    torch_llm_model.model, "tok_embeddings", None
                )
            if not embedding_layer:
                raise AttributeError("Model lacks embedding layer.")
            with torch.no_grad():
                tok_embs = embedding_layer(input_ids.to(embedding_layer.weight.device))
            if model_name == "internvl3_5-1b":
                input0 = tok_embs
                input1 = position_ids.unsqueeze(0) if position_ids.dim() == 1 else position_ids
                input2 = attn_mask
            else:
                input0 = tok_embs.squeeze(0) if tok_embs.dim() == 3 else tok_embs
                input1 = position_ids.squeeze(0) if position_ids.dim() == 2 else position_ids
                input2 = attn_mask.squeeze(0) if attn_mask.dim() == 3 else attn_mask
        elif model_name in QWEN2_5_VL_MODELS:
            # Qwen2.5-VL uses inputs_embeds directly (input_ids is already embeddings)
            # input_ids shape: [batch, seq_len, hidden_size]
            # position_ids shape: [batch, 3, seq_len] for prefill
            # attn_mask shape: [batch, seq_len, cache_len]
            # BC/HBM artifacts for Qwen2.5-VL keep the batch dimension on all
            # primary inputs, so do not squeeze the leading axis here.
            input0 = input_ids
            input1 = position_ids
            input2 = attn_mask
        else:
            input0, input1, input2 = input_ids, position_ids, attn_mask

        primary_inputs = [input0, input1, input2]
        for i, input_data in enumerate(primary_inputs):
            name, shape, dtype = model_meta[i]
            input_dict[name] = input_data.detach().cpu().numpy().astype(dtype)

        for idx in range(3, len(model_meta)):
            name, _, dtype = model_meta[idx]
            input_data = past_key_value_list[idx - 3]
            input_dict[name] = input_data.detach().cpu().numpy().astype(dtype)

        return input_dict

    def run_llm(self, text: str):
        """Run LLM inference using the appropriate backend based on args."""
        if self.args.compare_mode == "bc":
            return self._run_bc_llm_inference(text)
        elif self.args.compare_mode == "hbm":
            return self._run_hbm_llm_inference(text)
        else:
            raise ValueError(f"Unsupported compare mode: {self.args.compare_mode}")

    def run_vlm(self, image: torch.Tensor):
        """Run VLM inference using the appropriate backend based on args."""
        if self.args.compare_mode == "bc":
            return self._run_bc_vlm_inference(image)
        elif self.args.compare_mode == "hbm":
            return self._run_hbm_vlm_inference(image)
        else:
            raise ValueError(f"Unsupported compare mode: {self.args.compare_mode}")

    def _run_torch_llm(self, text: str):
        """Run LLM inference using PyTorch backend."""
        self.torch_layers_outputs.clear()
        hooks = self._register_torch_hooks(self.torch_llm_model_core, "llm")
        try:
            outputs = self._run_torch_llm_inference(text)
        finally:
            if hooks:
                for hook in hooks:
                    hook.remove()
        step_idx = self._get_compare_step_index(text)
        outputs = cast_to_tensor_info(outputs[0])
        outputs = self._slice_tensor_dict_at_step(outputs, step_idx)
        layers = self._slice_tensor_dict_at_step(self.torch_layers_outputs, step_idx)
        return outputs, layers

    def _run_torch_vlm(self, image: torch.Tensor):
        """Run VLM inference using PyTorch backend."""
        self.torch_vlm_layers_outputs.clear()
        if self.torch_vlm_model_core is None:
            raise ValueError("Torch VLM model core not loaded")

        hooks = self._register_torch_hooks(self.torch_vlm_model_core, "vlm")
        try:
            outputs = self._run_torch_vlm_inference(image)
        finally:
            if hooks:
                for hook in hooks:
                    hook.remove()
        return outputs, self.torch_vlm_layers_outputs

    def _run_torch_llm_inference(self, text_input: str):
        if self.args.model_name in QWEN2_5_VL_MODELS:
            # Use Qwen2.5-VL specific input preparation
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = prepare_qwen2_5_vl_inputs(
                text_input=text_input,
                tokenizer=self.tokenizer,
                llm_wrapper=self.torch_llm_model,
                chunk_size=self.args.chunk_size,
                cache_len=self.args.cache_len,
                device=self.device,
                mask_value=mask_value_dict[self.args.model_name],
            )
            model_args = self.torch_llm_model.get_model_args()
            num_layers = model_args.num_hidden_layers

            for inputs_embeds, attention_mask, position_ids in zip(
                input_chunks, causal_mask_chunks, position_ids_chunks, strict=False
            ):
                inputs_embeds = inputs_embeds.to(self.device)
                position_ids = position_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                with torch.no_grad():
                    outputs = self.torch_llm_model_core.forward(
                        inputs_embeds, position_ids, attention_mask, past_key_value_list
                    )

                self.update_kv_cache(
                    past_key_value_list,
                    outputs,
                    chunk_size=inputs_embeds.shape[1],
                    num_layers=num_layers,
                    convert_from_numpy=False,
                )

            return outputs

        if self.args.model_name == "internvl3_5-1b":
            # Use InternVL3.5-1b specific input preparation
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = prepare_internvl35_inputs(
                text_input=text_input,
                tokenizer=self.tokenizer,
                llm_wrapper=self.torch_llm_model,
                chunk_size=self.args.chunk_size,
                cache_len=self.args.cache_len,
                device=self.device,
                mask_value=mask_value_dict[self.args.model_name],
                pos_mask_value=pos_mask_value_dict[self.args.model_name],
                padding_side=padding_side_dict[self.args.model_name],
            )
            model_args = self.torch_llm_model.get_model_args()
            num_layers = model_args.num_hidden_layers

            for input_ids, attention_mask, position_ids in zip(
                input_chunks, causal_mask_chunks, position_ids_chunks, strict=False
            ):
                input_ids = input_ids.to(self.device)
                position_ids = position_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                with torch.no_grad():
                    outputs = self.torch_llm_model_core.forward(
                        input_ids, position_ids, attention_mask, past_key_value_list
                    )

                self.update_kv_cache(
                    past_key_value_list,
                    outputs,
                    chunk_size=input_ids.shape[-1],
                    num_layers=num_layers,
                    convert_from_numpy=False,
                )

            return outputs

        # Default path for other models
        if self.calib_data_preparer is None:
            raise ValueError("CalibrationDataPreparer is not initialized.")

        (
            input_chunks,
            causal_mask_chunks,
            position_ids_chunks,
            past_key_value_list,
        ) = self.calib_data_preparer.prepare_inputs(text_input)
        past_key_value_list = [t.to(self.device) for t in past_key_value_list]

        for input_ids, attention_mask, position_ids in zip(
            input_chunks, causal_mask_chunks, position_ids_chunks, strict=False
        ):
            input_ids = input_ids.to(self.device)
            position_ids = position_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                if (
                    self.args.model_name == "internvl2-1b"
                    or self.args.model_name == "internvl2-2b"
                    or self.args.model_name == "internvl2_5-1b"
                    or self.args.model_name == "internvl2_5-2b"
                ):
                    input_embeds = self.torch_llm_model_core.get_embedding(input_ids)
                    outputs = self.torch_llm_model_core.forward(
                        input_embeds, position_ids, attention_mask, past_key_value_list
                    )
                else:
                    outputs = self.torch_llm_model_core.forward(
                        input_ids, position_ids, attention_mask, past_key_value_list
                    )

            self.update_kv_cache(
                past_key_value_list,
                outputs,
                chunk_size=input_ids.shape[-1],
                transpose_cache=self.calib_data_preparer.transpose_cache,
                num_layers=self.calib_data_preparer.block_num,
                convert_from_numpy=False,
            )

        return outputs

    def _run_torch_vlm_inference(self, image_input: torch.Tensor):
        input_to_use = image_input.to(self.device)

        # For Qwen2.5-VL, convert image from [N,C,H,W] to [1, seq_len, patch_dim]
        if self.args.model_name in QWEN2_5_VL_MODELS:
            input_to_use = self.torch_vlm_model.prepare_vision_input(input_to_use)

        with torch.no_grad():
            outputs = self.torch_vlm_model_core.forward(input_to_use)
        return outputs

    def _run_bc_vlm_inference(self, image: torch.Tensor):
        """Run VLM inference using BC backend."""
        self.bc_vlm_layers_outputs.clear()
        if not self.bc_vlm_model:
            raise ValueError("BC VLM model not loaded")

        if image.dim() != 4:
            raise ValueError(f"Expected 4D image input (N,C,H,W), got {image.dim()}D")

        if self.args.model_name in QWEN2_5_VL_MODELS:
            image = self.torch_vlm_model.prepare_vision_input(image)

        self.bc_vlm_model.functions[0].register_callback(self._bc_vlm_callback)
        dtype = self.bc_vlm_model.functions[0].inputs[0].type.np_dtype
        feed_dict = {"_input_0": image.detach().cpu().numpy().astype(dtype)}
        bc_run_outputs = self.bc_vlm_model.functions[0].feed(inputs=feed_dict)
        outputs = OrderedDict([("output", bc_run_outputs["_output_0"])])
        return cast_to_tensor_info(outputs), self.bc_vlm_layers_outputs

    def _bc_callback(self, op, results, operands):
        if op.type == "func.func":
            return True
        if (
            self.check_bc_op_name(op.name)
            and len(results) > 0
            and type(results[0])
            in [
                torch.Tensor,
                np.ndarray,
                Value,
            ]
        ):
            self.bc_layers_outputs[op.name] = TensorInfo(results[0], op.name)
        return True

    def _bc_vlm_callback(self, op, results, operands):
        if op.type == "func.func":
            return True
        if len(results) > 0 and isinstance(results[0], (torch.Tensor, np.ndarray, Value)):
            self.bc_vlm_layers_outputs[op.name] = TensorInfo(results[0], op.name)
        return True

    def _run_bc_llm_inference(self, text_input: str):
        self.bc_model.functions[0].register_callback(self._bc_callback)

        if self.args.model_name in QWEN2_5_VL_MODELS:
            # Use Qwen2.5-VL specific input preparation
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = prepare_qwen2_5_vl_inputs(
                text_input=text_input,
                tokenizer=self.tokenizer,
                llm_wrapper=self.torch_llm_model,
                chunk_size=self.args.chunk_size,
                cache_len=self.args.cache_len,
                device=self.device,
                mask_value=mask_value_dict[self.args.model_name],
            )
        elif self.args.model_name == "internvl3_5-1b":
            # Use InternVL3.5-1b specific input preparation
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = prepare_internvl35_inputs(
                text_input=text_input,
                tokenizer=self.tokenizer,
                llm_wrapper=self.torch_llm_model,
                chunk_size=self.args.chunk_size,
                cache_len=self.args.cache_len,
                device=self.device,
                mask_value=mask_value_dict[self.args.model_name],
                pos_mask_value=pos_mask_value_dict[self.args.model_name],
                padding_side=padding_side_dict[self.args.model_name],
            )
        else:
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = self.calib_data_preparer.prepare_inputs(text_input)

        model_args = self.torch_llm_model.get_model_args()

        bc_inputs_meta = [(inp.name, inp.type.shape, inp.type.np_dtype) for inp in self.bc_model.functions[0].inputs]

        for input_ids, attn_mask, position_ids in zip(
            input_chunks, causal_mask_chunks, position_ids_chunks, strict=False
        ):
            feed_dict = self.prepare_llm_chunk_inputs(
                input_ids,
                position_ids,
                attn_mask,
                past_key_value_list,
                bc_inputs_meta,
                self.args.model_name,
                self.torch_llm_model,
            )

            outputs = self.bc_model.functions[0].feed(inputs=feed_dict)

            if self.args.model_name in QWEN2_5_VL_MODELS:
                self.update_kv_cache(
                    past_key_value_list,
                    outputs,
                    chunk_size=(input_ids.shape[1] if input_ids.dim() == 3 else input_ids.shape[-1]),  # noqa: E501
                    num_layers=model_args.num_hidden_layers,
                    convert_from_numpy=True,
                )
            elif self.args.model_name == "internvl3_5-1b":
                self.update_kv_cache(
                    past_key_value_list,
                    outputs,
                    chunk_size=input_ids.shape[-1],
                    num_layers=model_args.num_hidden_layers,
                    convert_from_numpy=True,
                )
            else:
                self.update_kv_cache(
                    past_key_value_list,
                    outputs,
                    chunk_size=input_ids.shape[-1],
                    transpose_cache=self.calib_data_preparer.transpose_cache,
                    num_layers=model_args.num_hidden_layers,
                    convert_from_numpy=True,
                )
        step_idx = self._get_compare_step_index(text_input)
        outputs = self._slice_tensor_dict_at_step(cast_to_tensor_info(outputs), step_idx)
        layers = self._slice_tensor_dict_at_step(cast_to_tensor_info(self.bc_layers_outputs), step_idx)
        return outputs, layers

    def _run_hbm_llm_inference(self, text_input: str):
        if self.args.model_name in QWEN2_5_VL_MODELS:
            # Use Qwen2.5-VL specific input preparation
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = prepare_qwen2_5_vl_inputs(
                text_input=text_input,
                tokenizer=self.tokenizer,
                llm_wrapper=self.torch_llm_model,
                chunk_size=self.args.chunk_size,
                cache_len=self.args.cache_len,
                device=self.device,
                mask_value=mask_value_dict[self.args.model_name],
            )
        elif self.args.model_name == "internvl3_5-1b":
            # Use InternVL3.5-1b specific input preparation
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = prepare_internvl35_inputs(
                text_input=text_input,
                tokenizer=self.tokenizer,
                llm_wrapper=self.torch_llm_model,
                chunk_size=self.args.chunk_size,
                cache_len=self.args.cache_len,
                device=self.device,
                mask_value=mask_value_dict[self.args.model_name],
                pos_mask_value=pos_mask_value_dict[self.args.model_name],
                padding_side=padding_side_dict[self.args.model_name],
            )
        else:
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = self.calib_data_preparer.prepare_inputs(text_input)

        model_args = self.torch_llm_model.get_model_args()

        hbm_inputs_meta = [(inp.name, inp.type.shape, inp.type.np_dtype) for inp in self._hbm_llm_module.inputs]

        for input_ids, attn_mask, position_ids in zip(
            input_chunks, causal_mask_chunks, position_ids_chunks, strict=False
        ):
            model_inputs = self.prepare_llm_chunk_inputs(
                input_ids,
                position_ids,
                attn_mask,
                past_key_value_list,
                hbm_inputs_meta,
                self.args.model_name,
                self.torch_llm_model,
            )

            outputs = self._hbm_llm_module.feed(
                feed_dict=model_inputs,
                remote_ip=self.args.remote_ip,
                username=self.args.username,
                remote_port=self.args.port,
                password=self.args.password,
                remote_work_root=self.args.remote_path,
            )
            if self.args.model_name in QWEN2_5_VL_MODELS:
                self.update_kv_cache(
                    past_key_value_list,
                    outputs,
                    chunk_size=(input_ids.shape[1] if input_ids.dim() == 3 else input_ids.shape[-1]),  # noqa: E501
                    num_layers=model_args.num_hidden_layers,
                    convert_from_numpy=True,
                )
            elif self.args.model_name == "internvl3_5-1b":
                self.update_kv_cache(
                    past_key_value_list,
                    outputs,
                    chunk_size=input_ids.shape[-1],
                    num_layers=model_args.num_hidden_layers,
                    convert_from_numpy=True,
                )
            else:
                self.update_kv_cache(
                    past_key_value_list,
                    outputs,
                    chunk_size=input_ids.shape[-1],
                    transpose_cache=self.calib_data_preparer.transpose_cache,
                    num_layers=model_args.num_hidden_layers,
                    convert_from_numpy=True,
                )

        step_idx = self._get_compare_step_index(text_input)
        llm_output = self._extract_llm_output_tensor(outputs)
        outputs = self._slice_tensor_dict_at_step(cast_to_tensor_info(llm_output), step_idx)
        return outputs, None

    def _run_hbm_vlm_inference(self, image_input: torch.Tensor):
        if self._hbm_vlm_module is None:
            raise RuntimeError("HBM VLM module not loaded. Check configuration.")

        hbm_inputs = getattr(self._hbm_vlm_module, "inputs", None)
        if not hbm_inputs or len(hbm_inputs) == 0:
            raise RuntimeError("HBM VLM module has no declared inputs.")

        first_input = hbm_inputs[0]
        input_name = getattr(first_input, "name", "_input_0")
        input_dtype = getattr(first_input.type, "np_dtype", None)

        # For Qwen2.5-VL, convert image from [N,C,H,W] to [1, seq_len, patch_dim]
        if self.args.model_name in QWEN2_5_VL_MODELS:
            image_input = self.torch_vlm_model.prepare_vision_input(image_input)

        image_np = image_input.cpu().numpy()
        if input_dtype is not None:
            image_np = image_np.astype(input_dtype)

        model_inputs = {input_name: image_np}
        return (
            self._hbm_vlm_module.feed(
                feed_dict=model_inputs,
                remote_ip=self.args.remote_ip,
                username=self.args.username,
                remote_port=self.args.port,
                password=self.args.password,
                remote_work_root=self.args.remote_path,
            ),
            None,
        )

    def _slice_tensor_dict_at_step(self, tensors: dict, step_index: int):
        result = OrderedDict()
        for _, info in tensors.items():
            if info is None or info.data is None:
                continue
            dims_equal_chunk = np.where(np.array(info.data.shape) == self.args.chunk_size)[0]
            if dims_equal_chunk.size == 1:
                dim = int(dims_equal_chunk[0])
                index_slices: list[int | slice] = [slice(None)] * info.data.ndim
                # index_slices[dim] = step_index
                index_slices[dim] = slice(step_index, None)
                sliced = info.data[tuple(index_slices)]
                result[info.name] = TensorInfo(sliced, info.name)
            elif self.is_special_op(info.name):
                src_shape = info.data.shape
                equal_dim = np.where(np.array(src_shape) == self.args.chunk_size * self.key_value_groups)[0]
                if equal_dim.size != 1:
                    continue
                dim = int(equal_dim[0])
                new_shape = list(src_shape)
                new_shape[dim - 1] *= self.key_value_groups
                new_shape[dim] = self.args.chunk_size
                info.data = info.data.reshape(new_shape)
                index_slices: list[int | slice] = [slice(None)] * info.data.ndim
                # index_slices[dim] = step_index
                index_slices[dim] = slice(step_index, None)
                sliced = info.data[tuple(index_slices)]
                result[info.name] = TensorInfo(sliced, info.name)
            elif info.name == "_output_0" or info.name == "output_0":
                # _output_0 for bc output, output_0 for hbm output
                result[info.name] = TensorInfo(info.data, info.name)
        return result

    def check_bc_op_name(self, op_name: str):
        name_match = re.search(r'"([^\"]+)"[^\"]*$', op_name)
        return bool(name_match)

    def is_special_op(self, op_name: str):
        if op_name in self.special_name:
            return True
        name_match = re.search(r'"([^\"]+)"[^\"]*$', op_name)
        if not name_match:
            return False
        return name_match.group(1) in self.special_name
