import os
from pathlib import Path

import torch
from hbdk4.compiler import leap
from transformers import AutoModelForCausalLM

from leap_llm.apis.calibration.calibration import CalibrationDataPreparer
from leap_llm.apis.calibration.data_loader import load_image_data, load_text_data
from leap_llm.models.internvl_1b.model import Internlm1b, Internvl1bVision
from leap_llm.nn.utils import (
    standard_lm_name,
    standard_token_embeddings_name,
    standard_vit_name,
    timeit,
)


class Internvl1bApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        calib_image_path: str = None,
        chunk_size: int = 256,
        cache_len: int = 512,
        device: str = "cpu",
        vlm_model_type: str = "internvl",
        dtype: str = "float32",
        w_bits: int = 8,
        weight_scales_file: str = None,
        vit_core_num: list[int] = [1],
        prefill_core_num: list[int] = [1],
        decode_core_num: list[int] = [1],
        kept_tokens_file: str = None,
        march: str = "nash-p",
    ):
        self.input_model_path = input_model_path
        self.calib_text_data = load_text_data(calib_text_path)
        self.calib_image_data = load_image_data(calib_image_path)
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.device = device
        self.dtype = dtype

        self.vit_file_name = standard_vit_name(
            input_model_path, output_model_path, march, vit_core_num
        )

        self.lm_file_name = standard_lm_name(
            input_model_path,
            output_model_path,
            chunk_size,
            cache_len,
            w_bits,
            march,
            prefill_core_num,
            decode_core_num,
        )

        self.token_embeddings_file_name = standard_token_embeddings_name(
            input_model_path, output_model_path
        )

        self.vit_core_num = vit_core_num
        self.prefill_core_num = prefill_core_num
        self.decode_core_num = decode_core_num

        model = AutoModelForCausalLM.from_pretrained(
            input_model_path, trust_remote_code=True
        )
        checkpoint = model.state_dict()
        self.vit_model = Internvl1bVision.load_model(input_model_path, checkpoint)
        self.lm_model = Internlm1b.load_model(
            input_model_path,
            checkpoint=checkpoint,
            chunk_size=chunk_size,
            cache_len=cache_len,
            w_bits=w_bits,
            weight_scales_file=weight_scales_file,
            kept_tokens_file=kept_tokens_file,
        )
        # save token embeddings to file
        os.makedirs(output_model_path, exist_ok=True)
        tok_embs = checkpoint["language_model.model.embed_tokens.weight"]
        tok_embs.numpy().tofile(self.token_embeddings_file_name)

    @timeit
    def calib_compile_vit(self, dtype, device, **kwargs):
        vit_module = self.vit_model
        vit_module.set_model_device(device, dtype=dtype)
        vit_module.set_compile_mode(False)
        for image_pixel in self.calib_image_data:
            vit_module.forward(image_pixel.to(device))
        vit_module.set_model_device("cpu", dtype=torch.float16)
        vit_module.set_compile_mode(True)
        vit_module.compile(
            dtype=leap.float16,
            output_model_path=self.vit_file_name,
            vit_core_num=self.vit_core_num,
            **kwargs,
        )

    @timeit
    def calib_compile_lm(self, dtype, device, transpose_cache=True, **kwargs):
        mask_value = -8192
        calib_data = CalibrationDataPreparer(
            model_type="internvl",
            model_dir=self.input_model_path,
            seq_len=self.chunk_size,
            kv_cache_len=self.cache_len,
            device=device,
            transpose_cache=transpose_cache,
            mask_value=mask_value,
            pos_mask_value=0,
            data_type=torch.float16,
            padding_side="left",
        )
        module = self.lm_model
        # calibrate
        module.set_model_device(device, dtype=dtype)
        module.set_compile_mode(False)
        seq_len = module.model_args.prefill_seq_len

        for prompt in self.calib_text_data:
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                pask_key_value_list,
            ) = calib_data.prepare_inputs(prompt)

            # Prefill stage loop
            for i, (input_ids, attn_mask, position_ids) in enumerate(
                zip(input_chunks, causal_mask_chunks, position_ids_chunks)
            ):
                with torch.no_grad():
                    outputs = module.forward(
                        input_ids, position_ids, attn_mask, pask_key_value_list
                    )
                for z in range(0, module.model_args.num_hidden_layers * 2):
                    new_cache = outputs[z + 1]
                    past = pask_key_value_list[z]
                    slice_past = (
                        past[seq_len:, :, :]
                        if transpose_cache
                        else past[:, seq_len:, :]
                    )
                    dim = 0 if transpose_cache else -2
                    update_cache = torch.concat([slice_past, new_cache], dim=dim)
                    pask_key_value_list[z] = update_cache

        # compile
        module.set_model_device("cpu", dtype=torch.float16)
        module.set_compile_mode(True)
        module.compile(
            stage="all",
            dtype=leap.float16,
            output_model_path=self.lm_file_name,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            **kwargs,
        )

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        device = self.device if torch.cuda.is_available() else "cpu"

        if self.dtype == "float16" and device != "cpu":
            dtype = torch.float16
        else:
            dtype = torch.float32
        # compile vision model
        self.calib_compile_vit(
            dtype=dtype,
            device=device,
            **vit_kwargs,
        )
        # compile lm model
        self.calib_compile_lm(
            dtype=dtype,
            device=device,
            transpose_cache=True,
            **llm_kwargs,
        )

    def get_quant_path(self) -> tuple[str, str]:
        """Return fixed InternVL-1B BC paths."""

        llm_bc = str(Path(self.lm_file_name).with_suffix(".prefill_convert_rm.bc"))
        vlm_bc = str(Path(self.vit_file_name).with_suffix(".convert.bc"))
        return llm_bc, vlm_bc

    def get_hbm_path(self) -> tuple[str, str]:
        """Return fixed InternVL-1B HBM path."""

        return self.lm_file_name, self.vit_file_name
