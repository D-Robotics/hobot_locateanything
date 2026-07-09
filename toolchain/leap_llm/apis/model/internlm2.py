import os

import torch

from leap_llm.apis.calibration.calibration import CalibrationDataPreparer
from leap_llm.apis.calibration.data_loader import load_text_data
from leap_llm.models.internlm2.model import Internlm2, save_model_checkpoint


class Internlm2Api:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        chunk_size: int = 256,
        cache_len: int = 512,
        device: str = "cpu",
        dtype: str = "float32",
        preserve_precision: bool = False,
        model_type: str = "internlm2-1_8b",
        w_bits: int = 8,
    ):
        self.input_model_path = input_model_path
        self.calib_text_data = load_text_data(calib_text_path)
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.device = device
        self.dtype = dtype

        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_path = os.path.join(
            output_model_path,
            f"{model_type}_chunk_{chunk_size}_cache_{cache_len}_q{w_bits}.hbm",
        )

        ckpt_dir = save_model_checkpoint(input_model_path, self.output_model_path)
        self.internlm2_model = Internlm2.build(
            ckpt_dir,
            chunk_size=chunk_size,
            cache_len=cache_len,
            preserve_precision=preserve_precision,
        )

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        device = self.device if torch.cuda.is_available() else "cpu"

        if self.dtype == "float16" and device != "cpu":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.internlm2_model.model.to(device, dtype=dtype)
        self.internlm2_model.model.compile_mode(False)
        self.internlm2_model.model.eval()

        transpose_cache = True
        mask_value = -512
        preparer = CalibrationDataPreparer(
            model_type="internlm",
            model_dir=self.input_model_path,
            seq_len=self.chunk_size,  # 使用 chunk_size 作为 seq_len
            kv_cache_len=self.cache_len,
            transpose_cache=transpose_cache,
            device=device,
            mask_value=mask_value,
            pos_mask_value=0,
            data_type=dtype,
            padding_side="right",
        )

        for prompt in self.calib_text_data:
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                pask_key_value_list,
            ) = preparer.prepare_inputs(prompt)

            for i, (input_ids, attn_mask, position_ids) in enumerate(
                zip(input_chunks, causal_mask_chunks, position_ids_chunks)
            ):
                with torch.no_grad():
                    outputs = self.internlm2_model.model.forward(
                        input_ids.to(device),
                        position_ids.to(device),
                        attn_mask.to(device),
                        pask_key_value_list,
                    )

                for z in range(
                    0, self.internlm2_model.model_args.num_hidden_layers * 2
                ):
                    new_cache = outputs[z + 1]
                    past = pask_key_value_list[z]

                    if transpose_cache:
                        slice_past = past[self.chunk_size :, :, :]
                    else:
                        slice_past = past[:, self.chunk_size :, :]

                    dim = 0 if transpose_cache else -2
                    update_cache = torch.concat([slice_past, new_cache], dim=dim)
                    pask_key_value_list[z] = update_cache

        self.internlm2_model.model.compile_mode(True)
        self.internlm2_model.model.to("cpu")

        self.internlm2_model.compile(
            stage="all",
            output_model_path=self.output_model_path,
            **llm_kwargs,
        )
