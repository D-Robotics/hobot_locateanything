import os
from pathlib import Path

import torch

from leap_llm.apis.calibration.calibration import CalibrationDataPreparer
from leap_llm.apis.calibration.data_loader import load_text_data
from leap_llm.models.deepseek.model import DeepSeek


class DeepSeekApi:
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
        model_type: str = "deepseek-qwen-1_5b",
        w_bits: int = 8,
        mask_value: int = -512,
        march: str = "nash-e",
        vit_core_num: list[int] = None,
        prefill_core_num: list[int] = None,
        decode_core_num: list[int] = None,
        input_model_format: str = "hf",
    ):
        self.input_model_path = input_model_path
        self.calib_text_data = load_text_data(calib_text_path)
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.device = device
        self.dtype = dtype
        self.w_bits = w_bits
        self.mask_value = mask_value
        self.model_type = model_type
        self.prefill_core_num = prefill_core_num[0]
        self.decode_core_num = decode_core_num[0]
        self.march = march

        if "7b" in self.model_type:
            self.prefix = "DeepSeek-R1-Distill-Qwen-7B_language"
        else:
            self.prefix = "DeepSeek-R1-Distill-Qwen-1.5B_language"

        if self.march == "nash-p":
            self.mask_value = -32768.0
        self.prefill_core_num = prefill_core_num[0]
        self.decode_core_num = decode_core_num[0]
        self.march = march

        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_path = os.path.join(
            output_model_path,
            f"{self.prefix}_chunk_{chunk_size}_cache_{cache_len}_w{w_bits}"
            f"_{self.march}_corenum_{prefill_core_num[0]}_{decode_core_num[0]}.hbm",
        )

        print(f"hbm filepath: {self.output_model_path}")

        self.deepseek_model = DeepSeek.build(
            input_model_path,
            chunk_size=chunk_size,
            cache_len=cache_len,
            preserve_precision=preserve_precision,
            w_bits=w_bits,
            march=self.march,
        )

    def compile(self, skip_final_compile=False, vit_kwargs=None, llm_kwargs=None):
        device = self.device if torch.cuda.is_available() else "cpu"

        dtype = torch.float16 if "7b" in self.model_type else torch.float32

        self.deepseek_model.model.to(device, dtype=dtype)
        self.deepseek_model.model.compile_mode(False)

        transpose_cache = True
        preparer = CalibrationDataPreparer(
            model_type="deepseek",
            model_dir=self.input_model_path,
            seq_len=self.chunk_size,
            kv_cache_len=self.cache_len,
            transpose_cache=transpose_cache,
            device=device,
            mask_value=self.mask_value,
            pos_mask_value=1,
            data_type=dtype,
            padding_side="left",
        )
        # set the padding_side to left on tokenizer
        preparer.tokenizer.padding_side = "left"

        for prompt in self.calib_text_data:
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_value_list,
            ) = preparer.prepare_inputs(prompt)

            # nash-p creates the batch dim=0
            if "nash-p" in self.march:
                past_key_value_list = [c.unsqueeze(0) for c in past_key_value_list]

            for _, (input_ids, attn_mask, position_ids) in enumerate(
                zip(input_chunks, causal_mask_chunks, position_ids_chunks)
            ):
                with torch.no_grad():
                    if "nash-p" in self.march:
                        position_ids = position_ids.unsqueeze(0)
                        attn_mask = attn_mask.unsqueeze(0)
                    outputs = self.deepseek_model.model.forward(
                        input_ids.to(device),
                        position_ids.to(device),
                        attn_mask.to(device),
                        past_key_value_list,
                    )

                for z in range(0, self.deepseek_model.model_args.num_hidden_layers * 2):
                    new_cache = outputs[z + 1]
                    past = past_key_value_list[z]
                    if "nash-p" in self.march:
                        slice_past = past[:, self.chunk_size :]
                        past_key_value_list[z] = torch.concat([slice_past, new_cache], dim=1)
                    else:
                        slice_past = past[self.chunk_size :, :, :] if transpose_cache else past[:, self.chunk_size :, :]

                        dim = 0 if transpose_cache else -2
                        update_cache = torch.concat([slice_past, new_cache], dim=dim)
                        past_key_value_list[z] = update_cache

        print("data calibrated")

        self.deepseek_model.model.compile_mode(True)
        self.deepseek_model.model.to(device="cpu", dtype=torch.float16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not skip_final_compile:
            self.deepseek_model.compile(
                stage="all",
                output_model_path=self.output_model_path,
                prefill_core_num=self.prefill_core_num,
                decode_core_num=self.decode_core_num,
                **llm_kwargs,
            )

    def get_quant_path(self) -> tuple[str, None]:
        """Return fixed DeepSeek BC path."""

        return str(Path(self.output_model_path).with_suffix(".prefill_convert_removed.bc")), None

    def get_hbm_path(self) -> tuple[str, None]:
        """Return fixed DeepSeek HBM path."""

        return self.output_model_path, None
