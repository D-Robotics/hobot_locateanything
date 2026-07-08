import os

import torch

from leap_llm.apis.calibration.calibration import (
    CalibrationDataPreparer,
    update_causal_mask,
)
from leap_llm.apis.calibration.data_loader import (
    load_image_data,
    load_text_data,
)
from leap_llm.models.internvl3_5.model import InterVL3_5
from leap_llm.nn.utils import (
    standard_lm_name,
    standard_token_embeddings_name,
    standard_vit_name,
)


class InternVL3_5Api:
    """API for InternVL3.5 multimodal model compilation and calibration."""

    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        calib_image_path: str = None,
        chunk_size: int = 256,
        cache_len: int = 2048,
        device: str = "cpu",
        dtype: str = "float32",
        w_bits: int = 8,
        model_type: str = "internvl3_5-1b",
        vit_core_num: list[int] = [1],
        prefill_core_num: list[int] = [1],
        decode_core_num: list[int] = [1],
        march: str = "nash-p",
    ):
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.device = device
        self.dtype = dtype
        self.w_bits = w_bits
        self.model_type = model_type

        self.calib_text_data = load_text_data(calib_text_path)
        self.calib_image_data = load_image_data(calib_image_path)
        self.vit_core_num = vit_core_num
        self.prefill_core_num = prefill_core_num
        self.decode_core_num = decode_core_num
        os.makedirs(output_model_path, exist_ok=True)

        self.vision_hbm_path = standard_vit_name(
            input_model_path, output_model_path, march, vit_core_num
        )

        self.text_hbm_path = standard_lm_name(
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

        self.internvl3_5 = InterVL3_5.build(
            input_model_path, chunk_size=chunk_size, cache_len=cache_len
        )

        self.vision_model = self.internvl3_5.get_vit_model()
        self.text_model = self.internvl3_5.get_language_model()
        self.embed_tokens = self.text_model.get_input_embeddings()

        self.mask_value = -512.0
        self.pos_mask_value = 1

    def calib_compile_visual(self, dtype, device, **kwargs):
        vision = self.vision_model
        vision.compile_mode(False)
        vision.to(device, dtype=dtype)
        vision.eval()

        for image_pixel in self.calib_image_data:
            with torch.no_grad():
                vision.forward(image_pixel.to(device))

        vision.compile_mode(True)
        vision.to("cpu", dtype=torch.float16)
        self.internvl3_5.compile(
            stage="vit",
            output_model_path=self.vision_hbm_path,
            enable_vpu=True,
            vit_core_num=self.vit_core_num,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            **kwargs,
        )

    def calib_compile_text(self, dtype, device, **kwargs):
        calib_preparer = CalibrationDataPreparer(
            model_type=self.model_type,
            model_dir=self.input_model_path,
            seq_len=self.chunk_size,
            kv_cache_len=self.cache_len,
            device=device,
            transpose_cache=False,
            mask_value=self.mask_value,
            pos_mask_value=self.pos_mask_value,
        )

        self._save_embed_tokens()
        text_model = self.text_model
        text_model.compile_mode(False)
        text_model.to(device, dtype=dtype)
        text_model.eval()

        config = text_model.config
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim

        init_kv_shape = [1, self.cache_len, num_key_value_heads, head_dim]
        init_kv_data = torch.zeros(init_kv_shape, dtype=torch.float32).to(device)
        past_key_values_list = [init_kv_data] * num_hidden_layers + [
            init_kv_data
        ] * num_hidden_layers

        for prompt in self.calib_text_data:
            (
                input_chunks,
                _,
                position_ids_chunks,
                _,
            ) = calib_preparer.prepare_inputs(prompt)

            valid_token_count = sum(chunk.shape[-1] for chunk in input_chunks)

            attention_mask_1d = torch.zeros(
                (1, self.cache_len), dtype=torch.int32, device=device
            )
            attention_mask_1d[0, -valid_token_count:] = 1

            cache_position = torch.arange(
                0, self.cache_len, dtype=torch.long, device=device
            )

            dummy_input_ids = torch.zeros(
                (1, valid_token_count), dtype=torch.long, device=device
            )

            causal_mask_4d = update_causal_mask(
                attention_mask_1d,
                dummy_input_ids,
                cache_position,
                min_dtype=self.mask_value,
                sequence_length=self.cache_len,
                kv_cache_len=self.cache_len,
                dtype=torch.float32,
                device=device,
                padding_side="left",
            )

            inputs_pad_len = sum(chunk.shape[-1] for chunk in input_chunks)
            causal_mask_4d = causal_mask_4d[:, :, -inputs_pad_len:, :]
            causal_mask_chunks = causal_mask_4d.split(self.chunk_size, dim=2)

            for input_ids_chunk, position_ids_chunk, mask_chunk in zip(
                input_chunks, position_ids_chunks, causal_mask_chunks
            ):
                input_ids_chunk = input_ids_chunk.to(device)
                position_ids_chunk = position_ids_chunk.to(device)

                with torch.no_grad():
                    inputs_embeds = self.embed_tokens(input_ids_chunk)

                with torch.no_grad():
                    logits, new_keys, new_values = text_model.forward(
                        input_embeds=inputs_embeds.to(dtype),
                        position_ids=position_ids_chunk,
                        attention_mask=mask_chunk.to(dtype),
                        cache_keys=past_key_values_list[:num_hidden_layers],
                        cache_values=past_key_values_list[num_hidden_layers:],
                    )

                for idx in range(num_hidden_layers):
                    past_keys = past_key_values_list[idx]
                    new_key = new_keys[idx]
                    slice_past = past_keys[:, self.chunk_size :, :, :]
                    past_key_values_list[idx] = torch.cat([slice_past, new_key], dim=1)

                    past_values = past_key_values_list[num_hidden_layers + idx]
                    new_value = new_values[idx]
                    slice_past = past_values[:, self.chunk_size :, :, :]
                    past_key_values_list[num_hidden_layers + idx] = torch.cat(
                        [slice_past, new_value], dim=1
                    )

        text_model.compile_mode(True)
        text_model.to("cpu", dtype=torch.float16)
        self.internvl3_5.compile(
            stage="llm",
            output_model_path=self.text_hbm_path,
            enable_vpu=True,
            vit_core_num=self.vit_core_num,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            **kwargs,
        )

    def _save_embed_tokens(self):
        embed_tokens_path = self.token_embeddings_file_name
        if not os.path.exists(embed_tokens_path):
            with torch.no_grad():
                weights = self.embed_tokens.weight.detach().cpu().numpy()
                weights.tofile(embed_tokens_path)

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        device = self.device if torch.cuda.is_available() else "cpu"

        if self.dtype == "float16" and device != "cpu":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.calib_compile_visual(dtype=dtype, device=device, **vit_kwargs)

        self.calib_compile_text(dtype=dtype, device=device, **llm_kwargs)

    def get_hbm_path(self) -> tuple[str, str]:
        return self.text_hbm_path, self.vision_hbm_path
