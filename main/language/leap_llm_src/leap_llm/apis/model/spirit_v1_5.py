import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer

from leap_llm.models.Spirit_v1_5.model_dit import SpiritDitModel
from leap_llm.models.Spirit_v1_5.spirit_utils import get_rope_index_3, preprocess_qwen_visual
from leap_llm.models.Spirit_v1_5.model_vlm import SpiritVisionModel, SpiritLLMModel


def _image_to_spirit_patch_tokens(image_bgr: np.ndarray) -> np.ndarray:
    # Spirit vision patch embed expects per-token dim: 2 * 16 * 16 * 3 = 1536.
    patch = 16
    target_h, target_w = 256, 320  # fixed grid: 16 x 20
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rgb = rgb / 255.0
    chw = np.transpose(rgb, (2, 0, 1))  # [3, H, W]
    frames = np.stack([chw, chw], axis=0)  # [2, 3, H, W], emulate temporal_patch_size=2

    # [2,3,256,320] -> [16,20,2,16,16,3] -> [320,1536]
    frames = frames.reshape(2, 3, target_h // patch, patch, target_w // patch, patch)
    frames = np.transpose(frames, (2, 4, 0, 3, 5, 1))
    tokens = frames.reshape((target_h // patch) * (target_w // patch), -1)
    return tokens


def load_calib_data_spirit_v1_5(
    calib_image_path, calib_text_path, device="cuda", dtype=torch.float16
):
    # 读取 prompt json
    with open(calib_text_path, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    prompts = [item["text"] for item in text_data]
    N_prompt = len(prompts)

    # 读取图像主目录下以数字命名的文件夹
    folder_names = sorted(
        [name for name in os.listdir(calib_image_path) if name.isdigit()],
        key=lambda x: int(x),
    )
    N_folder = len(folder_names)

    # 数量校验
    if N_folder != N_prompt:
        raise ValueError(
            f"图像文件夹数量({N_folder}) 与 "
            f"prompt 数量({N_prompt}) 不一致，请保持两者相同！"
        )

    results = []

    for idx, folder_name in enumerate(folder_names):
        folder = os.path.join(calib_image_path, folder_name)

        img_files = glob.glob(os.path.join(folder, "image_*.jpg"))
        if len(img_files) != 3:
            raise ValueError(
                f"文件夹 {folder} 中图片数量不为3（当前为 {len(img_files)}），请检查！"
            )
        images = []
        for i in range(3):  # image_0, image_1, image_2
            img_path = os.path.join(folder, f"image_{i}.jpg")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"缺少文件: {img_path}")

            inp = cv2.imread(img_path)
            if inp is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            inp = _image_to_spirit_patch_tokens(inp)
            inp = np.expand_dims(inp, axis=0)  # [1, 320, 1536]
            tensor = torch.from_numpy(inp).to(device).to(dtype=dtype)

            images.append(tensor)

        prompt = prompts[idx]

        results.append((images, prompt))

    return results


def pack_visual_embeds_by_mask(
    deepstack_visual_embeds_torch: list[torch.Tensor],
    visual_pos_masks_torch: torch.Tensor,
) -> list[torch.Tensor]:
    visual_pos_masks_torch = visual_pos_masks_torch.bool()
    batch_size, seq_len = visual_pos_masks_torch.shape
    true_count = int(visual_pos_masks_torch.sum().item())

    packed_visual_embeds: list[torch.Tensor] = []
    for layer_idx, visual_embeds in enumerate(deepstack_visual_embeds_torch):
        hidden_size = visual_embeds.shape[-1]
        dense = torch.zeros(
            (batch_size, seq_len, hidden_size),
            device=visual_embeds.device,
            dtype=visual_embeds.dtype,
        )

        if visual_embeds.ndim == 2:
            if visual_embeds.shape[0] != true_count:
                raise ValueError(
                    f"Layer {layer_idx}: visual_embeds token count {visual_embeds.shape[0]} "
                    f"!= mask true count {true_count}"
                )
            dense[visual_pos_masks_torch, :] = visual_embeds
        elif visual_embeds.ndim == 3:
            if visual_embeds.shape[0] != batch_size:
                raise ValueError(
                    f"Layer {layer_idx}: batch mismatch, embeds batch {visual_embeds.shape[0]} "
                    f"!= mask batch {batch_size}"
                )
            for b in range(batch_size):
                b_mask = visual_pos_masks_torch[b]
                b_true_count = int(b_mask.sum().item())
                if visual_embeds.shape[1] != b_true_count:
                    raise ValueError(
                        f"Layer {layer_idx}, batch {b}: visual_embeds token count {visual_embeds.shape[1]} "
                        f"!= mask true count {b_true_count}"
                    )
                dense[b, b_mask, :] = visual_embeds[b]
        else:
            raise ValueError(
                f"Layer {layer_idx}: unsupported visual_embeds ndim={visual_embeds.ndim}, expected 2 or 3"
            )

        packed_visual_embeds.append(dense.to(torch.float16))

    return packed_visual_embeds


def left_pad_last_dim(x: torch.Tensor, target_len: int, pad_value: float | int) -> torch.Tensor:
    cur_len = x.shape[-1]
    if cur_len > target_len:
        raise ValueError(f"Current length {cur_len} is larger than target_len {target_len}")
    if cur_len == target_len:
        return x
    pad_len = target_len - cur_len
    pad_shape = list(x.shape)
    pad_shape[-1] = pad_len
    pad_tensor = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([pad_tensor, x], dim=-1)


def left_pad_input_embeds_with_eot(
    input_embeds_torch: torch.Tensor,
    eot_embed: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    bsz, cur_len, hidden = input_embeds_torch.shape
    if cur_len > target_len:
        raise ValueError(f"Current length {cur_len} is larger than target_len {target_len}")
    if cur_len == target_len:
        return input_embeds_torch
    pad_len = target_len - cur_len
    eot_embed = eot_embed.to(dtype=input_embeds_torch.dtype, device=input_embeds_torch.device)
    pad = eot_embed.view(1, 1, hidden).expand(bsz, pad_len, hidden).contiguous()
    return torch.cat([pad, input_embeds_torch], dim=1)


def left_pad_attention_mask_to_square(
    attention_mask_torch: torch.Tensor,
    target_len: int,
    pad_value: float,
) -> torch.Tensor:
    q_len = attention_mask_torch.shape[-2]
    k_len = attention_mask_torch.shape[-1]
    if q_len != k_len:
        raise ValueError(f"Expect square attention mask, got [{q_len}, {k_len}]")
    if q_len > target_len:
        raise ValueError(f"Current length {q_len} is larger than target_len {target_len}")
    if q_len == target_len:
        return attention_mask_torch
    pad_len = target_len - q_len
    left_k = torch.full(
        (*attention_mask_torch.shape[:-1], pad_len),
        pad_value,
        dtype=attention_mask_torch.dtype,
        device=attention_mask_torch.device,
    )
    mask = torch.cat([left_k, attention_mask_torch], dim=-1)
    left_q = torch.full(
        (*mask.shape[:-2], pad_len, mask.shape[-1]),
        pad_value,
        dtype=mask.dtype,
        device=mask.device,
    )
    return torch.cat([left_q, mask], dim=-2)


class SpiritV1_5Api:
    def __init__(
        self,
        input_model_path: str,
        config_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        calib_image_path: str = None,
        calib_action_data_path: str = None,
        device: str = "cpu",
        model_type: str = "spirit_v1_5",
        dtype: str = "float16",
        chunk_size=320,
        w_bits: int = 8,
    ):
        self.input_model_path = input_model_path
        self.device = device
        self.dtype = dtype
        self.model_type = model_type
        self.config_path = config_path
        self.output_vision_model_path = os.path.join(
            output_model_path,
            f"{self.model_type}_vision_ptq.hbm",
        )
        self.output_llm_model_path = os.path.join(
            output_model_path,
            f"{self.model_type}_llm_ptq.hbm",  # noqa: E501
        )
        self.output_dit_model_path = os.path.join(
            output_model_path,
            f"{self.model_type}_dit_ptq.hbm",  # noqa: E501
        )
        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_dir = output_model_path
        self.chunk_size = chunk_size
        self.neg_mask_value = -32767.0
        self.image_token_id = 151655

        # self.calib_text_data = load_text_data(calib_text_path)
        self.calib_data = load_calib_data_spirit_v1_5(
            calib_image_path, calib_text_path, device=device
        )
        self.calib_action_data_path = calib_action_data_path

        self.model_vision = SpiritVisionModel.build(
            self.config_path, f"{self.input_model_path}/model.safetensors"
        )
        self.model_llm = SpiritLLMModel.build(
            self.config_path, f"{self.input_model_path}/model.safetensors"
        )
        model_dir = Path(f"{self.input_model_path}")
        self.model_dit = SpiritDitModel.build(
            model_dir / "config.json", model_dir / "model.safetensors"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.config_path),
            add_eos_token=False,
            trust_remote_code=True,
            use_fast=False,
        )
        print("Load model success!")

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        device = (
            self.device
            if torch.cuda.is_available() and self.device.startswith("cuda")
            else "cpu"
        )
        dtype = torch.float16
        self.model_vision.model.to(device=device, dtype=dtype)
        self.model_llm.model.to(device=device, dtype=dtype)
        self.model_dit.model.to(device=device, dtype=dtype)
        self.model_vision.model.compile_mode(False)
        self.model_llm.model.compile_mode(False)
        self.model_dit.model.compile_mode(False)

        compile_vit_kwargs = vit_kwargs or {}
        compile_llm_kwargs = llm_kwargs or {}
        compile_kwargs = {}
        compile_kwargs.update(compile_vit_kwargs)
        compile_kwargs.update(compile_llm_kwargs)
        # Save embedding weights for engine consumption before to fp16
        self._calibrate_forward(device=device, dtype=dtype, **compile_kwargs)

        self.model_vision.model.compile_mode(True)
        self.model_llm.model.compile_mode(True)
        self.model_dit.model.compile_mode(True)
        self.model_vision.model.to(device="cpu", dtype=dtype)
        self.model_llm.model.to(device="cpu", dtype=dtype)
        self.model_dit.model.to(device="cpu", dtype=dtype)

        self.model_vision.compile(
            output_model_path=self.output_vision_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )
        self.model_llm.compile(
            output_model_path=self.output_llm_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )
        self.model_dit.compile(
            output_model_path=self.output_dit_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )

    def _tokenize(self, prompt):
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self.tokenizer.encode(cleaned_text, add_special_tokens=True)
        tokens = tokens + self.tokenizer.encode("\n", add_special_tokens=False)
        valid_len = len(tokens)
        return np.asarray(tokens, dtype=np.int32), valid_len

    def _calibrate_forward(self, *, device: str, dtype, **kwargs):
        calib_index = 0
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if eot_token_id is None or eot_token_id < 0:
            raise ValueError("Failed to resolve <|endoftext|> token id.")
        eot_embed = self.model_llm.model.get_input_embeddings().weight[eot_token_id]

        for image_data, prompt in self.calib_data:
            _, valid_token_len = self._tokenize(prompt)

            # Vision model expects 3-camera patches concatenated as one sequence.
            vision_input = torch.cat(image_data, dim=1)
            vision_hs, deepstack_i = self.model_vision.model.forward(vision_input)
            vision_hs = vision_hs.to(dtype=dtype)
            deepstack_visual_embeds = [layer_embed.squeeze(0).to(dtype=dtype) for layer_embed in deepstack_i]

            num_images = len(image_data)
            # image_grid_thw is fixed to [1,16,20], merge_size=2 -> 80 image pads per image.
            grid_thw_merged = [80] * num_images
            image_grid_thw = torch.tensor(
                [[1, 16, 20] for _ in range(num_images)],
                dtype=torch.int64,
                device=device,
            )
            expected_vision_tokens = num_images * 80
            if vision_hs.shape[1] != expected_vision_tokens:
                raise ValueError(
                    f"Vision token count mismatch: got {vision_hs.shape[1]}, "
                    f"expected {expected_vision_tokens} ({num_images} * 80)"
                )

            image_placeholders = " ".join(["<image>"] * num_images)
            user_prompt = f"{image_placeholders}\nThe current robot type is Franka. What is the current task?"
            input_id_dict = preprocess_qwen_visual(
                [
                    [
                        {"from": "human", "value": user_prompt},
                        {"from": "gpt", "value": prompt},
                    ]
                ],
                self.tokenizer,
                grid_thw_image=grid_thw_merged,
            )
            input_ids_torch = input_id_dict["input_ids"].to(device)
            attention_mask_1d = input_ids_torch.ne(self.tokenizer.pad_token_id).to(device=device)
            position_ids_torch, _ = get_rope_index_3(
                2,
                input_ids_torch,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask_1d,
            )

            input_embeds_torch = self.model_llm.model.get_input_embeddings()(input_ids_torch).to(dtype=dtype)
            visual_pos_masks_torch = input_ids_torch.eq(self.image_token_id)
            visual_token_count = int(visual_pos_masks_torch.sum().item())
            if visual_token_count != vision_hs.shape[1]:
                raise ValueError(
                    f"Visual placeholder token count {visual_token_count} != vision token count {vision_hs.shape[1]}"
                )
            input_embeds_torch[visual_pos_masks_torch] = vision_hs.reshape(-1, vision_hs.shape[-1])

            seq_len = input_embeds_torch.shape[1]
            if seq_len > self.chunk_size:
                # Keep all visual placeholders and truncate only text prefix.
                vision_token_end = expected_vision_tokens
                if vision_token_end >= self.chunk_size:
                    raise ValueError(
                        f"Visual tokens ({vision_token_end}) exceed/equal chunk_size ({self.chunk_size}), "
                        "cannot preserve full visual region in calibration."
                    )
                start = min(seq_len - self.chunk_size, vision_token_end)
                input_embeds_torch = input_embeds_torch[:, start:, :]
                position_ids_torch = position_ids_torch[:, :, start:]
                visual_pos_masks_torch = visual_pos_masks_torch[:, start:]
                attention_mask_1d = attention_mask_1d[:, start:]
                seq_len = self.chunk_size

            packed_visual_embeds_torch = pack_visual_embeds_by_mask(
                deepstack_visual_embeds, visual_pos_masks_torch
            )
            attention_mask_torch = torch.triu(
                torch.full((1, 1, seq_len, seq_len), self.neg_mask_value, dtype=dtype, device=device),
                diagonal=1,
            )

            input_embeds_torch = left_pad_input_embeds_with_eot(input_embeds_torch, eot_embed, self.chunk_size)
            position_ids_torch = left_pad_last_dim(position_ids_torch, self.chunk_size, 1)
            attention_mask_torch = left_pad_attention_mask_to_square(
                attention_mask_torch, self.chunk_size, self.neg_mask_value
            )
            packed_visual_embeds_torch = [
                left_pad_last_dim(x.transpose(1, 2), self.chunk_size, 0).transpose(1, 2).contiguous()
                for x in packed_visual_embeds_torch
            ]

            vlm_last_embed = self.model_llm.model.forward(
                input_embeds_torch,
                position_ids_torch,
                attention_mask_torch,
                packed_visual_embeds_torch,
            )

            valid_total_len = min(self.chunk_size, seq_len)
            pad_len = self.chunk_size - valid_total_len
            encoder_attention_mask = torch.zeros((1, 1, 61, self.chunk_size), dtype=dtype, device=device)
            if pad_len > 0:
                encoder_attention_mask[:, :, :, :pad_len] = self.neg_mask_value

            action_state = np.load(
                f"{self.calib_action_data_path}/{calib_index}/state.npy"
            )
            action_x_t = np.load(f"{self.calib_action_data_path}/{calib_index}/x_t.npy")
            action_state = torch.from_numpy(action_state).to(device=device, dtype=dtype)
            if action_state.ndim == 2:
                action_state = action_state.unsqueeze(0)
            action_state[:, :, [2, 9]] = 0
            action_x_t = torch.from_numpy(action_x_t).to(device=device, dtype=dtype)

            for step in range(10):
                action_denoise_step = torch.tensor([step], dtype=torch.int32, device=device)
                action_x_t = self.model_dit.model.forward(
                    state=action_state,
                    x_t=action_x_t,
                    timestep=action_denoise_step,
                    vlm_last_embed=vlm_last_embed,
                    encoder_attention_mask=encoder_attention_mask,
                )

            calib_index += 1
