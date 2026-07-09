"""SmolVLA quantization API (vision + VLM prefix + action expert)."""

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from leap_llm.models.smolvla.model_action_expert import SmolVLMActionExpert
from leap_llm.models.smolvla.model_vision import SmolVLMVision
from leap_llm.models.smolvla.model_vlm import SmolVLMPrefix
from leap_llm.models.smolvla.smolvla_utils import (
    build_action_expert_mask,
    build_prefix_pad_att_masks,
    build_vlm_prefix_mask,
    generate_denoise_position_ids,
    generate_prefix_position_ids,
    load_policy_config,
    resize_with_pad,
)


def load_calib_data_smolvla(
    calib_image_path,
    calib_text_path,
    policy_cfg,
    device="cuda",
    dtype=torch.float16,
):
    with open(calib_text_path, encoding="utf-8") as f:
        text_data = json.load(f)
    prompts = [item["text"] for item in text_data]

    folder_names = sorted(
        [name for name in os.listdir(calib_image_path) if name.isdigit()],
        key=lambda x: int(x),
    )
    if len(folder_names) != len(prompts):
        raise ValueError(
            f"图像文件夹数量({len(folder_names)}) 与 "
            f"prompt 数量({len(prompts)}) 不一致"
        )

    h, w = policy_cfg.image_height, policy_cfg.image_width
    results = []
    for folder_name in folder_names:
        folder = os.path.join(calib_image_path, folder_name)
        img_files = sorted(glob.glob(os.path.join(folder, "image_*.jpg")))
        if len(img_files) < 1:
            raise ValueError(f"文件夹 {folder} 中未找到 image_*.jpg")

        images = []
        for img_path in img_files[: policy_cfg.num_images]:
            try:
                import cv2
            except ImportError as e:
                raise ImportError("opencv-python required for SmolVLA calib images") from e
            inp = cv2.imread(img_path)
            if inp is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            inp = inp.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))
            inp = np.expand_dims(inp, axis=0)
            tensor = torch.from_numpy(inp).to(device).to(dtype=torch.float32)
            tensor = resize_with_pad(tensor, w, h, pad_value=0.0)
            tensor = tensor * 2.0 - 1.0
            tensor = tensor.to(dtype=dtype)
            images.append(tensor)

        results.append((images, prompts[int(folder_name)]))
    return results


class SmolVLAApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        calib_image_path: str = None,
        calib_action_data_path: str = None,
        policy_config_path: str = None,
        device: str = "cpu",
        model_type: str = "smolvla",
        dtype: str = "float16",
        vision_tokens_num: int | None = None,
        num_vlm_layers: int | None = None,
        vpu_align_prefix: bool = True,
    ):
        self.input_model_path = input_model_path
        self.device = device
        self.dtype = dtype
        self.model_type = model_type
        self.policy_cfg = load_policy_config(input_model_path, policy_config_path)
        if vision_tokens_num is not None:
            self.policy_cfg.vision_tokens_num = vision_tokens_num
        if num_vlm_layers is not None:
            self.policy_cfg.num_vlm_layers = num_vlm_layers

        # B30 VPU RMSNorm requires the "batch" dimension (everything except the
        # last normalisation dim) to be a power of 2 or a multiple of 32.
        # Pad the language portion of the prefix sequence to satisfy this.
        # Disable for float-vs-HF verification (LeRobot uses config tokenizer length).
        total_prefix = (
            self.policy_cfg.vision_tokens_num * self.policy_cfg.num_images
            + self.policy_cfg.tokenizer_max_length
            + 1
        )
        if vpu_align_prefix and total_prefix % 32 != 0:
            pad = 32 - (total_prefix % 32)
            self.policy_cfg.tokenizer_max_length += pad
            print(
                f"Padded tokenizer_max_length "
                f"{self.policy_cfg.tokenizer_max_length - pad} → "
                f"{self.policy_cfg.tokenizer_max_length} "
                f"(+{pad} pad tokens for VPU alignment, "
                f"prefix {total_prefix} → {total_prefix + pad})"
            )

        os.makedirs(output_model_path, exist_ok=True)
        self.output_vision_path = os.path.join(
            output_model_path, f"{model_type}_vision_ptq.hbm"
        )
        self.output_vlm_prefix_path = os.path.join(
            output_model_path, f"{model_type}_vlm_prefix_ptq.hbm"
        )
        self.output_action_expert_path = os.path.join(
            output_model_path, f"{model_type}_action_expert_ptq.hbm"
        )

        self.calib_data = load_calib_data_smolvla(
            calib_image_path,
            calib_text_path,
            self.policy_cfg,
            device=device,
        )
        self.calib_action_data_path = calib_action_data_path

        vtn = self.policy_cfg.vision_tokens_num
        self.model_vision = SmolVLMVision.build(
            input_model_path, self.policy_cfg, vtn
        )
        self.model_vlm = SmolVLMPrefix.build(input_model_path, self.policy_cfg, vtn)
        self.model_expert = SmolVLMActionExpert.build(
            input_model_path, self.policy_cfg, vtn
        )
        print("SmolVLA models loaded.")

        self._tokenizer = None
        self._load_processor_tokenizer()

    def _load_processor_tokenizer(self):
        try:
            from transformers import AutoProcessor
        except ImportError:
            print("transformers not installed; using dummy token ids for calibration")
            return
        root = Path(self.input_model_path)
        # Search for a local HF snapshot first (avoids network download)
        candidates = [
            root,
            *sorted(root.glob("models--*/**/snapshots/*/tokenizer.json")),
        ]
        local_path = None
        for c in candidates:
            p = c if c.is_dir() else c.parent
            if (p / "tokenizer.json").exists():
                local_path = str(p)
                break
        try:
            if local_path:
                self._processor = AutoProcessor.from_pretrained(
                    local_path, local_files_only=True
                )
            else:
                self._processor = AutoProcessor.from_pretrained(str(root))
            self._tokenizer = self._processor.tokenizer
        except Exception:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.policy_cfg.vlm_model_name,
                    cache_dir=str(root),
                    local_files_only=True,
                )
                self._tokenizer = self._processor.tokenizer
            except Exception:
                print("Warning: could not load processor/tokenizer; using dummy tokens for calibration")
                self._tokenizer = None

    def _tokenize(self, prompt: str):
        max_len = self.policy_cfg.tokenizer_max_length
        if self._tokenizer is None:
            tokens = [1] * min(8, max_len)
            return np.asarray(tokens + [0] * (max_len - len(tokens))), len(tokens)

        cleaned = prompt.strip()
        enc = self._tokenizer(
            cleaned,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        ids = enc["input_ids"][0].tolist()
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        valid = sum(1 for t in ids if t != pad_id)
        return np.asarray(ids, dtype=np.int64), min(valid, max_len)

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        device = (
            self.device
            if torch.cuda.is_available() and str(self.device).startswith("cuda")
            else "cpu"
        )
        dtype = torch.float16
        for m in (self.model_vision, self.model_vlm, self.model_expert):
            m.model.to(device=device, dtype=dtype)
            m.model.compile_mode(False)

        compile_kwargs = {}
        if vit_kwargs:
            compile_kwargs.update(vit_kwargs)
        if llm_kwargs:
            compile_kwargs.update(llm_kwargs)

        self._calibrate_forward(device=device, dtype=dtype, **compile_kwargs)

        for m in (self.model_vision, self.model_vlm, self.model_expert):
            m.model.compile_mode(True)
            m.model.to(device="cpu", dtype=dtype)

        self.model_vision.compile(
            output_model_path=self.output_vision_path, **compile_kwargs
        )
        self.model_vlm.compile(
            output_model_path=self.output_vlm_prefix_path, **compile_kwargs
        )
        self.model_expert.compile(
            output_model_path=self.output_action_expert_path, **compile_kwargs
        )

    def _calibrate_forward(self, *, device: str, dtype, **kwargs):
        patch = self.policy_cfg.vision_patch_size
        num_patches = (self.policy_cfg.image_height // patch) ** 2
        pos_ids = torch.arange(0, num_patches).view(1, num_patches).to(device)

        for calib_index, (image_data, prompt) in enumerate(self.calib_data):
            vision_chunks = []
            for img in image_data:
                out = self.model_vision.model(img, pos_ids)
                vision_chunks.append(out)
            inputs_embeds = torch.concat(vision_chunks, dim=1)
            vision_token_len = inputs_embeds.shape[1]

            lang_token, valid_token_len = self._tokenize(prompt)
            lang_token_t = (
                torch.from_numpy(lang_token)
                .unsqueeze(0)
                .to(device)
                .to(dtype=torch.int32)
            )
            prefix_pad_masks, _ = build_prefix_pad_att_masks(
                vision_token_len,
                valid_token_len,
                total_lang_len=self.policy_cfg.tokenizer_max_length,
                device=device,
            )
            attn_mask = build_vlm_prefix_mask(
                vision_token_len,
                valid_token_len,
                total_lang_len=self.policy_cfg.tokenizer_max_length,
                device=device,
            )

            state = np.load(f"{self.calib_action_data_path}/{calib_index}/state.npy")
            state_t = torch.from_numpy(state).to(device).to(dtype=dtype)
            position_ids = generate_prefix_position_ids(prefix_pad_masks)
            vlm_out = self.model_vlm.model.forward(
                tokens=lang_token_t,
                inputs_embeds=inputs_embeds,
                state=state_t,
                attention_mask=attn_mask,
                position_ids=position_ids,
            )
            kv_cache = vlm_out[1:]

            action_mask = build_action_expert_mask(
                prefix_pad_masks,
                action_len=self.policy_cfg.chunk_size,
                device=device,
            )
            position_ids = generate_denoise_position_ids(
                prefix_pad_masks,
                self.policy_cfg.chunk_size,
                device=device,
            )

            x_t = np.load(f"{self.calib_action_data_path}/{calib_index}/x_t.npy")
            x_t_t = torch.from_numpy(x_t).to(device).to(dtype=dtype)
            for step in range(self.policy_cfg.num_steps):
                denoise_idx = torch.tensor([step], dtype=torch.int32, device=device)
                x_t_t = self.model_expert.model.forward(
                    x_t=x_t_t,
                    denoise_idx=denoise_idx,
                    attention_mask=action_mask,
                    position_ids=position_ids,
                    caches=kv_cache,
                )
