import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import sentencepiece
import torch

from leap_llm.models.pi0.model_gemma import LanguageModel
from leap_llm.models.pi0.model_gemma_expert import GemmaExpertModel
from leap_llm.models.pi0.model_siglip import Siglip


def load_calib_data_pi0(
    calib_image_path, calib_text_path, device="cuda", dtype=torch.float16
):
    # 读取 prompt json
    with open(calib_text_path, encoding="utf-8") as f:
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

            inp = (inp - 127.5) / 127.5
            inp = np.transpose(inp, (2, 0, 1))
            inp = np.expand_dims(inp, axis=0)
            tensor = torch.from_numpy(inp).to(device).to(dtype=dtype)

            images.append(tensor)

        prompt = prompts[idx]

        results.append((images, prompt))

    return results

def build_gemma_mask(
    vision_len: int,
    valid_lang_len: int,
    neg_value: float = -32767.0,
    total_lang_len: int = 48,
    device: str = "cpu",
    dtype=torch.float16,
):

    seq_len = vision_len + total_lang_len

    # 初始化全部为 0
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)

    # 计算无效语言 token 索引
    invalid_lang_len = total_lang_len - valid_lang_len
    if invalid_lang_len > 0:
        invalid_idx = torch.arange(valid_lang_len, total_lang_len, device=device)
        invalid_pos = vision_len + invalid_idx  # shift 到全局 token 序列

        # 使用 index_fill 填整行/整列
        for idx in invalid_pos:
            mask[:, :, idx, :] = neg_value
            mask[:, :, :, idx] = neg_value

    return mask


def build_action_expert_mask(
    vision_len: int,
    valid_prompt_len: int,
    min_val: float = -32767.0,
    prompt_len: int = 48,
    action_len: int = 50,
    state_len: int = 14,
    rows: int = 64,
    device: str = "cpu",
    dtype=torch.float16,
):
    # 计算总列数
    total_cols = vision_len + prompt_len + state_len + action_len

    # 初始化 mask 全为 0
    mask = torch.zeros((1, 1, rows, total_cols), dtype=dtype, device=device)

    prompt_start = vision_len
    prompt_end = vision_len + prompt_len + state_len -1  # 不包含

    if valid_prompt_len < prompt_len:
        invalid_prompt_idx = torch.arange(valid_prompt_len, prompt_len + state_len -1, device=device)
        invalid_prompt_cols = prompt_start + invalid_prompt_idx

        # 整列设为 min_val
        mask[:, :, :, invalid_prompt_cols] = min_val

    state_col = prompt_end
    action_start = state_col + 1
    action_end = action_start + action_len  # 不包含

    mask[:, :, 0:state_len, action_start:action_end] = min_val

    return mask


def generate_action_position_ids(vision_token_num, valid_prompt_token, device="cuda"):
    start = vision_token_num + valid_prompt_token

    first_part = torch.full(
        (14,), start, dtype=torch.int32, device=device
    )  # 前14个相同

    second_part = torch.arange(
        start, start + 50, dtype=torch.int32, device=device
    )  # 后50个递增

    pos_ids = torch.cat([first_part, second_part], dim=0)
    pos_ids = pos_ids.view(1, 64)

    return pos_ids



class Pi0Api:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_text_path: str = None,
        calib_image_path: str = None,
        calib_action_data_path: str = None,
        device: str = "cpu",
        model_type: str = "pi0",
        dtype: str = "float16",
        vision_tokens_num=144,
        w_bits: int = 8,
    ):
        self.input_model_path = input_model_path
        self.device = device
        self.dtype = dtype
        self.model_type = model_type
        self.output_siglip_model_path = os.path.join(
            output_model_path,
            f"{self.model_type}_siglip_ptq.hbm",
        )
        self.output_gemma_llm_model_path = os.path.join(
            output_model_path,
            f"{self.model_type}_gemma_llm_ptq.hbm",  # noqa: E501
        )
        self.output_gemma_expert_model_path = os.path.join(
            output_model_path,
            f"{self.model_type}_gemma_expert_ptq.hbm",  # noqa: E501
        )
        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_dir = output_model_path

        # self.calib_text_data = load_text_data(calib_text_path)
        self.calib_data = load_calib_data_pi0(
            calib_image_path, calib_text_path, device=device
        )
        self.calib_action_data_path = calib_action_data_path

        self.model_siglip = Siglip.build(
            f"{self.input_model_path}/model.safetensors",
            vision_tokens_num
        )
        self.model_gemma_llm = LanguageModel.build(
            f"{self.input_model_path}/model.safetensors",
            vision_tokens_num
        )
        self.model_gemma_expert = GemmaExpertModel.build(
            f"{self.input_model_path}/model.safetensors",
            vision_tokens_num
        )
        print("Load model success!")
        self._max_len = 48
        path = Path(f"{input_model_path}/paligemma_tokenizer.model")
        print("Load tokenizer success!")
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        device = (
            self.device
            if torch.cuda.is_available() and self.device.startswith("cuda")
            else "cpu"
        )
        dtype = torch.float16
        self.model_siglip.model.to(device=device, dtype=dtype)
        self.model_gemma_llm.model.to(device=device, dtype=dtype)
        self.model_gemma_expert.model.to(device=device, dtype=dtype)
        self.model_siglip.model.compile_mode(False)
        self.model_gemma_llm.model.compile_mode(False)
        self.model_gemma_expert.model.compile_mode(False)

        compile_vit_kwargs = vit_kwargs or {}
        compile_llm_kwargs = llm_kwargs or {}
        compile_kwargs = {}
        compile_kwargs.update(compile_vit_kwargs)
        compile_kwargs.update(compile_llm_kwargs)
        # Save embedding weights for engine consumption before to fp16
        self._calibrate_forward(device=device, dtype=dtype, **compile_kwargs)

        self.model_siglip.model.compile_mode(True)
        self.model_gemma_llm.model.compile_mode(True)
        self.model_gemma_expert.model.compile_mode(True)
        self.model_siglip.model.to(device="cpu", dtype=dtype)
        self.model_gemma_llm.model.to(device="cpu", dtype=dtype)
        self.model_gemma_expert.model.to(device="cpu", dtype=dtype)

        self.model_siglip.compile(
            output_model_path=self.output_siglip_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )
        self.model_gemma_llm.compile(
            output_model_path=self.output_gemma_llm_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )
        self.model_gemma_expert.compile(
            output_model_path=self.output_gemma_expert_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )

    def _tokenize(self, prompt):
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(
            cleaned_text, add_bos=True
        ) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
        else:
            tokens = tokens[: self._max_len]
            tokens_len = self._max_len

        return np.asarray(tokens), tokens_len

    def _calibrate_forward(self, *, device: str, dtype, **kwargs):
        siglip_pos_ids = torch.arange(0, 256).view(1, 256).to(device)
        # np.save(f"{tmp_npy_dir}/inp.npy", inp.detach().cpu().numpy())
        calib_index = 0
        for image_data, prompt in self.calib_data:
            lang_token, valid_token_len = self._tokenize(prompt)
            siglip_outputs = []
            for i in range(len(image_data)):
                siglip_output = self.model_siglip.model.forward(
                    image_data[i], siglip_pos_ids
                )
                siglip_outputs.append(siglip_output)

            inputs_embeds = torch.concat(siglip_outputs, dim=1)
            vision_token_len = inputs_embeds.shape[1]
            lang_token = torch.from_numpy(lang_token).unsqueeze(0).to(device).to(dtype=torch.int32)
            attention_mask = build_gemma_mask(vision_token_len, valid_token_len, device=device)
            gemma_outputs = self.model_gemma_llm.model.forward(
                tokens=lang_token,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            kv_cache = gemma_outputs[1:]
            action_mask = build_action_expert_mask(vision_token_len, valid_token_len, device=device)
            
            position_ids = generate_action_position_ids(vision_token_len, valid_token_len, device=device)
            action_state = np.load(f"{self.calib_action_data_path}/{calib_index}/state.npy")
            action_x_t = np.load(f"{self.calib_action_data_path}/{calib_index}/x_t.npy")
            action_state = torch.from_numpy(action_state).to(device).to(dtype=dtype)
            action_x_t = torch.from_numpy(action_x_t).to(device).to(dtype=dtype)
            self.model_gemma_expert.model.forward(
                state=action_state,
                x_t=action_x_t,
                attention_mask=action_mask,
                position_ids=position_ids,
                caches=kv_cache,
            )
        
            calib_index += 1 # noqa: SIM113
