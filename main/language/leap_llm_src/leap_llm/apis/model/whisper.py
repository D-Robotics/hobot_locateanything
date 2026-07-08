import os

import librosa
import torch
from transformers import WhisperProcessor

from leap_llm.models.whisper.model import Whisper


def generate_kv_mask(
    batch_size: int,
    num_heads: int,
    max_cache_len: int,
    current_len: int,
    device="cpu",
    dtype=torch.float16,
):
    """
    为固定长度 KV cache 生成 padding mask
    - 左侧 (max_cache_len - current_len) 为 padding → -inf
    - 右侧 current_len 为有效 KV → 0
    - shape: [batch_size, 1, 1, max_cache_len] → 可广播到 [B, H, T_q, T_k]
    """
    mask = torch.zeros((batch_size, 1, 1, max_cache_len), device=device, dtype=dtype)

    # 左侧 padding 区域设为大负数（-32767 在 fp16 下足够）
    if current_len < max_cache_len:
        mask[:, :, :, : max_cache_len - current_len] = -32767.0

    # 右侧 current_len 区域保持 0（可见）
    return mask


class WhisperApi:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_audio_path: str,
        device: str = "cpu",
        model_type: str = "whisper",
        dtype: str = "float16",
        w_bits: int = 8,
    ):
        self.input_model_path = input_model_path
        self.device = device
        self.dtype = dtype
        self.model_type = model_type
        self.cache_len = 128

        self.output_model_path = os.path.join(
            output_model_path,
            f"{self.model_type}_{self.cache_len}_ptq.hbm",  # noqa: E501
        )
        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_dir = output_model_path
        self.calib_wav_dir = calib_audio_path
        self.model_whisper = Whisper.build(
            f"{self.input_model_path}", cache_len=self.cache_len
        )
        self.processor = WhisperProcessor.from_pretrained(self.input_model_path)
        print("Load model success!")

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        device = (
            self.device
            if torch.cuda.is_available() and self.device.startswith("cuda")
            else "cpu"
        )
        dtype = torch.float16
        self.model_whisper.model.to(device=device, dtype=dtype)
        self.model_whisper.model.compile_mode(False)

        compile_vit_kwargs = vit_kwargs or {}
        compile_llm_kwargs = llm_kwargs or {}
        compile_kwargs = {}
        compile_kwargs.update(compile_vit_kwargs)
        compile_kwargs.update(compile_llm_kwargs)
        # Save embedding weights for engine consumption before to fp16
        self._calibrate_forward(device=device, dtype=dtype, **compile_kwargs)

        compile_kwargs["cache_len"] = self.cache_len
        self.model_whisper.model.compile_mode(True)
        self.model_whisper.model.to(device="cpu", dtype=dtype)
        self.model_whisper.compile(
            stage="all",
            output_model_path=self.output_model_path,
            enable_vpu=True,
            **compile_kwargs,
        )

    def _calibrate_forward(self, *, device: str, dtype, **kwargs):
        audio_paths = []
        for filename in os.listdir(self.calib_wav_dir):
            if filename.endswith(".wav"):
                wav_path = os.path.join(self.calib_wav_dir, filename)
                audio_paths.append(wav_path)
        # for i in range(len(audio_paths)):
        for i in range(len(audio_paths)):
            audio_path = audio_paths[i]
            # ref = np.load("/mnt/data/weiyang.hu/whisper-medium/encoder_outputs.npy")
            print(f"process {audio_path}")
            audio, sr = librosa.load(
                audio_path,
                sr=16000,  # 强制重采样到 16k
                mono=True,  # 强制转单声道
            )
            input_feature = self.processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(dtype)

            position_ids = torch.arange(1500, dtype=torch.int32, device=device)
            position_ids = position_ids.unsqueeze(0)

            encoder_outputs = self.model_whisper.model.encoder.forward(
                input_features=torch.tensor(input_feature).to(device).to(dtype=dtype),
                position_ids=position_ids,
            )
            cur_position = 0
            next_tokens = 0

            pred_ids = []
            while next_tokens != 50257:
                if cur_position == 0:
                    position_ids = (
                        torch.arange(4)
                        .unsqueeze(0)
                        .to(dtype=torch.int32, device=device)
                    )
                    decoder_input_ids = [50258, 50260, 50359, 50363]

                    input_ids = torch.tensor(
                        [decoder_input_ids], dtype=torch.long, device=device
                    )
                    outputs = self.model_whisper.model.decoder.forward(
                        input_ids=input_ids,
                        encoder_hidden_states=encoder_outputs,
                        position_ids=position_ids,
                    )
                    next_token_logits = outputs[0][:, -1, :].to(
                        copy=True, dtype=dtype, device=device
                    )
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    cur_position += 4
                else:
                    position_ids = (
                        torch.arange(cur_position, cur_position + 1)
                        .unsqueeze(0)
                        .to(dtype=torch.int32, device=device)
                    )
                    input_ids = next_tokens.unsqueeze(0)
                    attn_mask = generate_kv_mask(
                        1,
                        1,
                        self.cache_len,
                        cur_position + 1,
                        device=device,
                        dtype=dtype,
                    )
                    outputs = self.model_whisper.model.decoder.forward(
                        input_ids=input_ids,
                        encoder_hidden_states=encoder_outputs,
                        position_ids=position_ids,
                        attention_mask=attn_mask,
                        caches=outputs[1:],
                    )
                    next_token_logits = outputs[0][:, -1, :].to(
                        copy=True, dtype=dtype, device=device
                    )
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    cur_position += 1
                pred_ids.append(next_tokens)
            pred_ids = torch.tensor(pred_ids).unsqueeze(0)
            transcription = self.processor.batch_decode(
                pred_ids, skip_special_tokens=False
            )
            print(f"result: {transcription}")
