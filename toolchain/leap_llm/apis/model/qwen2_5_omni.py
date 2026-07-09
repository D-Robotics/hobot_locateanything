import os
from typing import List, Optional, Tuple

import librosa
import torch
from qwen_omni_utils import process_mm_info
from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

from leap_llm.apis.calibration.audio_process import AudioChunkProcessor
from leap_llm.apis.calibration.calibration import (
    create_chunk_mask,
    pad_to_multiple,
    update_causal_mask,
)
from leap_llm.apis.calibration.data_loader import load_message_data
from leap_llm.models.qwen2_5_omni.model import Qwen2_5Omni, save_model_checkpoint


def has_audio_librosa(path, sr=16000):
    """
    Check if a video file has audio using librosa.

    Args:
        path: Path to the video file
        sr: Sampling rate (None means use the file's original sampling rate)

    Returns:
        True if an audio signal was successfully loaded and has length > 0
        False if loading fails or the returned signal is empty
    """
    try:
        # y: np.ndarray containing the audio waveform; _ is the actual sample rate used
        y, _ = librosa.load(path, sr=sr)
        return y.size > 0
    except Exception:
        # Possible reasons: no audio track, unsupported format, or missing ffmpeg
        return False


def find_videos(conversation):
    """Find videos in the conversation.

    Args:
        conversation (list): Conversation data.

    Returns:
        list: List of video paths.
    """
    videos = []
    for msg in conversation:
        for ele in msg.get("content", []):
            if ele.get("type") == "video":
                videos.append(ele)
    return videos


def ensure_visual_dimensions(conversation):
    """
    Ensure each video and image element in the conversation has the
    correct resized dimensions.

    Args:
        conversation (dict or list): A single message dict or a list of
            message dicts, each containing a 'content' key with a list of
            elements.

    Returns:
        dict or list: The updated conversation with enforced resized
            dimensions for video and image elements.
    """
    # Use the same dimensions as the test file
    resized_width = 448
    resized_height = 448

    for msg in conversation:
        contents = msg.get("content")
        if not isinstance(contents, list):
            continue

        for ele in contents:
            if ele.get("type") in ["image", "video"]:
                ele["resized_width"] = resized_width
                ele["resized_height"] = resized_height

    return conversation


def preprocess(processor, conversation):
    """Process conversation data for Qwen2.5-Omni.

    Args:
        processor (Qwen2_5OmniProcessor): Processor for Qwen2.5-Omni.
        conversation (list): Conversation data.

    Returns:
        dict: Processed conversation data.
    """
    use_audio_in_video = False
    videos = find_videos(conversation)

    if len(videos) > 0 and has_audio_librosa(videos[0]["video"]):
        use_audio_in_video = True

    conversation = ensure_visual_dimensions(conversation)

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    _mm = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    audios, images, videos = _mm[0], _mm[1], _mm[2]
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs["use_audio_in_video"] = use_audio_in_video
    return inputs


def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: List[int],
    grid_hs: List[int],
    grid_ws: List[int],
) -> torch.Tensor:
    """
    Calculate 3D RoPE indices for image/video tokens.

    Args:
        start_idx (int): Start index of the vision tokens.
        vision_idx (int): Index of the vision token.
        spatial_merge_size (int): Spatial merge size.
        t_index (List[int]): Temporal index.
        grid_hs (List[int]): Grid height.
        grid_ws (List[int]): Grid width.

    Returns:
        torch.Tensor: RoPE indices for image/video tokens.
    """
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = (
        torch.arange(llm_grid_h)
        .view(1, -1, 1)
        .expand(len(t_index), -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w)
        .view(1, 1, -1)
        .expand(len(t_index), llm_grid_h, -1)
        .flatten()
    )
    t_index = t_index.view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def get_chunked_index(
    token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
) -> list[tuple[int, int]]:
    """
    Split token index list into chunks based on token value ranges.

    Args:
        token_indices (torch.Tensor of shape (seq_len,)): Monotonically increasing
            token index values.
        tokens_per_chunk (int): Number of tokens per chunk (threshold).
        remove_index (int): Index to subtract from `token_indices` before chunking.

    Returns:
        list[tuple[int, int]]: Tuples of start (inclusive) and end (exclusive)
            indices for each chunk in `token_indices`.
    """

    def _iter():
        i, start_idx = 0, 0  # skip bos token
        current_chunk = 1
        while i < len(token_indices):  # skip eos token
            if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                yield (start_idx, i)
                start_idx = i
                current_chunk += 1
            i += 1
        yield (start_idx, len(token_indices))

    return list(_iter())


def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.LongTensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate 3D RoPE indices for image/video tokens and 1D RoPE indices for
    text tokens.

    Explanation:
        Each embedding sequence may contain both vision and text tokens or only
        text tokens.

        Pure text sequence:
            input_ids: [T T T T T]
            temporal/height/width position_ids: [0, 1, 2, 3, 4]

        Vision + text sequence:
            Temporal: N patches (time segments)
            Height: H patches
            Width: W patches
            Key parameters:
              - fps: frames per second (e.g., 1)
              - tokens_per_second: temporal granularity (e.g., 25)
              - temporal_patch_size: frames per temporal patch (e.g., 2)
              - interval: tokens_per_second * temporal_patch_size / fps
                (e.g., 25 * 2 / 1 = 50)

            Example layout:
              input_ids: [V ... V T ... T]
              vision temporal ids: [0, 0, ..., 50, 50, ..., 100, ...]
              vision height ids: [0, 0, 1, 1, ...]
              vision width ids: [0, 1, 0, 1, ...]
              text temporal/height/width ids: [start, start+1, ...]
              Text start is max(vision ids) + 1.

    Args:
        input_ids (torch.LongTensor of shape (batch_size, seq_len)):
            Indices of input sequence tokens in the vocabulary.
        image_grid_thw (torch.LongTensor of shape (num_images, 3), optional):
            Temporal, height and width of each image feature in LLM.
        video_grid_thw (torch.LongTensor of shape (num_videos, 3), optional):
            Temporal, height and width of each video feature in LLM.
        attention_mask (torch.Tensor of shape (batch_size, seq_len), optional):
            1 for tokens that are not masked, 0 for masked.
        use_audio_in_video (bool, optional):
            Whether to use audio within video.
        audio_seqlens (torch.LongTensor of shape (num_audios), optional):
            Feature length of each audio in LLM.
        second_per_grids (torch.LongTensor of shape (num_videos), optional):
            Time interval (seconds) for each temporal grid in 3D position IDs.

    Returns:
        position_ids (torch.LongTensor of shape (3, batch_size, seq_len))
        mrope_position_deltas (torch.Tensor of shape (batch_size))
    """
    spatial_merge_size = 2
    image_token_id = 151655
    video_token_id = 151656
    audio_token_id = 151646
    vision_start_token_id = 151652
    audio_start_token_id = 151647
    position_id_per_seconds = 25
    seconds_per_chunk = 2

    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_idx, video_idx, audio_idx = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums, audio_nums = 0, 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (
                (vision_tokens == audio_start_token_id).sum()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum()
            )
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = (
                image_nums,
                video_nums,
                audio_nums,
            )
            multimodal_nums = (
                image_nums + audio_nums
                if use_audio_in_video
                else image_nums + video_nums + audio_nums
            )
            # print("=== multimodal_nums:", multimodal_nums)
            for _ in range(multimodal_nums):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                min_ed = min(ed_image, ed_video, ed_audio)
                if min_ed == ed_audio:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    bos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    llm_pos_ids = (
                        torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    eos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st += text_len + bos_len + audio_len + eos_len
                    audio_idx += 1
                    remain_audios -= 1

                elif min_ed == ed_image:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    bos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t) * 1 * position_id_per_seconds
                    ).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (
                        spatial_merge_size**2
                    )
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    eos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st += text_len + bos_len + image_len + eos_len
                    image_idx += 1
                    remain_images -= 1

                elif min_ed == ed_video and not use_audio_in_video:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    bos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t)
                        * second_per_grids[video_idx].cpu().float()
                        * position_id_per_seconds
                    ).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_len = video_grid_thw[video_idx].prod() // (
                        spatial_merge_size**2
                    )
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    eos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st += text_len + bos_len + video_len + eos_len
                    video_idx += 1
                    remain_videos -= 1

                elif min_ed == ed_video and use_audio_in_video:
                    text_len = min_ed - st - 2
                    if text_len != 0:
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    bos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    audio_llm_pos_ids = (
                        torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]

                    t_index = (
                        torch.arange(grid_t)
                        * second_per_grids[video_idx].cpu().float()
                        * position_id_per_seconds
                    ).long()
                    video_llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )

                    t_ntoken_per_chunk = int(
                        position_id_per_seconds * seconds_per_chunk
                    )
                    video_chunk_indexes = get_chunked_index(
                        video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx
                    )
                    audio_chunk_indexes = get_chunked_index(
                        audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx
                    )
                    sub_len = 0
                    for j in range(
                        max(len(video_chunk_indexes), len(audio_chunk_indexes))
                    ):
                        video_chunk_index = (
                            video_chunk_indexes[j]
                            if j < len(video_chunk_indexes)
                            else None
                        )
                        audio_chunk_index = (
                            audio_chunk_indexes[j]
                            if j < len(audio_chunk_indexes)
                            else None
                        )
                        if video_chunk_index is not None:
                            sub_len += video_chunk_index[1] - video_chunk_index[0]

                            llm_pos_ids_list.append(
                                video_llm_pos_ids[
                                    :, video_chunk_index[0] : video_chunk_index[1]
                                ]
                            )
                        if audio_chunk_index is not None:
                            sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[
                                    :, audio_chunk_index[0] : audio_chunk_index[1]
                                ]
                            )
                    video_len = video_grid_thw[video_idx].prod() // (
                        spatial_merge_size**2
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    eos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            # print("llm_positions:", llm_positions.shape)
            # llm_positions: torch.Size([3, 908])
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)

        return position_ids, mrope_position_deltas
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = (
            position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        )
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[
            0
        ]
        mrope_position_deltas = (
            max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        )

        return position_ids, mrope_position_deltas


class Qwen2_5OmniApi:
    """API for Qwen2.5-Omni multimodal model compilation."""

    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_message_path: str = None,
        chunk_size: int = 256,
        cache_len: int = 1024,
        device: str = "cpu",
        dtype: str = "float32",
        preserve_precision: bool = True,
        model_type: str = "qwen2_5_omni",
        n_window: int = 100,
        feature_size: int = 128,
        chunk_length: int = 30,
        sampling_rate: int = 16000,
        attn_min_value: float = -512.0,
        w_bits: int = 8,
    ):
        self.input_model_path = input_model_path
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.device = device
        self.dtype = torch.float32 if dtype == "float32" else torch.float16
        self.preserve_precision = preserve_precision
        self.model_type = model_type
        self.w_bits = w_bits

        # Audio processing parameters
        self.n_window = n_window
        self.feature_size = feature_size
        self.chunk_length = chunk_length
        self.sampling_rate = sampling_rate
        self.attn_min_value = attn_min_value

        self.calib_messages = list(load_message_data(calib_message_path, model_type))

        self._setup_output_paths(output_model_path, model_type)

        self._build_model()

        # Initialize audio processor
        self.audio_processor = AudioChunkProcessor(
            device=device,
            n_window=n_window,
            feature_size=feature_size,
            chunk_length=chunk_length,
            sampling_rate=sampling_rate,
            attn_min_value=attn_min_value,
        )

    def _setup_output_paths(self, output_model_path, model_type):
        """Set up output paths for different modules."""
        os.makedirs(output_model_path, exist_ok=True)

        self.base_output_path = output_model_path
        self.output_paths = {
            "audio_tower": os.path.join(
                output_model_path,
                f"{model_type}_audio_tower_chunk_{self.chunk_size}.hbm",
            ),
            "visual": os.path.join(
                output_model_path,
                f"{model_type}_visual_chunk_{self.chunk_size}.hbm",
            ),
            "text_model": os.path.join(
                output_model_path,
                f"{model_type}_text_model_cache_{self.cache_len}_"
                f"chunk_{self.chunk_size}_q{self.w_bits}.hbm",
            ),
            "all": os.path.join(
                output_model_path,
                f"{model_type}_all_cache_{self.cache_len}_chunk_{self.chunk_size}.hbm",
            ),
        }

    def _build_model(self):
        """Build and initialize the model."""
        ckpt_dir = save_model_checkpoint(
            self.input_model_path, os.path.dirname(self.output_paths["all"])
        )
        self.qwen2_5_omni_model = Qwen2_5Omni.build(
            ckpt_dir,
            chunk_size=self.chunk_size,
            cache_len=self.cache_len,
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.input_model_path)

    def compile(self, stage: str = "all", skip_final_compile=False, vit_kwargs=None, llm_kwargs=None):
        """Compile the model for specified stage."""
        device = self.device if torch.cuda.is_available() else "cpu"
        self.qwen2_5_omni_model.model.to(device=device, dtype=self.dtype)
        self.qwen2_5_omni_model.model.compile_mode(False)
        self.qwen2_5_omni_model.model.eval()
        audio_tower = self.qwen2_5_omni_model.get_audio_tower()
        visual = self.qwen2_5_omni_model.get_visual()
        text_model = self.qwen2_5_omni_model.get_text_model()
        embed_tokens = self.qwen2_5_omni_model.get_input_embeddings()

        compile_vit_kwargs = vit_kwargs or {}
        compile_llm_kwargs = llm_kwargs or {}
        compile_kwargs = {}
        compile_kwargs.update(compile_vit_kwargs)
        compile_kwargs.update(compile_llm_kwargs)

        for conversation in self.calib_messages:
            inputs = preprocess(self.processor, conversation)

            input_ids = inputs["input_ids"].to(device)
            inputs_embeds = embed_tokens(input_ids)
            use_audio_in_video = inputs["use_audio_in_video"]

            audio_feature_lengths = None
            cfg = self.qwen2_5_omni_model.model_args.thinker_config
            if stage in ["audio_tower", "text_model", "all"]:
                audio_feature_lengths = self.calib_audio_features(
                    inputs, input_ids, inputs_embeds, audio_tower, device, cfg
                )

            if stage in ["visual", "text_model", "all"]:
                self.calib_visual_features(
                    inputs, input_ids, inputs_embeds, visual, device, cfg
                )

            if stage in ["text_model", "all"]:
                self._process_text_chunks(
                    inputs,
                    inputs_embeds,
                    text_model,
                    embed_tokens,
                    use_audio_in_video,
                    audio_feature_lengths,
                    device,
                    cfg,
                )

        if stage in ["text_model", "all"]:
            self._save_embed_tokens()
            text_model = self.qwen2_5_omni_model.get_text_model()
            text_model.save_cos_sin(self.base_output_path)

            text_model.compile_mode(True)
            text_model.to("cpu")
            if not skip_final_compile:
                text_model.compile(
                    stage="text_model",
                    output_model_path=self.output_paths["text_model"],
                    enable_vpu=True,
                    **compile_kwargs,
                )

        if stage in ["audio_tower", "text_model", "all"]:
            audio_tower = self.qwen2_5_omni_model.get_audio_tower()
            audio_tower.compile_mode(True)
            audio_tower.to("cpu")
            if not skip_final_compile:
                audio_tower.compile(
                    stage="audio_tower",
                    output_model_path=self.output_paths["audio_tower"],
                    enable_vpu=True,
                    **compile_kwargs,
                )

        if stage in ["visual", "text_model", "all"]:
            visual = self.qwen2_5_omni_model.get_visual()
            visual.compile_mode(True)
            visual.to("cpu")
            if not skip_final_compile:
                visual.compile(
                    stage="visual",
                    output_model_path=self.output_paths["visual"],
                    enable_vpu=True,
                    **compile_kwargs,
                )

    def _has_audio_data(self) -> bool:
        """Check if calibration data contains audio."""
        return any("audio" in str(conv) for conv in self.calib_messages)

    def _has_visual_data(self) -> bool:
        """Check if calibration data contains visual content."""
        return any(
            "image" in str(conv) or "video" in str(conv) for conv in self.calib_messages
        )

    def _save_embed_tokens(self):
        """Save embedding tokens to file."""
        embed_tokens = self.qwen2_5_omni_model.get_input_embeddings()
        embed_tokens_path = os.path.join(self.base_output_path, "embed_tokens.bin")
        if not os.path.exists(embed_tokens_path):
            with torch.no_grad():
                weights = embed_tokens.weight.detach().cpu().numpy()
                weights.tofile(embed_tokens_path)

    def calib_audio_features(
        self, inputs, input_ids, inputs_embeds, audio_tower, device, thinker_cfg
    ) -> Optional[torch.Tensor]:
        """Process audio features and update embeddings."""
        input_features = inputs.get("input_features", None)
        if input_features is None:
            return None

        feature_attention_mask = inputs.get("feature_attention_mask", None)
        audio_feature_lengths = (
            torch.sum(feature_attention_mask, dim=1)
            if feature_attention_mask is not None
            else None
        )

        if feature_attention_mask is not None:
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)

        # TODO: add feature_attention_mask None check
        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )

        feat_list, mask_list, attn_mask_list = self.audio_processor(
            input_features, feature_lens
        )

        audio_features = None
        for feat, mask, attn_mask in zip(feat_list, mask_list, attn_mask_list):
            valid_num = (attn_mask[0][0] == 0).sum().item()
            with torch.no_grad():
                audio_features_c = audio_tower.forward(feat, mask, attn_mask)
            if valid_num // 2 != audio_features_c.shape[0]:
                audio_features_c = audio_features_c[: valid_num // 2]

            if audio_features is None:
                audio_features = audio_features_c
            else:
                audio_features = torch.cat([audio_features, audio_features_c], dim=0)

        # Update embeddings with audio features
        audio_token_id = thinker_cfg.audio_token_index
        audio_mask = (
            (input_ids == audio_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds.masked_scatter_(audio_mask, audio_features)

        return audio_feature_lengths

    def calib_visual_features(
        self, inputs, input_ids, inputs_embeds, visual, device, thinker_cfg
    ):
        """Process visual features and update embeddings."""
        # Process images
        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

            image_grid_thw = inputs.get("image_grid_thw")

            seq_len = image_grid_thw[0][-1] * image_grid_thw[0][-2]
            pixel_values_chunks = pixel_values.split(seq_len)
            image_embeds = None

            for _, pixel_values_chunk in enumerate(pixel_values_chunks):
                with torch.no_grad():
                    image_embeds_chunk = visual.forward(pixel_values_chunk)
                    if image_embeds is None:
                        image_embeds = image_embeds_chunk
                    else:
                        image_embeds = torch.cat(
                            [image_embeds, image_embeds_chunk], dim=0
                        )

            image_token_id = thinker_cfg.image_token_index
            image_mask = (
                (input_ids == image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds.masked_scatter_(image_mask, image_embeds)

        # Process videos
        pixel_values_videos = inputs.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(device)
            video_grid_thw = inputs.get("video_grid_thw")
            seq_len = video_grid_thw[0][-1] * video_grid_thw[0][-2]

            pixel_values_videos_chunks = pixel_values_videos.split(seq_len)
            video_embeds = None

            for pixel_values_videos_chunk in pixel_values_videos_chunks:
                with torch.no_grad():
                    video_embeds_chunk = visual.forward(pixel_values_videos_chunk)

                if video_embeds is None:
                    video_embeds = video_embeds_chunk
                else:
                    video_embeds = torch.cat([video_embeds, video_embeds_chunk], dim=0)

            video_token_id = thinker_cfg.video_token_index
            video_mask = (
                (input_ids == video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds.masked_scatter_(video_mask, video_embeds)

    def _process_text_chunks(
        self,
        inputs,
        inputs_embeds,
        text_model,
        embed_tokens,
        use_audio_in_video,
        audio_feature_lengths,
        device,
        thinker_cfg,
    ):
        """Process text model with chunked inputs."""
        # Get position embeddings
        position_ids, _ = get_rope_index(
            inputs["input_ids"],
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            inputs["attention_mask"],
            use_audio_in_video,
            audio_feature_lengths,
            inputs.get("video_second_per_grid"),
        )
        inputs["position_ids"] = position_ids

        # Pad inputs to chunk size (match test cali_text_model)
        position_ids = inputs["position_ids"].to(device)
        raw_token_num = position_ids.shape[-1]
        inputs_pad_len, pad_num = pad_to_multiple(
            raw_token_num, multiple=self.chunk_size
        )
        valid_num = position_ids.shape[-1]
        pad_token_id = text_model.config.pad_token_id
        pad_token = [pad_token_id] * pad_num

        input_ids_pad = torch.tensor([pad_token], device=device)
        input_ids = inputs["input_ids"].to(device)
        input_ids = torch.concatenate([input_ids_pad, input_ids], dim=-1)

        pad_embeds = embed_tokens(input_ids_pad)

        attention_mask = torch.zeros((1, self.cache_len), dtype=torch.int32).to(device)
        attention_mask[0, -valid_num:] = 1
        mask_value = -512
        position_ids_pad = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        inputs_embeds_pad = pad_embeds
        inputs_embeds = torch.concatenate([inputs_embeds_pad, inputs_embeds], dim=1)

        cache_position = torch.arange(
            0, self.cache_len, dtype=torch.long, device=device
        )

        causal_mask = update_causal_mask(
            attention_mask,
            input_ids,
            cache_position,
            min_dtype=mask_value,
            sequence_length=self.cache_len,
            kv_cache_len=self.cache_len,
            dtype=torch.float32,
            device=device,
        )
        causal_mask = causal_mask[:, :, -inputs_pad_len:, :]

        position_ids_pad = position_ids_pad[:, :, :pad_num]
        position_ids = torch.concatenate([position_ids_pad, position_ids], dim=-1)

        inputs_embeds_chunks = inputs_embeds.split(self.chunk_size, dim=-2)
        position_ids_chunks = position_ids.split(self.chunk_size, dim=-1)
        causal_mask_chunks = causal_mask.split(self.chunk_size, dim=-2)

        mask_chunks = []
        for chunk_index, _ in enumerate(causal_mask_chunks):
            chunk_mask_2d = create_chunk_mask(
                chunk_index=chunk_index,
                valid_seq_len=valid_num,
                chunk_size=self.chunk_size,
                kv_cache_len=self.cache_len,
                mask_value=mask_value,
                padding_side="left",
                device=str(device),
            )
            chunk_mask = chunk_mask_2d.unsqueeze(0).unsqueeze(0)
            mask_chunks.append(chunk_mask.to(device))

        num_key_value_heads = text_model.config.num_key_value_heads
        head_dim = text_model.config.head_dim
        num_hidden_layers = text_model.config.num_hidden_layers

        init_kv_data = torch.zeros(
            [1, self.cache_len, num_key_value_heads, head_dim],
            dtype=torch.float32,
        ).to(device)

        # Match test: duplicate list for key and value stacks
        past_key_values_list = [init_kv_data] * num_hidden_layers + [
            init_kv_data
        ] * num_hidden_layers

        # Process chunks
        for inp_chunk, msk_chunk, pos_chunk in zip(
            inputs_embeds_chunks, mask_chunks, position_ids_chunks
        ):
            with torch.no_grad():
                rotary_pos_emb = text_model.get_rotary_pos_emb(pos_chunk)
                outputs = text_model.forward(
                    inputs_embeds=inp_chunk,
                    attention_mask=msk_chunk,
                    rotary_pos_emb=rotary_pos_emb,
                    caches=past_key_values_list,
                )

            # Update KV caches
            for idx in range(len(past_key_values_list)):
                new_cache = outputs[idx + 1]
                past = past_key_values_list[idx]
                slice_past = past[:, self.chunk_size :]
                past_key_values_list[idx] = torch.concat([slice_past, new_cache], dim=1)
