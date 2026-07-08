import json
import math
import os
from typing import Any, List, Tuple

import torch
from transformers import AutoTokenizer


def pad_to_multiple(n: int, multiple: int = 256) -> Tuple[int, int]:
    """
    Compute the smallest multiple of 'multiple' that is greater than or equal to n,
    and the amount required to pad n to that multiple.

    Args:
        n: The original value.
        multiple: The factor to multiply.

    Returns:
        rounded: The smallest multiple of 'multiple' that is >= n.
        padding: The number needed to pad n.
    """
    rounded = math.ceil(n / multiple) * multiple
    padding = rounded - n
    return rounded, padding


def create_chunk_mask(
    *,
    chunk_index: int,
    valid_seq_len: int,
    chunk_size: int,
    kv_cache_len: int,
    mask_value: float,
    padding_side: str,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a causal mask matrix for a single Q-chunk based on the chunking rules."""
    if valid_seq_len <= 0:
        return torch.full(
            (chunk_size, kv_cache_len), mask_value, device=device, dtype=torch.float32
        )

    row_idx = torch.arange(chunk_size, device=device).view(-1, 1)
    col_idx = torch.arange(kv_cache_len, device=device).view(1, -1)

    if padding_side == "right":
        S = chunk_size
        L = valid_seq_len
        K = kv_cache_len

        tokens_before = min(chunk_index * S, L)
        current = max(min(L - tokens_before, S), 0)
        active_rows = row_idx < current

        start_scalar = K - (chunk_index + 1) * S
        start = torch.full_like(col_idx, start_scalar)
        end_exclusive = (K - S) + row_idx
        end_exclusive = end_exclusive.clamp(max=K)
        allowed = active_rows & (col_idx >= start) & (col_idx < end_exclusive)
    else:
        S = chunk_size
        L = valid_seq_len
        K = kv_cache_len
        r = L % S
        first_chunk_tokens = r if r != 0 else S
        if chunk_index == 0 and first_chunk_tokens < S:
            valid_row_offset = row_idx - (S - first_chunk_tokens)
            active_rows = valid_row_offset >= 0
            start_scalar = K - first_chunk_tokens
            start = torch.full_like(col_idx, start_scalar)
            end = (start_scalar + valid_row_offset + 1).clamp(min=0, max=K)
            allowed = active_rows & (col_idx >= start) & (col_idx < end)
        else:
            if first_chunk_tokens < S:
                tokens_before = first_chunk_tokens + (chunk_index - 1) * S
            else:
                tokens_before = chunk_index * S
            tokens_before = min(tokens_before, L)

            rect_start = K - S - tokens_before
            tri_start = K - S
            start = torch.full_like(col_idx, rect_start)
            end = (tri_start + row_idx + 1).clamp(max=K)
            allowed = (col_idx >= start) & (col_idx < end)

    full = torch.full(
        (chunk_size, kv_cache_len), mask_value, device=device, dtype=torch.float32
    )
    return full.masked_fill(allowed, 0)


def get_position_ids(
    attention_mask: torch.Tensor, pos_mask_value: int = 1
) -> torch.Tensor:
    """
    Compute position_ids by cumulatively summing the attention mask.
    Positions corresponding to actual tokens (attention_mask == 1)
    are assigned cumulative positions, and positions for padding
    (attention_mask == 0) are set to 1.

    Args:
        attention_mask: The attention mask tensor.
        pos_mask_value: The value to fill for padding positions.

    Returns:
        A tensor of position_ids.
    """
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, pos_mask_value)
    return position_ids


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
    padding_side: str = "left",
) -> torch.Tensor:
    """
    Construct a 4D causal attention mask of shape
    (batch_size, 1, query_length, key_value_length).

    If the input attention_mask is already 4D, it is returned directly.

    Args:
        attention_mask: A tensor with shape (batch_size, key_value_length)
            or (batch_size, 1, query_length, key_value_length).
        sequence_length: The length of the input sequence.
        target_length: The target length. When using a static cache,
            the mask length should match the cache.
        dtype: The data type of the constructed mask.
        device: The device on which the mask is placed.
        min_dtype: The minimum representable value for the given dtype.
        cache_position: Tensor indicating the positions of input tokens in the sequence.
        batch_size: Batch size.

    Returns:
        The constructed 4D causal mask.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # If the mask is already 4D, return it directly.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device).reshape(
            1, -1
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            kv_len = causal_mask.shape[-1]
            if padding_side == "left":
                start = kv_len - mask_length
                end = kv_len
            else:
                start = 0
                end = mask_length
            padding_mask = (
                causal_mask[:, :, :, start:end] + attention_mask[:, None, None, :]
            ) == 0
            causal_mask[:, :, :, start:end] = causal_mask[
                :, :, :, start:end
            ].masked_fill(padding_mask, min_dtype)
    return causal_mask


def update_causal_mask(
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    min_dtype: float = -3.4028234663852886e38,
    sequence_length: int = 2048,
    kv_cache_len: int = 2048,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    padding_side: str = "left",
) -> torch.Tensor:
    """
    Update and generate a 4D causal mask based on a 2D attention mask.

    Args:
        attention_mask: Input attention mask tensor.
        input_tensor: Model input tensor used to obtain the batch_size.
        cache_position: Tensor indicating token positions.
        min_dtype: The minimum value for the given dtype.
        sequence_length: The sequence length.
        kv_cache_len: The KV cache length.
        dtype: The data type.
        device: The device information.

    Returns:
        Updated 4D causal mask.
    """
    target_length = kv_cache_len
    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        min_dtype=min_dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
        padding_side=padding_side,
    )
    return causal_mask


def split_valid_lens(valid_lens, chunk_size, padding_side):
    if valid_lens <= 0 or chunk_size <= 0:
        return []

    if padding_side == "right":
        num_full_chunks = valid_lens // chunk_size
        remainder = valid_lens % chunk_size

        result = [chunk_size] * num_full_chunks
        if remainder > 0:
            result.append(remainder)
        return result

    elif padding_side == "left":
        if valid_lens <= chunk_size:
            return [valid_lens]

        num_chunks = (valid_lens + chunk_size - 1) // chunk_size
        remainder = valid_lens % chunk_size

        if remainder == 0:
            return [chunk_size] * num_chunks
        else:
            return [remainder] + [chunk_size] * (num_chunks - 1)

    else:
        raise ValueError("padding_side must be either 'right' or 'left'")


def get_mask(
    chunk_size,
    kv_cache_len,
    valid_lens,
    padding_side,
    mask_value,
    data_type,
):
    causal_mask_chunks = []

    valid_split_lens = split_valid_lens(valid_lens, chunk_size, padding_side)

    if "right" == padding_side:
        for i in range(len(valid_split_lens)):
            item = torch.full((chunk_size, kv_cache_len), mask_value, dtype=data_type)
            begin_idx = max(kv_cache_len - (i + 1) * chunk_size, 0)
            for idx in range(valid_split_lens[i]):
                end_idx = kv_cache_len - chunk_size + idx + 1
                item[idx, begin_idx:end_idx] = 0
            causal_mask_chunks.append(item)
    elif "left" == padding_side:
        for i in range(len(valid_split_lens)):
            item = torch.full((chunk_size, kv_cache_len), mask_value, dtype=data_type)
            begin_idx = max(kv_cache_len - sum(valid_split_lens[: i + 1]), 0)
            for idx in range(valid_split_lens[i]):
                item[
                    chunk_size - valid_split_lens[i] + idx,
                    begin_idx : begin_idx + sum(valid_split_lens[:i]) + idx + 1,
                ] = 0
            causal_mask_chunks.append(item)

    return causal_mask_chunks


def get_pos_ids(
    chunk_size,
    valid_lens,
    padding_side,
    pos_mask_value,
):
    position_ids_chunks = []
    num_loops = (valid_lens + chunk_size - 1) // chunk_size
    value_ids = torch.arange(valid_lens)
    pos_item = torch.full((chunk_size * num_loops,), pos_mask_value, dtype=torch.int64)

    if padding_side == "right":
        pos_item[:valid_lens] = value_ids
    elif padding_side == "left":
        pos_item[-valid_lens:] = value_ids
    position_ids_chunks = torch.chunk(pos_item, num_loops)

    return position_ids_chunks


class CalibrationDataPreparer:
    def __init__(
        self,
        model_type: str,
        model_dir: str,
        seq_len: int,
        kv_cache_len: int,
        transpose_cache: bool = True,
        device: str = "cpu",
        mask_value: float = -512,
        pos_mask_value: int = 1,
        data_type=torch.float16,
        padding_side: str = "left",
    ):
        """
        初始化阶段加载 tokenizer 及模型配置

        参数:
          model_dir: 模型及配置所在的目录
          seq_len: 分块的序列长度
          kv_cache_len: kv 缓存长度
          transpose_cache: 是否转置缓存
          device: 设备，默认为 "cpu"
          mask_value: 注意力掩码的填充值，默认为 -512
          pos_mask_value: 位置掩码的填充值，默认为 1
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.tokenizer.padding_side = padding_side
        self.seq_len = seq_len
        self.kv_cache_len = kv_cache_len
        self.transpose_cache = transpose_cache
        self.full_logits = True
        self.vaild_idx = 0
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "llm_config" in config.keys():
            config_dict = config["llm_config"]
        else:
            config_dict = config
        self.hidden_size = config_dict["hidden_size"]
        self.num_attention_heads = config_dict["num_attention_heads"]
        self.head_dim = config_dict.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.num_key_value_heads = config_dict["num_key_value_heads"]
        self.block_num = config_dict["num_hidden_layers"]
        self.mask_value = mask_value
        self.pos_mask_value = pos_mask_value
        self.data_type = data_type
        self.padding_side = padding_side
        if model_type:
            self.model_type = model_type
        else:
            raise ValueError("model type empty !!!!!!")

    def set_full_logits(self, full_logits: bool):
        self.full_logits = full_logits

    def prepare_inputs(self, prompt: str):
        """
        处理单个 prompt，返回 input_chunks、causal_mask_chunks、
        position_ids_chunks、past_key_values_list

        参数:
          prompt: 待处理的文本 prompt
        """
        # 对 prompt 进行初步 tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids_valid = inputs.input_ids.to(self.device)
        raw_inputs_len = input_ids_valid.shape[-1]
        if not self.full_logits:
            self.vaild_idx = self.seq_len - 1
        else:
            if raw_inputs_len < self.seq_len:
                self.vaild_idx = self.seq_len - raw_inputs_len
            else:
                self.vaild_idx = 0

        if raw_inputs_len > self.kv_cache_len:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.kv_cache_len - 10,
            )
            input_ids_valid = inputs.input_ids.to(self.device)

        n = input_ids_valid.shape[-1]
        inputs_pad_len, _ = pad_to_multiple(n, self.seq_len)

        chunk_size = self.seq_len
        max_length = inputs_pad_len

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = inputs.input_ids.to(self.device)
        # 初始化 past key/value（KV）缓存数据
        init_kv_data = torch.zeros(
            [self.num_key_value_heads, self.kv_cache_len, self.head_dim],
            dtype=torch.float32,
        ).to(self.device)

        if self.transpose_cache:
            init_kv_data = init_kv_data.transpose(0, 1)

        past_key_values_list: List[Any] = [init_kv_data] * self.block_num + [
            init_kv_data
        ] * self.block_num

        if "internvl" in self.model_type:
            input_chunks = input_ids.split(chunk_size, dim=-1)

            causal_mask_chunks = get_mask(
                chunk_size,
                self.kv_cache_len,
                n,
                self.padding_side,
                self.mask_value,
                self.data_type,
            )
            position_ids_chunks = get_pos_ids(
                chunk_size,
                n,
                self.padding_side,
                self.pos_mask_value,
            )

            # Ensure calibration outputs are on the target device
            causal_mask_chunks = [item.to(self.device) for item in causal_mask_chunks]
            position_ids_chunks = [item.to(self.device) for item in position_ids_chunks]

            if len(causal_mask_chunks) != len(position_ids_chunks) or len(
                input_chunks
            ) != len(position_ids_chunks):
                raise ValueError("mask/position lens error!!!!!!")

            return (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                past_key_values_list,
            )
        else:
            # deepseek internlm qwen
            attention_mask = inputs.attention_mask.to(self.device)
            position_ids = get_position_ids(
                attention_mask, pos_mask_value=self.pos_mask_value
            ).to(self.device)

            # 按 chunk_size 拆分 input_ids 和 position_ids
            input_chunks = input_ids.split(chunk_size, dim=-1)
            position_ids_chunks = position_ids[0].split(chunk_size, dim=-1)

            valid_seq_len = int(torch.sum(attention_mask).item())
            padding_side = self.tokenizer.padding_side
            total_chunks = input_ids.shape[-1] // chunk_size
            update_causal_mask_chunks = []
            for chunk_index in range(total_chunks):
                chunk_mask = create_chunk_mask(
                    chunk_index=chunk_index,
                    valid_seq_len=valid_seq_len,
                    chunk_size=chunk_size,
                    kv_cache_len=self.kv_cache_len,
                    mask_value=self.mask_value,
                    padding_side=padding_side,
                    device=str(self.device),
                )
                update_causal_mask_chunks.append(chunk_mask.to(self.device))

            return (
                input_chunks,
                update_causal_mask_chunks,
                position_ids_chunks,
                past_key_values_list,
            )
