import torch
from horizon_plugin_pytorch.nn import Matmul
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from torch import nn

from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from horizon_plugin_pytorch.nn.flash_correction import flash_correction
    from horizon_plugin_pytorch.nn.flash_softmax import FlashSoftmax

    _FLASH_ATTN_IMPL_AVAILABLE = True
except ImportError:
    logger.warning(
        "Flash attention implementation not available. HzFlashAttention will not be defined. "
        "To use flash attention, please upgrade horizon_plugin_pytorch."
    )
    flash_correction = None
    FlashSoftmax = None
    _FLASH_ATTN_IMPL_AVAILABLE = False


_FLASH_ATTN_LOGGED_ONCE = False


def log_flash_attention_enabled_once(block_size: int) -> None:
    global _FLASH_ATTN_LOGGED_ONCE
    if _FLASH_ATTN_LOGGED_ONCE:
        return
    logger.info("Using flash attention with block size: %s", block_size)
    _FLASH_ATTN_LOGGED_ONCE = True


class HzFlashAttention(nn.Module):
    """Block-wise flash attention core using FlashSoftmax + flash_correction.

    This module is parameter-free and can be reused by different model families.
    Inputs are expected in shape [..., seq_len, head_dim].
    """

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size
        log_flash_attention_enabled_once(block_size)
        self.flash_softmax = FlashSoftmax(dim=-1)
        self.qk_matmul = Matmul()
        self.sv_matmul = Matmul()
        self.score_scale_mul = FloatFunctional()
        self.score_mask_add = FloatFunctional()

    def _split_blocks(self, seq_len: int):
        return [(start, min(seq_len, start + self.block_size)) for start in range(0, seq_len, self.block_size)]

    def _slice_attention_mask(self, attention_mask: torch.Tensor, start: int, end: int):
        if attention_mask is None:
            return None
        return attention_mask[..., start:end]

    def _prepare_gqa(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        """Reshape Q for GQA by merging groups into the seq dimension.

        Q:    [batch, num_heads, q_len, d] -> [batch, kv_heads, groups*q_len, d]
        K/V:  [batch, kv_heads, kv_len, d]   (unchanged)

        Returns (query, key, value, num_groups) where num_groups=0 means
        no GQA reshape was applied.
        """
        assert (
            query_states.dim() == 4 and key_states.dim() == 4 and value_states.dim() == 4
        ), "Expected query, key, value to be 4D tensors [batch, num_heads, seq_len, head_dim]"
        assert (
            query_states.shape[1] % key_states.shape[1] == 0
        ), "Number of query heads must be divisible by number of key/value heads for GQA"

        q_heads = query_states.shape[-3]
        kv_heads = key_states.shape[-3]
        if q_heads == kv_heads or kv_heads == 1:
            return query_states, key_states, value_states, 0

        num_groups = q_heads // kv_heads
        batch = query_states.shape[0]
        head_dim = query_states.shape[-1]

        query_states = query_states.reshape(batch, kv_heads, num_groups * query_states.shape[-2], head_dim)
        return query_states, key_states, value_states, num_groups

    def _add_gqa_mask(self, scores, mask, num_groups):
        """Add attention mask to scores, reshaping to per-head view to avoid
        repeating the mask.

        scores: [bsz, kv_heads, groups*q_len, block_kv_len]
          -> reshape to [bsz, num_heads, q_len, block_kv_len]
          -> add mask    [bsz, 1,         q_len, block_kv_len]  (broadcasts)
          -> keep expanded view for flash_softmax
        """
        if mask is None:
            return scores, None
        if num_groups == 0:
            return self.score_mask_add.add(scores, mask), None
        shape = scores.shape
        scores = scores.reshape(shape[0], shape[1] * num_groups, -1, shape[-1])
        scores = self.score_mask_add.add(scores, mask)
        return scores, shape

    def _restore_gqa_shape(self, tensor: torch.Tensor, original_shape):
        if original_shape is None:
            return tensor
        return tensor.reshape(original_shape[:-1] + (tensor.shape[-1],))

    def _flash_softmax_with_gqa_mask(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        num_groups: int,
    ):
        scores, gqa_shape = self._add_gqa_mask(scores, mask, num_groups)
        probs, mi, li = self.flash_softmax(scores)
        probs = self._restore_gqa_shape(probs, gqa_shape)
        mi = self._restore_gqa_shape(mi, gqa_shape)
        li = self._restore_gqa_shape(li, gqa_shape)
        return probs, mi, li

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        query_states, key_states, value_states, num_groups = self._prepare_gqa(query_states, key_states, value_states)

        kv_len = key_states.shape[-2]
        blocks = self._split_blocks(kv_len)
        if not blocks:
            raise ValueError("key_states has empty sequence length")

        s0, e0 = blocks[0]
        k0 = key_states[..., s0:e0, :]
        scores_0 = self.score_scale_mul.mul_scalar(
            self.qk_matmul(query_states, k0.transpose(-2, -1)),
            scale,
        )
        mask_0 = self._slice_attention_mask(attention_mask, s0, e0)
        probs_0, mi, li = self._flash_softmax_with_gqa_mask(scores_0, mask_0, num_groups)
        attn_output = self.sv_matmul(probs_0, value_states[..., s0:e0, :])

        for start, end in blocks[1:]:
            k_block = key_states[..., start:end, :]
            scores_i = self.score_scale_mul.mul_scalar(
                self.qk_matmul(query_states, k_block.transpose(-2, -1)),
                scale,
            )
            mask_i = self._slice_attention_mask(attention_mask, start, end)
            probs_i, mi_i, li_i = self._flash_softmax_with_gqa_mask(scores_i, mask_i, num_groups)
            attn_output_i = self.sv_matmul(
                probs_i,
                value_states[..., start:end, :],
            )
            attn_output, mi, li = flash_correction(
                attn_output_i,
                attn_output,
                mi_i,
                li_i,
                mi,
                li,
            )

        if num_groups > 0:
            batch = attn_output.shape[0]
            kv_heads = attn_output.shape[1]
            head_dim = attn_output.shape[-1]
            attn_output = attn_output.reshape(batch, kv_heads * num_groups, -1, head_dim)

        return attn_output


if not _FLASH_ATTN_IMPL_AVAILABLE:
    del HzFlashAttention
