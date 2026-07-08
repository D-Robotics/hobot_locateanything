import torch
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4VisionConfig
from leap_llm.nn.modules import (
    DynamicQuantMatmul,
)
from leap_llm.nn.utils import Module


class Gemma4VisionPooler(Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5
        self.avg_pool_weight = self._compute_avg_pool_wt()
        self.avg_pool_weight *= self.root_hidden_size  # NOTE: merge scaling into weight
        self.wt_matmul = DynamicQuantMatmul()

    def _compute_avg_pool_wt(
        self,
        k: int = 3,
    ):
        patch_height, patch_width = self.config.image_thw
        k_squared = k**2
        patch_grid = torch.meshgrid(
            torch.arange(patch_width),
            torch.arange(patch_height),
            indexing="xy",
        )
        stacked_grid = torch.stack(patch_grid, dim=-1)
        position_ids = stacked_grid.reshape(1, -1, 2)
        length = position_ids.shape[1] // k_squared
        max_x = position_ids[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(position_ids, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = torch.nn.functional.one_hot(kernel_idxs.long(), length).float() / k_squared
        weights = weights.transpose(1, 2)
        return weights

    def forward(self, hidden_states):
        """Average-pool encoder hidden states by ``k x k`` patch grid.

        Pools the 48x48 encoder output down to a ``(48/k) x (48/k)`` grid
        using an ``avg_pool`` with the scaling factor merged into the
        weight matrix. With the leap default ``k=3`` this is 16x16 ->
        256 tokens, the same as ``(48*48) / 3**2``.

        Args:
            hidden_states (torch.Tensor): Encoder output for the fixed
                patch grid. Shape:
                ``(batch_size, num_patches, hidden_size)``, e.g.
                ``(batch_size, 2304, 768)`` for the 768x768 path.

        Returns:
            torch.Tensor: Pooled encoder hidden states.
                Shape: ``(batch_size, num_patches // k**2, hidden_size)``,
                e.g. ``(batch_size, 256, 768)`` for the 768x768 path with
                ``k=3``.
        """
        wt = self.avg_pool_weight.to(hidden_states)
        hidden_states = wt @ hidden_states
        return hidden_states

    def build(self, hidden_states):
        """Leap export path. See :meth:`forward` for shapes.

        Args:
            hidden_states: Shape ``(1, 2304, 768)`` for the 768x768 path.
                Output shape: ``(1, 256, 768)``.
        """
        wt = self.avg_pool_weight.to(device="cpu", dtype=torch.float16).contiguous()
        hidden_states = leap.transpose(hidden_states, (0, 2, 1))  # for dq_matmul
        hidden_states = self.wt_matmul(wt, hidden_states)
        return hidden_states
