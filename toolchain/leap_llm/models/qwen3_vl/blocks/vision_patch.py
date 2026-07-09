import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules.activation import FakeQuantGELU
from leap_llm.nn.modules.layer_norm import LayerNorm
from leap_llm.nn.modules.linear import DynamicQuantLinear
from leap_llm.nn.utils import Module


class Qwen3VLVisionPatchEmbed(Module):
    """
    (patch_embed): Qwen3VLVisionPatchEmbed(
    (proj): Conv3d(3, 1024, kernel_size=(2, 16, 16), stride=(2, 16, 16))
    )
    """

    def __init__(self, config, use_plugin: bool = False):
        super().__init__()
        self.use_plugin = use_plugin
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        # kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        # self.proj = Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size,
        #                    stride=kernel_size, bias=True)
        flatten_size = (
            self.temporal_patch_size
            * self.patch_size
            * self.patch_size
            * self.in_channels
        )
        self.proj = DynamicQuantLinear(
            in_features=flatten_size, out_features=self.embed_dim
        )

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
        return hidden_states

    def build(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return hidden_states


class Qwen3VLVisionPatchMerger(Module):
    """
    (merger): Qwen3VLVisionPatchMerger(
    (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
    (linear_fc1): Linear(in_features=4096, out_features=4096, bias=True)
    (act_fn): GELU(approximate='none') NOTE: no approximation function
    (linear_fc2): Linear(in_features=4096, out_features=2560, bias=True)
    )
    """

    def __init__(
        self,
        config,
        use_postshuffle_norm=False,
        use_plugin: bool = False,
    ):
        super().__init__()
        self.use_plugin = use_plugin
        # NOTE: hidden_size multipled by factor of merge_size**2
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            self.norm = LayerNorm(self.hidden_size, eps=1e-6)
        else:
            self.norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = DynamicQuantLinear(self.hidden_size, self.hidden_size)
        self.act_fn = FakeQuantGELU()
        self.linear_fc2 = DynamicQuantLinear(self.hidden_size, config.out_hidden_size)

    def build(self, x):
        bs = x.type.shape[0]
        if self.use_postshuffle_norm:
            x = leap.reshape(x, [bs, -1, self.hidden_size])
            x = self.norm(x)
        else:
            x = self.norm(x)
        x = leap.reshape(x, [bs, -1, self.hidden_size])
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

    def forward(self, x: torch.Tensor):
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x)
        x = x.view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x
