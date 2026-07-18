import torch
import torch.nn.functional as F
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant
from .conv import Conv2d, Conv3d


class Qwen2_5_VisionPatchEmbed(Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        use_conv2d=False,
        quant_output=True,
        use_plugin=False,
    ) -> None:
        super().__init__()
        self.use_plugin = use_plugin
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_conv2d = use_conv2d

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            quant_output=quant_output,
        )

        if self.use_conv2d is True:
            if not self.use_plugin:
                self.proj_2d = Conv2d(
                    in_channels,
                    embed_dim,
                    kernel_size=[patch_size, patch_size],
                    stride=[patch_size, patch_size],
                    bias=False,
                    quant_output=quant_output,
                )
            else:
                self.proj_2d = nn.Conv2d(
                    in_channels,
                    embed_dim,
                    kernel_size=[patch_size, patch_size],
                    stride=[patch_size, patch_size],
                    bias=False,
                )

    def build(self, hidden_states):
        if self.use_conv2d is False:
            hidden_states = leap.reshape(
                hidden_states,
                [
                    -1,
                    self.in_channels,
                    self.temporal_patch_size,
                    self.patch_size,
                    self.patch_size,
                ],
            )
            hidden_states = leap.transpose(hidden_states, [0, 2, 3, 4, 1])
            hidden_states = self.proj(hidden_states)
            dtype = hidden_states.type.element_type
        else:
            hidden_states = leap.reshape(
                hidden_states,
                [
                    -1,
                    self.in_channels,
                    self.patch_size,
                    self.patch_size,
                ],
            )
            hidden_states = leap.transpose(hidden_states, [0, 2, 3, 1])
            dtype = hidden_states.type.element_type
            hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)
            hidden_states = self.proj_2d(hidden_states)
        hidden_states = leap.cast_type(hidden_states, output_type=dtype)
        hidden_states = leap.reshape(hidden_states, [-1, self.embed_dim])
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        if self.use_conv2d is False:
            hidden_states = hidden_states.view(
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
            hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
                -1, self.embed_dim
            )
        else:
            hidden_states = hidden_states.view(
                -1,
                self.in_channels,
                self.patch_size,
                self.patch_size,
            )
            hidden_states = self.proj_2d(hidden_states.to(dtype=target_dtype)).view(
                -1, self.embed_dim
            )
        return hidden_states


class FakeQuantPatchEmbedding(Module):
    def __init__(self, dim: int, num_channels: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.weight = nn.Parameter(
            torch.empty(dim, patch_size, patch_size, num_channels)
        )
        self.bias = nn.Parameter(torch.empty(dim))
        self.absmax_weight = None
        self.quant_bits = 8  # quantize to 8 bits
        self.x_fake_quant = ConstFakeQuant(self.quant_bits)

    def build(self, x):
        x_cast = leap.cast_type(x, output_type=leap.float32)
        x_quant = self.x_fake_quant(x_cast)
        weight_min = (-self.absmax_weight).tolist()
        weight_max = self.absmax_weight.tolist()

        # weight 是 CoutxCin 的权重矩阵
        weight_quant = leap.const_fake_quant(
            self.weight.data,
            weight_min,
            weight_max,
            self.quant_bits,
            True,
            axis=0,
        )

        conv_res = leap.conv2d(
            input=x_quant,
            weight=weight_quant,
            bias=self.bias.data,
            stride=(self.patch_size, self.patch_size),
        )
        return leap.cast_type(conv_res, output_type=x.type.element_type)

    def forward(self, x: torch.Tensor):
        x = self.x_fake_quant(x)
        if self.absmax_weight is None:
            last_dim = self.patch_size * self.patch_size * self.num_channels
            weight = torch.reshape(self.weight.data, [self.dim, last_dim])
            per_channel_max, _ = torch.max(weight.abs(), dim=1)
            self.absmax_weight = per_channel_max

        weight_t = self.weight.data.permute(0, 3, 1, 2).contiguous()
        inputs_embeds = F.conv2d(
            input=x,
            weight=weight_t,
            bias=self.bias.data,
            stride=(self.patch_size, self.patch_size),
        )
        return inputs_embeds


class VisionEmbeddings(Module):
    def __init__(self, dim: int, num_channels: int, patch_size: int, image_size: int):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.empty(1, 1, dim))
        self.position_embedding = nn.Parameter(torch.empty(1, 1 + dim, dim))
        self.patch_embedding = FakeQuantPatchEmbedding(dim, num_channels, patch_size)
        self.patch_size = patch_size
        self.image_size = image_size

    def build(self, x):
        hwc_img_pixel = leap.transpose(x, [0, 2, 3, 1])
        self.patch_embedding.to("cpu", dtype=torch.float32)
        patch_embeds = self.patch_embedding(hwc_img_pixel)
        batch_size, height, width, channel = patch_embeds.type.shape
        # 变成 (B, H*W, C) 的形式
        patch_embeds = leap.reshape(patch_embeds, (batch_size, height * width, channel))
        class_embeds = self.class_embedding.data
        embeddings = leap.concat([class_embeds, patch_embeds], dim=1)
        pos_embed = self.position_embedding.data
        return leap.add(embeddings, pos_embed)

    def get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(
                1,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, x: torch.Tensor):
        # x = x.transpose(1, 3)
        patch_embeds = self.patch_embedding(x)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self.get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        return embeddings + position_embedding
