import torch
import torch.nn as nn
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4AudioConfig
from leap_llm.nn.modules import (
    Conv2d,
    DynamicQuantLinear,
    LayerNorm,
)
from leap_llm.nn.utils import Module


class Gemma4AudioSubSampleConvProjectionLayer(Module):
    def __init__(self, in_channels, out_channels, norm_eps):
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.norm = LayerNorm(
            out_channels,
            eps=norm_eps,
            bias=False,
        )
        self.act = nn.ReLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ):
        hidden_states = hidden_states * mask
        print(f"[before conv] hidden_states: {hidden_states.shape}")
        hidden_states = self.conv(hidden_states)
        print(f"[after conv] hidden_states: {hidden_states.shape}")
        hidden_states = hidden_states.permute(0, 2, 3, 1)  # transpose C_out to norm dim
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 1, 2).contiguous()
        hidden_states = self.act(hidden_states)

        return hidden_states

    def build(self, hidden_states, mask):
        bs, c, h, w = hidden_states.type.shape
        hidden_states = leap.mul(hidden_states, mask)
        hidden_states = leap.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)
        print(f"[before conv] hidden_states: {hidden_states.type.shape}")
        hidden_states = self.conv(hidden_states)
        print(f"[after conv] hidden_states: {hidden_states.type.shape}")
        hidden_states = leap.cast_type(hidden_states, output_type=leap.float16)
        hidden_states = self.norm(hidden_states)
        hidden_states = leap.transpose(hidden_states, (0, 3, 1, 2))
        hidden_states = leap.relu(hidden_states)

        return hidden_states


class Gemma4AudioConvProjection(Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
    ):
        super().__init__()
        self.config = config
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
        )
        proj_input_dim = (config.subsampling_conv_channels[0] // 4) * config.subsampling_conv_channels[1]
        self.input_proj_linear = DynamicQuantLinear(proj_input_dim, config.hidden_size, bias=False)

    def forward(
        self,
        input_features,
        conv_layer_mask_0,
        conv_layer_mask_1,
    ):
        """Two stacked sub-sample convs + a linear projection to ``hidden_size``.

        Args:
            input_features (torch.Tensor): Padded mel-spectrogram
                features. Shape: ``(batch_size, 1, seq_len, feature_dim)``.
                For the default config: ``(1, 1, 3000, 128)``.
            conv_layer_mask_0 (torch.Tensor): Validity mask for
                layer 0, time-axis strided by 2.
                Shape: ``(batch_size, 1, seq_len, 1)`` (input
                resolution). For ``seq_len = 3000``:
                ``(1, 1, 3000, 1)``.
            conv_layer_mask_1 (torch.Tensor): Validity mask for
                layer 1, time-axis strided by 4 (so it lines up with
                the output of layer 0's stride-2 conv).
                Shape: ``(batch_size, 1, seq_len // 2, 1)``.
                For ``seq_len = 3000``: ``(1, 1, 1500, 1)``.

        Returns:
            torch.Tensor: Sub-sampled + projected features. Shape:
                ``(batch_size, seq_len_after, hidden_size)``. For
                ``seq_len = 3000`` with two stride-2 layers:
                ``(batch_size, 750, 1024)``.
        """
        hidden_states = self.layer0(input_features, conv_layer_mask_0)
        print(f"after layer0 conv hidden_states.shape: {hidden_states.shape}")
        hidden_states = self.layer1(hidden_states, conv_layer_mask_1)

        batch_size, _, seq_len, _ = hidden_states.shape
        print(f"after 2-layer conv2d: hidden_states.shape: {hidden_states.shape}")
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
        hidden_states = self.input_proj_linear(hidden_states)
        return hidden_states

    def build(
        self,
        input_features,
        conv_layer_mask_0,
        conv_layer_mask_1,
    ):
        """Leap export path. See :meth:`forward` for shapes.

        Args:
            input_features: ``(1, 1, 3000, 128)``.
            conv_layer_mask_0: ``(1, 1, 3000, 1)``.
            conv_layer_mask_1: ``(1, 1, 1500, 1)``.
        """
        hidden_states = self.layer0(input_features, conv_layer_mask_0)
        hidden_states = self.layer1(hidden_states, conv_layer_mask_1)

        batch_size, _, seq_len, _ = hidden_states.type.shape

        hidden_states = leap.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = leap.reshape(hidden_states, (batch_size, seq_len, -1))
        hidden_states = self.input_proj_linear(hidden_states)

        return hidden_states
