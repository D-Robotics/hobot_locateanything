import torch
import torch.nn as nn
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.blocks.audio_attention import Gemma4AudioAttention
from leap_llm.models.gemma4_e2b.blocks.audio_mlp import Gemma4AudioFeedForward
from leap_llm.models.gemma4_e2b.blocks.linear import Gemma4ClippableLinear
from leap_llm.models.gemma4_e2b.blocks.rmsnorm import Gemma4RMSNorm
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4AudioConfig
from leap_llm.nn.modules import Conv1d
from leap_llm.nn.utils import Module


class Glu(Module):
    """GLU activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim=-1) -> torch.Tensor:
        return nn.functional.glu(x, dim=dim)

    def build(self, x):
        s0, s1, dim = x.type.shape
        assert dim % 2 == 0, "last dimension shall be evenly divided for glu"
        a = leap.slice(
            x,
            [0, 0, 0],
            [s0, s1, dim // 2],
            [1, 1, 1],
        )
        b = leap.slice(
            x,
            [0, 0, dim // 2],
            [s0, s1, dim],
            [1, 1, 1],
        )
        b_act = leap.sigmoid(b)
        hidden_states = leap.mul(a, b_act)
        return hidden_states


class Gemma4AudioCausalConv1d(Conv1d):
    """Causal 1D convolution for audio."""

    @property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)

    def build(self, x):
        print(f"x.type.shape: {x.type.shape}")
        s0, s1, s2 = x.type.shape
        pad_zeros = torch.zeros((s0, self.left_pad, s2), dtype=torch.float16)
        x = leap.concat([pad_zeros, x], dim=-2)
        print(f"x.type.shape: {x.type.shape}")
        x = leap.cast_type(x, output_type=leap.float32)
        x = super().build(x)
        x = leap.cast_type(x, output_type=leap.float16)
        return x


class Gemma4AudioLightConv1d(Module):
    """Light convolution with GLU activation."""

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config

        self.linear_start = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 2)
        self.linear_end = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )

        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.glu = Glu()
        self.act_fn = F.silu

        self.gradient_clipping = config.gradient_clipping
        self.clipping_absmax = min(
            self.gradient_clipping,
            torch.finfo(self.linear_start.linear.weight.dtype).max - 1000,  # FIXME: for avoiding quantization overflow
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = self.glu(hidden_states, dim=-1)

        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = torch.clamp(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.conv_norm(hidden_states)

        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states += residual
        return hidden_states

    def build(self, hidden_states):
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = self.glu(hidden_states)
        print(f"hidden_states: {hidden_states.type.shape}")
        # hidden_states = leap.transpose(hidden_states, (0, 2, 1))
        hidden_states = self.depthwise_conv1d(hidden_states)
        # hidden_states = leap.transpose(hidden_states, (0, 2, 1))

        hidden_states = leap.clip(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.conv_norm(hidden_states)

        hidden_states = leap.swish(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states = leap.add(hidden_states, residual)

        return hidden_states


class Gemma4AudioLayer(Module):
    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.config = config

        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config, layer_idx)
        self.lconv1d = Gemma4AudioLightConv1d(config)

        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_out = Gemma4RMSNorm(config.hidden_size)

        self.gradient_clipping = config.gradient_clipping
        self.clipping_absmax = min(
            self.gradient_clipping,
            torch.finfo(self.norm_pre_attn.weight.dtype).max - 1000,  # FIXME: for avoiding quantization overflow
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor | None,
        position_embeddings: torch.Tensor,
    ):
        """One Conformer-style audio encoder layer: FFN -> MHSA -> LConv1D -> FFN.

        Args:
            hidden_states (torch.Tensor): Audio hidden states. Shape:
                ``(batch_size, seq_len, hidden_size)``, e.g.
                ``(1, 750, 1024)`` after the conv-subsample projection.
            attention_mask (torch.BoolTensor | None): 5D blocked
                attention mask (see
                :func:`_convert_4d_mask_to_blocked_5d`). Shape:
                ``(batch_size, num_blocks, chunk_size, context_size)``.
                For ``seq_len = 750``, ``chunk_size = 12``:
                ``(batch_size, 63, 12, 24)`` (or similar). ``True``
                marks positions that are allowed to attend.
            position_embeddings (torch.Tensor): Sinusoidal relative
                position embeddings for the audio attention. Shape:
                ``(1, 2 * context_size - 1, hidden_size)`` (sin first
                half, cos second half — see
                :class:`Gemma4AudioRelPositionalEncoding`).

        Returns:
            torch.Tensor: Layer output. Same shape as ``hidden_states``.
        """
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = torch.clamp(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.norm_pre_attn(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = torch.clamp(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states += residual

        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states

    def build(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
    ):
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = leap.clip(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.norm_pre_attn(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = leap.clip(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states = leap.add(hidden_states, residual)

        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)

        hidden_states = leap.clip(hidden_states, -self.clipping_absmax, self.clipping_absmax)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states
