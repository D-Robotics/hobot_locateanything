"""SmolVLM multimodal connector (vision tokens -> text hidden size via pixel shuffle)."""

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Model, Module


class SmolVLMSimpleMLP(Module):
    """Mirrors transformers SmolVLMSimpleMLP: a single Linear (named proj)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = DynamicQuantLinear(in_features, out_features, bias=False)

    def build(self, x):
        return self.proj(x)

    def forward(self, x):
        return self.proj(x)


class SmolVLMConnector(Model):
    """
    PixelShuffle connector matching transformers SmolVLMConnector.

    Pipeline (for scale_factor=4, 512×512 image, patch=16):
        [B, 1024, 768] → pixel_shuffle 4× → [B, 64, 12288] → linear → [B, 64, 960]

    Weight key in checkpoint: connector.modality_projection.proj.{weight,bias}
    """

    def __init__(self, vision_hidden_size: int, text_hidden_size: int, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        self.modality_projection = SmolVLMSimpleMLP(
            vision_hidden_size * scale_factor * scale_factor,
            text_hidden_size,
        )

    # ------------------------------------------------------------------
    # LEAP (BPU compile) path
    # ------------------------------------------------------------------
    def build(self, x):
        S = self.scale_factor
        # x shape tag: [1, N, C]  where N=H*W, H=W=sqrt(N)
        N = x.type.shape[1]
        C = x.type.shape[2]
        H = W = int(N ** 0.5)

        x = leap.reshape(x, [1, H // S, S, W // S, S, C])
        x = leap.transpose(x, [0, 1, 3, 2, 4, 5])
        x = leap.reshape(x, [1, (H // S) * (W // S), C * S * S])
        return self.modality_projection(x)

    # ------------------------------------------------------------------
    # PyTorch (CPU calibration / forward) path
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.scale_factor
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        x = x.reshape(B, H // S, S, W // S, S, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, (H // S) * (W // S), C * S * S)
        return self.modality_projection(x)
