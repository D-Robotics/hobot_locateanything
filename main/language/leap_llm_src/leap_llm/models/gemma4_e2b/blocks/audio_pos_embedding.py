import math

import torch

from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4AudioConfig
from leap_llm.nn.utils import Module


class Gemma4AudioRelPositionalEncoding(Module):
    """Sinusoidal relative positional encoding for the audio encoder.

    Produces position embeddings of shape [1, 2*context_size - 1, hidden_size] with
    concatenated [sin..., cos...] layout matching the original Gemma4 convention.
    """

    inv_timescales: torch.Tensor

    def __init__(
        self,
        config: Gemma4AudioConfig,
        device="cpu",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.device = device
        self.context_size = (
            config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        )
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
        self.register_buffer("inv_timescales", inv_timescales.unsqueeze(0).unsqueeze(0), persistent=False)
        pos_emb = self._set_cos_sin_cache()
        self.register_buffer("audio_pe", pos_emb, persistent=False)

    @torch.no_grad()
    def _set_cos_sin_cache(self):
        position_ids = torch.arange(12, -1, -1, device=self.device)
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales.to(device=self.device)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.float()
