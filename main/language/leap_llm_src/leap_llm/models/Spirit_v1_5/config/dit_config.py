import torch
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig

class BaseDiTConfig(PretrainedConfig):
    model_type = "BaseDiT"

    def __init__(
        self,
        num_attention_heads,
        attention_head_dim,
        num_layers: int = 12,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_vlm_last_embd = 1
        self.param_dict = {
            "num_attention_heads": num_attention_heads,
            "attention_head_dim": attention_head_dim,
            "num_layers": num_layers,
            "attention_bias": attention_bias,
            "activation_fn": activation_fn,
            "num_embeds_ada_norm": num_embeds_ada_norm,
            "upcast_attention": upcast_attention,
            "norm_type": norm_type,
            "norm_elementwise_affine": norm_elementwise_affine,
            "norm_eps": norm_eps,
            "max_num_positional_embeddings": max_num_positional_embeddings,
            "compute_dtype": compute_dtype,
            "positional_embeddings": positional_embeddings,
            "interleave_self_attention": interleave_self_attention,
            "cross_attention_dim": cross_attention_dim,
        }


class DiTConfig:
    num_attention_heads: int = 8
    attention_head_dim: int = 64
    num_layers: int = 12
    attention_bias: bool = True
    activation_fn: str = "gelu-approximate"
    num_embeds_ada_norm: int | None = 1000
    upcast_attention: bool = False
    norm_type: str = "ada_norm"
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-5
    max_num_positional_embeddings: int = 512
    compute_dtype: torch.dtype = torch.float32
    positional_embeddings: str | None = "sinusoidal"
    interleave_self_attention: bool = False
    cross_attention_dim: int | None = None
    max_state_dim: int = 32
    max_action_dim: int = 32
    dit_hidden_size: int = 1536
    n_action_steps: int = 50