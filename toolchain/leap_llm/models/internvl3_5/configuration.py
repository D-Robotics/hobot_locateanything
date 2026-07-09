from dataclasses import dataclass


@dataclass
class InternVL3_5VisionConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    layer_norm_eps: float = 1e-06
    image_size: int = 448
    patch_size: int = 14
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    qkv_bias: bool = True
    initializer_factor: float = 1.0


@dataclass
class InternVL3_5LLMConfig:
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    pad_token_id: int = 151643
    num_key_value_heads: int = 8
    attention_bias: bool = False
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000
    head_dim: int = 128
    max_prefill_tokens: int = 3072  # 1024
    max_cache_tokens: int = 4096
    prefill_seq_len: int = 256
    decode_seq_len: int = 1
    use_fastv: bool = False
    fastv_k: int = 3
    fastv_r: float = 0.5
    image_token_start_index: int = 41
    image_token_length: int = 512
    min_value: float = -1e3
    fastv_max_cache_tokens: int = 2048


@dataclass
class InternVL3_5Config:
    vision_config: InternVL3_5VisionConfig = None
    llm_config: InternVL3_5LLMConfig = None
    downsample_ratio: float = 0.5
