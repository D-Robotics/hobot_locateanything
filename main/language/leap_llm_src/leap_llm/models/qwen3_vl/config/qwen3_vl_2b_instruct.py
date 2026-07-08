from dataclasses import dataclass


@dataclass
class Qwen3VLVisionConfig:
    deepstack_visual_indexes = [5, 11, 17]
    depth = 24
    hidden_act = "gelu_pytorch_tanh"
    hidden_size = 1024
    in_channels = 3
    initializer_range = 0.02
    intermediate_size = 4096
    num_heads = 16
    num_position_embeddings = 2304
    out_hidden_size = 2048
    patch_size = 16
    spatial_merge_size = 2
    temporal_patch_size = 2
    head_dim = int(hidden_size // num_heads)
    has_scale = False
    w_bits = 8


@dataclass
class Qwen3VLTextConfig:
    bs = 1
    bos_token_id = 151643
    eos_token_id = 151645
    vocab_size = 151936
    hidden_size = 2048
    intermediate_size = 6144
    num_hidden_layers = 28
    num_attention_heads = 16
    num_key_value_heads = 8
    head_dim = 128
    hidden_act = "silu"
    max_position_embeddings = 262144
    initializer_range = 0.02
    rms_norm_eps = 1e-6
    tie_word_embeddings = True
    mrope_interleaved = True
    mrope_section = [24, 20, 20]
    rope_type = "default"
    rope_theta = 5_000_000
    attention_bias = False
    attention_q_bit = 8
    attention_k_bit = 16
    attention_s_bit = 16
    attention_v_bit = 8
    has_scale = False
    w_bits = 8
    mask_value = -32768.0
    chunk_size = None
    cache_len = None


@dataclass
class Qwen3VLConfig:
    vision_config: Qwen3VLVisionConfig = None
    text_config: Qwen3VLTextConfig = None
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    vision_end_token_id = 151653
    tie_word_embeddings = True
