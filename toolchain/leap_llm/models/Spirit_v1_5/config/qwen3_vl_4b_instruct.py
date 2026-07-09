from dataclasses import dataclass

# Qwen3VLVisionConfig {
#   "deepstack_visual_indexes": [
#     5,
#     11,
#     17
#   ],
#   "depth": 24,
#   "dtype": "bfloat16",
#   "hidden_act": "gelu_pytorch_tanh",
#   "hidden_size": 1024,
#   "in_channels": 3,
#   "initializer_range": 0.02,
#   "intermediate_size": 4096,
#   "model_type": "qwen3_vl",
#   "num_heads": 16,
#   "num_position_embeddings": 2304,
#   "out_hidden_size": 2560,
#   "patch_size": 16,
#   "spatial_merge_size": 2,
#   "temporal_patch_size": 2,
#   "transformers_version": "5.0.0.dev0"
# }


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
    out_hidden_size = 2560
    patch_size = 16
    spatial_merge_size = 2
    temporal_patch_size = 2
    head_dim = int(hidden_size // num_heads)
    has_scale = False
    w_bits = 8


# Qwen3VLTextConfig {
#   "attention_bias": false,
#   "attention_dropout": 0.0,
#   "bos_token_id": 151643,
#   "dtype": "bfloat16",
#   "eos_token_id": 151645,
#   "head_dim": 128,
#   "hidden_act": "silu",
#   "hidden_size": 2560,
#   "initializer_range": 0.02,
#   "intermediate_size": 9728,
#   "max_position_embeddings": 262144,
#   "model_type": "qwen3_vl_text",
#   "num_attention_heads": 32,
#   "num_hidden_layers": 36,
#   "num_key_value_heads": 8,
#   "rms_norm_eps": 1e-06,
#   "rope_parameters": {
#     "mrope_interleaved": true,
#     "mrope_section": [
#       24,
#       20,
#       20
#     ],
#     "rope_theta": 5000000,
#     "rope_type": "default"
#   },
#   "tie_word_embeddings": true,
#   "transformers_version": "5.0.0.dev0",
#   "use_cache": true,
#   "vocab_size": 151936
# }
@dataclass
class Qwen3VLTextConfig:
    bs = 1
    bos_token_id = 151643
    eos_token_id = 151645
    vocab_size = 151936
    hidden_size = 2560
    intermediate_size = 9728
    num_hidden_layers = 36
    num_attention_heads = 32
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
