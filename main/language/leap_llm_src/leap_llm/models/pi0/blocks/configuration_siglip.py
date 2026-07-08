class SiglipVisionConfig:
    model_type = "siglip_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class SiglipTextConfig:
    model_type = "siglip_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        projection_size=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.projection_size = (
            projection_size if projection_size is not None else hidden_size
        )


class SiglipConfig:
    model_type = "siglip"
    sub_configs = {"text_config": SiglipTextConfig, "vision_config": SiglipVisionConfig}

    def __init__(self, text_config=None, vision_config=None):
        super().__init__()

        if text_config is None:
            text_config = {}

        if vision_config is None:
            vision_config = {}

        self.text_config = SiglipTextConfig(**text_config)
        self.vision_config = SiglipVisionConfig(**vision_config)

        self.initializer_factor = 1.0
