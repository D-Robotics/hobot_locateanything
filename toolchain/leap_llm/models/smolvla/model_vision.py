"""SmolVLM vision encoder + connector for SmolVLA quantization."""

from pathlib import Path

from hbdk4.compiler import leap

from leap_llm.models.smolvla.blocks.configuration_smolvlm import (
    SmolVLAPolicyConfig,
    SmolVLMVisionConfig,
)
from leap_llm.models.smolvla.blocks.smolvlm_vision import SmolVLMVisionEmbeddings
from leap_llm.models.smolvla.blocks.smolvlm_vision_encoder import SmolVLMVisionEncoder
from leap_llm.models.smolvla.blocks.vision_connector import SmolVLMConnector
from leap_llm.models.smolvla.smolvla_utils import load_policy_config
from leap_llm.models.smolvla.weight_mapper import (
    connector_state_dict,
    load_full_state_dict,
    vision_state_dict,
)
from leap_llm.nn.modules.layer_norm import LayerNorm
from leap_llm.nn.utils import Model, timeit


class SmolVLMVisionCore(Model):
    """SigLIP vision tower (SmolVLM2) without Pi0 token pruning."""

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SmolVLMVisionEmbeddings(config)
        self.encoder = SmolVLMVisionEncoder(config)
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def build(self, pixel_values, position_ids):
        hidden_states = self.embeddings(pixel_values, position_ids)
        last_hidden_state, _ = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

    def forward(self, pixel_values, position_ids=None):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state, _ = self.encoder(inputs_embeds=hidden_states)
        return self.post_layernorm(last_hidden_state)


class SmolVLMVisionModel(Model):
    def __init__(
        self,
        vision: SmolVLMVisionCore,
        connector: SmolVLMConnector,
        policy_cfg: SmolVLAPolicyConfig,
    ):
        super().__init__()
        self.vision = vision
        self.connector = connector
        self.policy_cfg = policy_cfg

    def build(self, pixel_values, position_ids):
        vision_out = self.vision(pixel_values, position_ids)
        return self.connector(vision_out)

    def forward(self, pixel_values, position_ids):
        vision_out = self.vision(pixel_values, position_ids)
        return self.connector(vision_out)


class SmolVLMVision:
    @staticmethod
    @timeit
    def build(
        model_path: str,
        policy_cfg: SmolVLAPolicyConfig | None = None,
        vision_tokens_num: int | None = None,
    ) -> "SmolVLMVision":
        root = Path(model_path)
        if policy_cfg is None:
            policy_cfg = load_policy_config(root)
        if vision_tokens_num is not None:
            policy_cfg.vision_tokens_num = vision_tokens_num

        state = load_full_state_dict(root)
        v_state = vision_state_dict(state)
        wkey = "embeddings.patch_embedding.weight"
        if wkey in v_state and v_state[wkey].ndim == 4 and v_state[wkey].shape[1] == 3:
            v_state = dict(v_state)
            v_state[wkey] = v_state[wkey].permute(0, 2, 3, 1).contiguous()
        c_state = connector_state_dict(state)

        v_cfg = SmolVLMVisionConfig.from_policy_config(policy_cfg)
        vision_core = SmolVLMVisionCore(v_cfg)
        vision_core.load_state_dict(v_state, strict=False)

        import math
        # Infer scale_factor directly from the connector weight in the checkpoint
        # to avoid being affected by the CLI default --vision_tokens_num (144).
        conn_weight = c_state.get("modality_projection.proj.weight")
        if conn_weight is not None:
            scale_factor = round(math.sqrt(conn_weight.shape[1] / policy_cfg.vision_hidden_size))
        else:
            raw_patches = (policy_cfg.image_height // policy_cfg.vision_patch_size) ** 2
            scale_factor = round(math.sqrt(raw_patches / policy_cfg.vision_tokens_num))
        connector = SmolVLMConnector(
            policy_cfg.vision_hidden_size, policy_cfg.text_hidden_size, scale_factor=scale_factor
        )
        if c_state:
            connector.load_state_dict(c_state, strict=False)

        model = SmolVLMVisionModel(vision_core, connector, policy_cfg)
        return SmolVLMVision(model, policy_cfg)

    def __init__(self, model: SmolVLMVisionModel, policy_cfg: SmolVLAPolicyConfig):
        self.model = model
        self.policy_cfg = policy_cfg

    def get_leap_input_types(self, image_size: int) -> list[leap.TensorType]:
        return [
            leap.TensorType([1, 3, image_size, image_size], leap.float16),
            leap.TensorType(
                [
                    1,
                    (image_size // self.policy_cfg.vision_patch_size) ** 2,
                ],
                leap.int64,
            ),
        ]

    def compile(self, output_model_path: str, **kwargs):
        assert self.model.is_compiled, "Model must be compiled before compiling."
        image_size = self.policy_cfg.image_height
        inputs = self.get_leap_input_types(image_size)
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "smolvla_vision", bc_path)
        hbos = []
        bc_path = str(Path(output_model_path).with_suffix(".convert.bc"))
        mlir_module = self.model.convert_mlir(
            bc_module,
            save_path=bc_path,
            enable_spu=False,
            march=kwargs["march"],
            dynamic_quant=True,
        )
        kwargs["core_num"] = kwargs.get("core_num", 1)
        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        hbo_model = self.model.compile_hbo(mlir_module, hbo_path, **kwargs)
        hbos.append(hbo_model)
        hbm_path = str(Path(output_model_path).with_suffix(".hbm"))
        return self.model.link_models(hbos, hbm_path)
