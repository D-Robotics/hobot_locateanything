from pathlib import Path

import torch
from hbdk4.compiler import leap
from safetensors.torch import load_file

from leap_llm.models.pi05.blocks.configuration_paligemma import PaliGemmaConfig
from leap_llm.models.pi05.blocks.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from leap_llm.nn.modules import ConstFakeQuant, DynamicQuantLinear, DynamicQuantMatmul, FakeQuantGELU, FakeQuantMul
from leap_llm.nn.modules.layer_norm import LayerNorm
from leap_llm.nn.modules.vision_embedding import FakeQuantPatchEmbedding
from leap_llm.nn.utils import Model, Module, timeit  # noqa: E402


class SiglipFakeQuantEmbedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.weight_fake_quant = ConstFakeQuant(8)
        self.absmax_weight = None

    def build(self, x):
        weight_data = self.weight.data.to(torch.float16)
        weight_data = self.weight_fake_quant(weight_data)
        # print("Embedding build absmax:", self.weight_fake_quant.absmax)
        return leap.gather_nd(weight_data, x, 0)

    def forward(self, x: torch.Tensor):
        if self.absmax_weight is None:
            weight_data = self.weight_fake_quant(self.weight.data)
        inputs_embeds = weight_data[x]
        return inputs_embeds


class SiglipAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.k_proj = DynamicQuantLinear(self.embed_dim, self.embed_dim)
        self.v_proj = DynamicQuantLinear(self.embed_dim, self.embed_dim)
        self.q_proj = DynamicQuantLinear(self.embed_dim, self.embed_dim)
        self.out_proj = DynamicQuantLinear(self.embed_dim, self.embed_dim)
        
        self.qk = DynamicQuantMatmul()
        self.sv = DynamicQuantMatmul()
        self.mul_attn_weight = FakeQuantMul(quantized=False)

    def build(self, hidden_states, output_attentions):
        seqlen = hidden_states.type.shape[1]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(
            query_states, [seqlen, self.num_heads, self.head_dim]
        )
        query_states = leap.transpose(query_states, [1, 0, 2])
        key_states = leap.reshape(
            key_states, [seqlen, self.num_heads, self.head_dim]
        )

        key_states = leap.transpose(key_states, [1, 0, 2])
        value_states = leap.reshape(
            value_states, [seqlen, self.num_heads, self.head_dim]
        )
        value_states = leap.transpose(value_states, [1, 2, 0])
        
        attn_weights = self.qk(query_states, key_states)
        attn_weights = self.mul_attn_weight(
            attn_weights, self.scale
        )

        attn_weights = leap.softmax(attn_weights, -1)
        ret_attn_weights = attn_weights
        attn_weights = self.sv(attn_weights, value_states)
        
        attn_weights = leap.transpose(attn_weights, [1, 0, 2])
        attn_weights = leap.reshape(attn_weights, [seqlen, self.embed_dim])
        attn_output = self.out_proj(attn_weights)
        if not output_attentions:
            attn_weights = None
        return attn_output, ret_attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)


        attn_weights = self.qk(queries, keys.transpose(-1, -2)) * self.scale

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)

        attn_output = self.sv(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights        

class SiglipMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = FakeQuantGELU()
        self.fc1 = DynamicQuantLinear(config.hidden_size, config.intermediate_size)
        self.fc2 = DynamicQuantLinear(config.intermediate_size, config.hidden_size)

    def build(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = leap.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(Module):
    def __init__(self, config: SiglipVisionConfig | SiglipTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        
    def build(self, hidden_states, output_attentions = False):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = leap.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool | None = False,
    ) -> tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class SiglipEncoder(Module):    
    def __init__(
        self,
        config: SiglipConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def build(self, inputs_embeds):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:

            layer_outputs = encoder_layer(
                hidden_states,
                output_attentions = True
            )

            hidden_states = layer_outputs[0]
            attn_weight = layer_outputs[1]
        
        return hidden_states, attn_weight
    
    # Ignore copy
    def forward(
        self,
        inputs_embeds,
    ):

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:

            layer_outputs = encoder_layer(
                hidden_states,
                output_attentions = True
            )

            hidden_states = layer_outputs[0]
            attn_weight = layer_outputs[1]
        

        return hidden_states, attn_weight

class SiglipVisionEmbeddings(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = FakeQuantPatchEmbedding(self.embed_dim, config.num_channels, self.patch_size)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = SiglipFakeQuantEmbedding(self.num_positions, self.embed_dim)

    def build(self, pixel_values, position_ids):
        pixel_values = leap.transpose(pixel_values, [0, 2, 3, 1])
        self.patch_embedding.to("cpu", dtype=torch.float32)
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = leap.reshape(patch_embeds, (1, self.num_patches, self.embed_dim))
        position_ids = leap.reshape(position_ids, [self.num_patches, 1])
        embeddings = leap.add(embeddings, self.position_embedding(position_ids))
        return embeddings

    def forward(self, pixel_values: torch.FloatTensor, position_ids) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings

class SiglipMultiheadAttentionPoolingHead(Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

class PaliGemmaMultiModalProjector(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.linear = DynamicQuantLinear(config.hidden_size, config.projection_dim, bias=True)
        
    def build(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states
        
        
    def forward(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states

class SiglipVisionModel(Model):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)
            
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

    def build(self, pixel_values, position_ids):
        hidden_states = self.embeddings(pixel_values, position_ids)
        last_hidden_state, attn_weight = self.encoder(
            inputs_embeds=hidden_states,
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        visual_token_num = self.config.visual_token_num # T
        attn_weight = leap.reshape(
            attn_weight, [1, attn_weight.type.shape[0], attn_weight.type.shape[1], attn_weight.type.shape[2]]
        )
        A = leap.reduce_mean(attn_weight, dims=[1], keepDim=False)
        image_attentions = leap.reduce_mean(A, dims=[1], keepDim=False)
        important_indices = leap.sort(image_attentions, descending=True)[1]
        important_indices = leap.slice(
            important_indices, [0, 0], [1, visual_token_num], [1, 1]
        )
        important_indices = leap.sort(important_indices)[0]
        important_indices = leap.reshape(important_indices, [visual_token_num, 1])
        last_hidden_state = leap.reshape(
            last_hidden_state, [last_hidden_state.type.shape[1], last_hidden_state.type.shape[2]]
        )
        last_hidden_state = leap.gather_nd(last_hidden_state, important_indices, 0)
        last_hidden_state = leap.reshape(
            last_hidden_state, [1, last_hidden_state.type.shape[0], last_hidden_state.type.shape[1]]
        )
        image_features = self.multi_modal_projector(last_hidden_state)
        return image_features

    def forward(
        self,
        pixel_values,
        position_ids,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(pixel_values, position_ids)

        last_hidden_state, attn_weight = self.encoder(
            inputs_embeds=hidden_states,
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        B, N, D = last_hidden_state.shape
        K = self.config.visual_token_num # T

        A = attn_weight.mean(dim=1)       # [B, N, N]
        token_scores = A.mean(dim=1)     # [B, N]

        _, idx = torch.topk(token_scores, K, dim=-1)

        idx_sorted, _ = torch.sort(idx, dim=-1)

        batch_idx = torch.arange(B).unsqueeze(-1).to(last_hidden_state.device)
        x_pruned = last_hidden_state[batch_idx, idx_sorted]  # [B, K, D]

        image_features = self.multi_modal_projector(x_pruned)
        return image_features       
            
            
            
class Siglip:
    @staticmethod
    @timeit
    def build(
        model_dir: str,
        vision_token_num,
    ) -> "Siglip":

        state_dict = load_file(model_dir)
        vision_prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        projector_prefix = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."

        filtered_state_dict = {}

        # 提取 vision_tower 部分
        filtered_state_dict.update({
            k.replace(vision_prefix, ""): v 
            for k, v in state_dict.items() 
            if k.startswith(vision_prefix)
        })

        # 提取 multi_modal_projector.linear 部分
        filtered_state_dict.update({
            k.replace(projector_prefix, "multi_modal_projector.linear."): v 
            for k, v in state_dict.items() 
            if k.startswith(projector_prefix)
        })
        
        weight_name = "embeddings.patch_embedding.weight"
        filtered_state_dict[weight_name] = (
            filtered_state_dict[weight_name].permute(0, 2, 3, 1).contiguous()
        )
        # Initialize model and load filtered weights
        modelConfig = PaliGemmaConfig()
        modelConfig.vision_config.intermediate_size = 4304
        modelConfig.vision_config.projection_dim = 2048
        modelConfig.vision_config.projector_hidden_act = "gelu_fast"
        modelConfig.vision_config.torch_dtype = "float32"
        modelConfig.vision_config.visual_token_num = vision_token_num
        model = SiglipVisionModel(modelConfig.vision_config)
        model.load_state_dict(filtered_state_dict, strict=True)

        return Siglip(model, modelConfig)

    def __init__(self, model: SiglipVisionModel, model_args: PaliGemmaConfig):
        self.model = model
        self.model_args = model_args

    def get_leap_input_types(self, inp_image_size) -> list[leap.TensorType]:
        input_types = [
            leap.TensorType([1, 3, inp_image_size, inp_image_size], leap.float16),
            leap.TensorType([1, 256], leap.int64),
        ]
        return input_types

    def compile(
        self,
        output_model_path: str,
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        inputs = self.get_leap_input_types(224)
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "siglip", bc_path)

        hbos = []
        bc_path = str(Path(output_model_path).with_suffix(".convert.bc"))
        mlir_module = self.model.convert_mlir(
            bc_module,
            save_path=bc_path,
            enable_spu=False,
            march=kwargs["march"],
            dynamic_quant=True,
            softmax_version="skip",
        )
        # return
        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        hbo_model = self.model.compile_hbo(
            mlir_module,
            hbo_path,
            **kwargs,
        )
        hbos.append(hbo_model)

        hbm_path = str(Path(output_model_path).with_suffix(".hbm"))
        return self.model.link_models(hbos, hbm_path)