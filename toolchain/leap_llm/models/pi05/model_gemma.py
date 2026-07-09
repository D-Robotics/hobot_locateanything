import math
from pathlib import Path

import torch
from hbdk4.compiler import leap
from safetensors.torch import load_file

from leap_llm.models.pi05.blocks.attention import GemmaAttention
from leap_llm.models.pi05.blocks.configuration_gemma import GemmaConfig
from leap_llm.models.pi05.blocks.configuration_paligemma import PaliGemmaConfig
from leap_llm.models.pi05.blocks.mlp import GemmaMLP
from leap_llm.models.pi05.blocks.rmsnorm import GemmaRMSNorm
from leap_llm.nn.modules import Embedding
from leap_llm.nn.utils import Model, timeit  # noqa: E402


class GemmaDecoderLayer(Model):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def build(self, hidden_states, cos, sin, attention_mask, softmax_mask):
    # def build(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, new_k, new_v= self.self_attn(
            hidden_states=hidden_states,
            cos = cos,
            sin = sin,
            attention_mask=attention_mask,
            softmax_mask=softmax_mask,
        )
        hidden_states = leap.add(hidden_states, residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(hidden_states, residual)

        return hidden_states, new_k, new_v
        
    def forward(self, hidden_states, cos, sin, attention_mask, softmax_mask=None):
    # def forward(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, new_k, new_v = self.self_attn(
            hidden_states=hidden_states,
            cos = cos,
            sin = sin,
            attention_mask=attention_mask,
            softmax_mask=softmax_mask,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, new_k, new_v
    
class GemmaModel(Model):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.hidden_size = config.hidden_size
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        cos, sin = self._set_cos_sin_cache(
            config.max_position_embeddings,
            config.head_dim,
            base=config.rope_theta,
        )
        self.cos = cos[:config.vision_token_num*3 + 200, :]
        self.sin = sin[:config.vision_token_num*3 + 200, :]
        self.lang_sqrt = math.sqrt(2048)

    def _set_cos_sin_cache(self, max_seq_len_cached, head_dim, base=1000000.0):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        t = torch.arange(max_seq_len_cached, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(torch.float16)
        sin_cached = emb.sin().to(torch.float16)
        return cos_cached, sin_cached
    
    def build(self, tokens, inputs_embeds, attention_mask, position_ids, softmax_mask):
    # def build(self, tokens, inputs_embeds, attention_mask, position_ids):

        # embed positions
        _bsz, seqlen = tokens.type.shape
        tokens = leap.reshape(tokens, [seqlen, _bsz])
        lang_emb = self.embed_tokens(tokens)
        lang_emb = leap.mul(lang_emb, self.lang_sqrt)
        lang_emb = leap.reshape(lang_emb, (1, seqlen, self.hidden_size))
        hidden_states = leap.concat([inputs_embeds, lang_emb], dim=1)
        new_keys = []
        new_values = []
        position_ids = leap.reshape(position_ids, [-1, _bsz])
        cos = leap.gather_nd(self.cos, position_ids, 0)
        sin = leap.gather_nd(self.sin, position_ids, 0)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:

            layer_outputs, new_k, new_v = decoder_layer(
                hidden_states,
                cos = cos,
                sin = sin,
                attention_mask=attention_mask,
                softmax_mask=softmax_mask,
            )

            hidden_states = layer_outputs
            new_keys.append(new_k)
            new_values.append(new_v)

        hidden_states = self.norm(hidden_states)

        return hidden_states, *new_keys, *new_values
    
    def forward(
        self,
        tokens, 
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        softmax_mask: torch.Tensor | None = None,
    ):
        # embed positions
        lang_emb = self.embed_tokens(tokens)
        lang_emb = lang_emb * self.lang_sqrt
        hidden_states = torch.concat([inputs_embeds, lang_emb], dim=1)
        new_keys = []
        new_values = []
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:

            layer_outputs, new_k, new_v = decoder_layer(
                hidden_states,
                cos = self.cos.to(device=hidden_states.device)[position_ids],
                sin = self.sin.to(device=hidden_states.device)[position_ids],
                attention_mask=attention_mask,
                softmax_mask=softmax_mask,
            )

            hidden_states = layer_outputs
            new_keys.append(new_k)
            new_values.append(new_v)


        hidden_states = self.norm(hidden_states)

        return hidden_states, *new_keys, *new_values



class GemmaForCausalLM(Model):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__()
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        
class LanguageModel:
    @staticmethod
    @timeit
    def build(
        model_dir: str,
        vision_token_num
    ) -> "LanguageModel":

        state_dict = load_file(model_dir)

        # Filter keys for language_model model components
        language_model_prefix = "paligemma_with_expert.paligemma.model.language_model."
        filtered_state_dict = {
            k.replace(language_model_prefix, ""): v 
            for k, v in state_dict.items() 
            if k.startswith(language_model_prefix)
        }
        filtered_state_dict['embed_tokens.weight'] = state_dict['paligemma_with_expert.paligemma.lm_head.weight']


        modelConfig = PaliGemmaConfig()
        modelConfig.text_config.hidden_size = 2048
        modelConfig.text_config.intermediate_size = 16_384
        modelConfig.text_config.num_attention_heads = 8
        modelConfig.text_config.head_dim = 256
        modelConfig.text_config.num_hidden_layers = 18
        modelConfig.text_config.num_key_value_heads = 1
        modelConfig.text_config.hidden_activation = "gelu_pytorch_tanh"
        modelConfig.text_config.torch_dtype = "float16"
        modelConfig.text_config.vocab_size = 257152
        modelConfig.text_config.vision_token_num = vision_token_num

        model = GemmaModel(modelConfig.text_config)
        model.load_state_dict(filtered_state_dict, strict=True)
        return LanguageModel(model, modelConfig)

    def __init__(self, model: GemmaModel, model_args: PaliGemmaConfig):
        self.model = model
        self.model_args = model_args
        # self.tokenizer = tokenizer
        # self.formatter = ChatFormat(tokenizer)
        
    def get_leap_input_types(self, seq_len, token_id_len) -> list[leap.TensorType]:
        input_types = [
            leap.TensorType([1, token_id_len], leap.int32),
            leap.TensorType([1, seq_len, 2048], leap.float16),
            leap.TensorType([1, 1, seq_len+token_id_len, seq_len+token_id_len], leap.float16),
            leap.TensorType([1, seq_len+token_id_len], leap.int32),
            leap.TensorType([1, seq_len+token_id_len], leap.float16),
        ]
        return input_types

    def compile(
        self,
        output_model_path: str,
        **kwargs,
    ):
        assert self.model.is_compiled, "Model must be compiled before compiling."

        inputs = self.get_leap_input_types(self.model_args.text_config.vision_token_num*3, 200)
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "gemma", bc_path)
        # 编译 HBO 模型并链接成最终模型
        hbos = []
        bc_path = str(Path(output_model_path).with_suffix(".convert.bc"))
        mlir_module = self.model.convert_mlir(
            bc_module,
            save_path=bc_path,
            march=kwargs["march"],
            dynamic_quant=True,
            softmax_version="skip",
        )
        # exit(0)
        kwargs["core_num"] = 4
        kwargs["max_l2m_size"] = 25165824
        
        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        hbo_model = self.model.compile_hbo(
            mlir_module,
            hbo_path,
            **kwargs,
        )
        hbos.append(hbo_model)

        hbm_path = str(Path(output_model_path).with_suffix(".hbm"))
        return self.model.link_models(hbos, hbm_path)