"""PPL (Perplexity) evaluation dataset, supporting wikitext2 and c4."""

import glob
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from llm_compression.models.generate_utils import get_causal_mask, init_prefill_kv_cache
from llm_compression.registry_factory import DATASET_REGISTRY


@DATASET_REGISTRY("ppl")
class PPLDataset:
    """PPL evaluation dataset."""

    def __init__(self, **kwargs):
        self.dataset_name = kwargs["dataset_name"]
        self.seq_len = kwargs.get("seq_len", 2048)
        self.max_steps = kwargs.get("max_steps")
        self.bs = kwargs.get("bs", 1)
        self.tokenizer = kwargs.get("tokenizer")
        self.model = kwargs.get("model")
        self.dtype = kwargs.get("dtype", torch.float32)
        self.testenc = self._load_data(kwargs.get("data_root"))

    def _load_data(self, data_root: str):
        parquet_files = glob.glob(os.path.join(data_root, "test-*.parquet"))
        testdata = load_dataset("parquet", data_files={"test": parquet_files}, split="test")
        # Gemma4 tokenizer defaults add_bos_token=False, but the model expects
        # BOS at position 0.  Enable it for tokenization.
        if self._get_model_type() == "Gemma4TextModel":
            self.tokenizer.add_bos_token = True
        return self.tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    def __len__(self) -> int:
        nsamples = self.testenc.input_ids.numel() // self.seq_len
        return min(nsamples, self.max_steps) if self.max_steps else nsamples

    def get_full_input_ids(self) -> torch.Tensor:
        return self.testenc.input_ids

    def _get_model_type(self) -> str:
        """Get model type for adapting forward parameters across different models."""
        prefill = self.model.prefill
        original_model = getattr(prefill, "_original_model", None)
        model_class_name = (
            original_model.__class__.__name__ if original_model is not None else prefill.__class__.__name__
        )
        return model_class_name

    def _is_hbm_prefill(self) -> bool:
        from llm_compression.converters import HbmWrapper

        return isinstance(self.model.prefill, HbmWrapper)

    def _build_hbm_prefill_inputs(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """Build exported-style prefill mask and zero KV caches for HBM prefill."""
        cfg = getattr(self.model.prefill, "config", None)
        if cfg is None:
            raise ValueError("HBM prefill wrapper does not expose model config")

        batch_size = inputs_embeds.shape[0]
        seq_len = position_ids.shape[1]
        num_layers = cfg.num_hidden_layers
        num_attention_heads = cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_attention_heads)
        max_kvcache_len = cfg.max_kvcache_len

        token_mask = torch.ones(batch_size, seq_len, device=inputs_embeds.device, dtype=torch.int32)
        causal_mask = get_causal_mask(token_mask, max_kvcache_len).squeeze(1).to(inputs_embeds.dtype)

        cache_keys, cache_values = init_prefill_kv_cache(
            batch_size,
            num_layers,
            num_kv_heads,
            head_dim,
            token_mask,
            max_kvcache_len,
            dtype=inputs_embeds.dtype,
        )
        return causal_mask, cache_keys + cache_values

    def _build_attention_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Build causal attention mask.

        Uses -1e3 instead of -inf to avoid overflow in quantized (int) representations.
        """
        return (
            torch.triu(
                torch.full((seq_len, seq_len), -1e3, device=device, dtype=self.dtype),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def _build_linear_attention_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Build linear attention mask (for Qwen3_5Moe and similar models)."""
        return torch.ones(batch_size, seq_len, device=device, dtype=self.dtype)

    def _build_sliding_attention_mask(
        self,
        attention_mask: torch.Tensor,
        sliding_window: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build sliding-window mask from a full causal mask."""
        min_val = attention_mask.min()
        kv_pos = torch.arange(attention_mask.shape[-1], device=attention_mask.device).view(1, 1, -1)
        outside_window = kv_pos <= (position_ids.unsqueeze(-1) - sliding_window)
        return torch.where(outside_window.unsqueeze(1), min_val, attention_mask)

    def _get_chat_prefix(self, device: torch.device):
        """Get chat prefix for models requiring wrapped input (e.g. Gemma4).

        Returns:
            (prefix_ids, prefix_len) for chat-wrapped models.
            (None, 0) for standard models.
        """
        if self._get_model_type() != "Gemma4TextModel":
            return None, 0

        prompt = "."
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        return prefix_ids, prefix_ids.shape[1]

    def _call_model_forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Call the appropriate forward method based on model type."""
        model_type = self._get_model_type()
        device = inputs_embeds.device
        caches = []
        if self._is_hbm_prefill():
            attention_mask, caches = self._build_hbm_prefill_inputs(inputs_embeds, position_ids)

        # Qwen3_5Moe requires linear_attention_mask parameter
        if model_type == "Qwen3_5MoeTextModel":
            linear_attention_mask = self._build_linear_attention_mask(
                position_ids.shape[1], position_ids.shape[0], device
            )
            logits, _, _ = self.model.prefill(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                linear_attention_mask=linear_attention_mask,
                caches=caches,
                return_all_logits=True,
            )
        # Qwen3_5 (dense, hybrid attention) requires linear_attention_mask;
        # forward returns a 5-tuple (logits, k, v, conv_states, recurrent_states).
        # HBM prefill is not supported because conv/recurrent state init differs from KV.
        elif model_type == "Qwen3_5TextModel":
            if self._is_hbm_prefill():
                raise NotImplementedError("PPL eval for Qwen3_5TextModel under HBM prefill is not supported")
            linear_attention_mask = self._build_linear_attention_mask(
                position_ids.shape[1], position_ids.shape[0], device
            )
            logits, *_ = self.model.prefill(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                linear_attention_mask=linear_attention_mask,
                caches=caches,
                return_all_logits=True,
            )
        # Qwen3_VL has deepstack_visual_embeds parameter, pass None for PPL evaluation
        elif model_type == "Qwen3VLTextModel":
            logits, _, _ = self.model.prefill(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                deepstack_visual_embeds=None,
                caches=caches,
                return_all_logits=True,
            )
        # Gemma4 requires slide_attention_mask for sliding-window layers
        elif model_type == "Gemma4TextModel":
            slide_attention_mask = self._build_sliding_attention_mask(
                attention_mask,
                self.model.prefill.config.sliding_window,
                position_ids=position_ids,
            )
            logits, _, _ = self.model.prefill(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                slide_attention_mask=slide_attention_mask,
                caches=caches,
                return_all_logits=True,
            )
        # TODO: Qwen2_5_VLTextModel uses mrope (multimodal RoPE), requiring 3D position_ids
        # and its internal cache_cos/cache_sin implementation differs from standard RoPE,
        # PPL evaluation is not supported for now.
        # Other models: Qwen3TextModel, Qwen3Model (InternVL)
        else:
            logits, _, _ = self.model.prefill(
                input_embeddings=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                caches=caches,
                return_all_logits=True,
            )

        return logits

    def eval(self, predictions=None):
        self.model.eval()
        device = next(self.model.parameters()).device
        prefix_ids, prefix_len = self._get_chat_prefix(device)

        nlls = []
        total_text_tokens = 0
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        full_input_ids = self.get_full_input_ids().to(device)
        nsamples = len(self)

        for i in tqdm(range(0, nsamples, self.bs), desc="Computing PPL"):
            j = min(i + self.bs, nsamples)
            text_ids = full_input_ids[:, i * self.seq_len : j * self.seq_len]
            text_ids = text_ids.reshape(j - i, self.seq_len)

            if prefix_ids is not None:
                input_ids = torch.cat([prefix_ids.expand(j - i, -1), text_ids], dim=1)
            else:
                input_ids = text_ids

            with torch.inference_mode():
                inputs_embeds = self.model.prefill.embed_tokens(input_ids)
                seq_len = input_ids.shape[1]
                batch = input_ids.shape[0]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
                attention_mask = self._build_attention_mask(seq_len, batch, device)

                logits = self._call_model_forward(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

            if logits.dim() != 3:
                raise RuntimeError(
                    "PPL evaluation requires full-sequence logits of shape "
                    f"[batch, seq_len, vocab], but got a tensor of dimension {logits.dim()} "
                    f"with shape {tuple(logits.shape)}. "
                    "Re-export the HBM prefill graph with return_all_logits=True before running PPL evaluation."
                )

            logit_start = max(prefix_len - 1, 0)
            shift_logits = logits[:, logit_start:-1, :].contiguous()
            shift_labels = input_ids[:, logit_start + 1 :].to(shift_logits.device)
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
            )
            nlls.append(loss.sum())
            total_text_tokens += shift_labels.numel()

        total_nll = torch.stack(nlls).sum()
        ppl = torch.exp(total_nll / total_text_tokens)

        print(f"\n{'=' * 50}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Num samples: {nsamples}")
        print(f"Sequence length: {self.seq_len}")
        if prefix_len > 0:
            print(f"Chat prefix length: {prefix_len}")
        print(f"Perplexity: {ppl.item():.4f}")
        print(f"{'=' * 50}\n")

        return {
            "perplexity": ppl.item(),
            "num_samples": nsamples,
            "seq_len": self.seq_len,
        }
