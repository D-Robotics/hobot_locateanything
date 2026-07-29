"""Official-decoder hybrid generation over LocateAnything fixed Language graphs."""

from __future__ import annotations

import hashlib
import importlib.util
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from compiler.scripts.common.coordinates import token_ids_from_payload


@dataclass(frozen=True)
class HybridGenerationConfig:
    """Generation settings used by the calibration bundle's official worker."""

    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None
    repetition_penalty: float = 1.1
    generation_mode: str = "hybrid"

    def validate(self) -> None:
        if self.generation_mode != "hybrid":
            raise ValueError("fixed-graph generator only supports official hybrid mode")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OfficialDecoding:
    module: ModuleType
    source: Path
    sha256: str

    @property
    def sample_tokens(self):
        return self.module.sample_tokens

    @property
    def handle_pattern(self):
        return self.module.handle_pattern

    @property
    def get_token_ids_from_config(self):
        return self.module.get_token_ids_from_config

    def describe(self) -> dict[str, str]:
        return {"source": str(self.source), "sha256": self.sha256}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_decoding(model_path: Path) -> OfficialDecoding:
    """Load sampling and pattern parsing from the checkpoint's released code."""

    source = (model_path / "generate_utils.py").resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"official LocateAnything decoder is missing: {source}"
        )
    module_name = f"locateanything_official_generate_utils_{_sha256(source)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official decoder from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("sample_tokens", "handle_pattern", "get_token_ids_from_config"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"{source} does not define callable {name}()")
    return OfficialDecoding(module=module, source=source, sha256=_sha256(source))


def seed_generation(seed: int, device: Any) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if getattr(device, "type", str(device).split(":", 1)[0]) == "cuda":
        torch.cuda.manual_seed_all(seed)


class FixedGraphHybridGenerator:
    """Run the released hybrid cache contract with fixed graph profiles.

    The MTP draft K/V rows are never treated as generated tokens. Instead, the
    next forward consumes the prior accepted prefix followed by a fresh MTP
    window, then retains only that real prefix. This mirrors the upstream
    ``past_key_values[:, :, :generated.shape[1]]`` truncation without an
    artificial padded q6 cache-commit pass.
    """

    def __init__(
        self,
        model: Any,
        rotation: Any,
        device: Any,
        dtype: Any,
        zero_caches: list[Any],
        emulator: Any,
        official: OfficialDecoding,
        *,
        chunk_size: int,
        cache_len: int,
        pbd_query_len: int,
        image_token_id: int,
        token_ids: dict[str, int],
        model_max_length: int,
        replay: Any | None = None,
    ) -> None:
        self.model = model
        self.rotation = rotation
        self.device = device
        self.dtype = dtype
        self.zero_caches = zero_caches
        self.emulator = emulator
        self.official = official
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.pbd_query_len = pbd_query_len
        self.image_token_id = image_token_id
        self.token_ids = {str(name): int(value) for name, value in token_ids.items()}
        self.model_max_length = int(model_max_length)
        if self.model_max_length <= 0:
            raise ValueError("model_max_length must be positive")
        self.replay = replay

    def _validated_token_ids(self, payload: dict[str, Any]) -> dict[str, int]:
        payload_ids = token_ids_from_payload(payload)
        mismatches = {
            name: {"checkpoint": self.token_ids.get(name), "payload": value}
            for name, value in payload_ids.items()
            if self.token_ids.get(name) != value
        }
        if mismatches:
            raise ValueError(f"payload token IDs disagree with checkpoint: {mismatches}")
        return self.token_ids

    def _replay(self):
        if self.replay is None:
            from leap_llm.apis.calibration import locateanything_replay

            self.replay = locateanything_replay
        return self.replay

    def _stage(self, name: str, quantized: bool) -> None:
        self.emulator.set_stage(name)
        self.emulator.set_enabled(quantized)

    def _prefill(self, payload: dict[str, Any], quantized: bool):
        import torch
        replay = self._replay()
        inputs = replay.build_prefill_inputs(
            self.model,
            payload,
            self.rotation,
            chunk_size=self.chunk_size,
            cache_len=self.cache_len,
            image_token_id=self.image_token_id,
            device=self.device,
            dtype=self.dtype,
        )
        embeds, positions, attention, active_len = inputs
        self._stage("prefill", quantized)
        with torch.no_grad():
            logits, new_keys, new_values = self.model(
                embeds, positions, attention, *self.zero_caches
            )
        del logits
        keys, values = replay.build_right_aligned_caches(
            new_keys,
            new_values,
            active_len=active_len,
            cache_len=self.cache_len,
        )
        del new_keys, new_values
        return keys, values, active_len

    def _decode_inputs(
        self,
        token_ids: list[int],
        *,
        q_len: int,
        past_len: int,
        is_pbd: bool,
        pbd_prefix_len: int = 0,
    ):
        return self._replay().build_decode_inputs(
            self.model,
            token_ids,
            q_len=q_len,
            past_len=past_len,
            cache_len=self.cache_len,
            is_pbd=is_pbd,
            pbd_prefix_len=pbd_prefix_len,
            device=self.device,
            dtype=self.dtype,
        )

    def _run_causal(
        self,
        tokens: list[int],
        keys: list[Any],
        values: list[Any],
        history_len: int,
        quantized: bool,
    ):
        import torch

        if not tokens or history_len + len(tokens) > self.cache_len:
            raise OverflowError("accepted tokens exceed the fixed KV cache")
        inputs = self._decode_inputs(
            tokens, q_len=len(tokens), past_len=history_len, is_pbd=False
        )
        self._stage(f"ar_q{len(tokens)}", quantized)
        with torch.no_grad():
            logits, new_keys, new_values = self.model(*inputs, *keys, *values)
        return logits, new_keys, new_values

    def _run_pbd(
        self,
        tokens: list[int],
        keys: list[Any],
        values: list[Any],
        history_len: int,
        quantized: bool,
        pbd_prefix_len: int,
    ):
        import torch

        q_len = len(tokens)
        if q_len != pbd_prefix_len + self.pbd_query_len:
            raise ValueError("PBD graph must end with exactly one MTP window")
        if history_len + q_len > self.cache_len:
            raise OverflowError("PBD block exceeds the fixed KV cache")
        inputs = self._decode_inputs(
            tokens,
            q_len=q_len,
            past_len=history_len,
            is_pbd=True,
            pbd_prefix_len=pbd_prefix_len,
        )
        self._stage(f"pbd_q{q_len}", quantized)
        with torch.no_grad():
            logits, draft_keys, draft_values = self.model(*inputs, *keys, *values)
        return logits, draft_keys, draft_values

    @staticmethod
    def _retain_prefix(
        keys: list[Any],
        values: list[Any],
        new_keys: list[Any],
        new_values: list[Any],
        accepted: int,
    ):
        import torch

        if accepted <= 0:
            raise ValueError("accepted prefix must be non-empty")
        updated_keys = [
            torch.cat((cache[:, accepted:], current[:, :accepted]), dim=1)
            for cache, current in zip(keys, new_keys, strict=True)
        ]
        updated_values = [
            torch.cat((cache[:, accepted:], current[:, :accepted]), dim=1)
            for cache, current in zip(values, new_values, strict=True)
        ]
        return updated_keys, updated_values

    def generate(
        self,
        payload: dict[str, Any],
        *,
        quantized: bool,
        seed: int,
        config: HybridGenerationConfig,
    ) -> dict[str, Any]:
        import torch

        config.validate()
        seed_generation(seed, self.device)
        token_ids = self._validated_token_ids(payload)
        generated = payload["prompt_input_ids"].reshape(-1).to(
            device=self.device, dtype=torch.long
        )
        if generated.numel() == 0:
            raise ValueError("prompt_input_ids is empty")
        prompt_len = int(generated.numel())
        requested_total_limit = min(
            self.model_max_length, prompt_len + config.max_new_tokens
        )
        effective_total_limit = min(requested_total_limit, self.cache_len)
        if prompt_len >= effective_total_limit:
            raise ValueError(
                f"prompt length {prompt_len} leaves no generation capacity under "
                f"model/cache limit {effective_total_limit}"
            )
        keys, values, history_len = self._prefill(payload, quantized)
        response: list[int] = []
        steps: list[dict[str, Any]] = []
        mode = "pbd"
        ar_logits = None
        pending_pbd: list[int] = []
        if effective_total_limit == self.cache_len < requested_total_limit:
            limit_stop_reason = "cache_limit"
        elif requested_total_limit == self.model_max_length < (
            prompt_len + config.max_new_tokens
        ):
            limit_stop_reason = "model_max_length"
        else:
            limit_stop_reason = "max_new_tokens"
        stop_reason = limit_stop_reason
        pbd_calls = 0
        pbd_fused_prefix_calls = 0
        q1_commit_calls = 0
        ar_fallback_sample_calls = 0

        try:
            while int(generated.numel()) < effective_total_limit:
                if mode == "pbd":
                    pbd_prefix_len = len(pending_pbd)
                    if history_len + pbd_prefix_len + self.pbd_query_len > self.cache_len:
                        stop_reason = "cache_limit_before_pbd"
                        break
                    if pending_pbd:
                        pbd_tokens = [
                            *pending_pbd,
                            pending_pbd[-1],
                            *([token_ids["default_mask_token_id"]] *
                              (self.pbd_query_len - 1)),
                        ]
                    else:
                        pbd_tokens = [
                            int(generated[-1].item()),
                            *([token_ids["default_mask_token_id"]] *
                              (self.pbd_query_len - 1)),
                        ]
                    logits, draft_keys, draft_values = self._run_pbd(
                        pbd_tokens,
                        keys,
                        values,
                        history_len,
                        quantized,
                        pbd_prefix_len,
                    )
                    pbd_calls += 1
                    if pending_pbd:
                        next_keys, next_values = self._retain_prefix(
                            keys, values, draft_keys, draft_values, pbd_prefix_len
                        )
                        del keys, values
                        keys, values = next_keys, next_values
                        history_len += pbd_prefix_len
                        pbd_fused_prefix_calls += 1
                    del draft_keys, draft_values
                    pbd_logits = logits[:, pbd_prefix_len:]
                    probabilities, confidence, sampled, decoded = (
                        self.official.sample_tokens(
                            pbd_logits,
                            generated.unsqueeze(0),
                            token_ids,
                            keep_k=5,
                            generation_mode="hybrid",
                            temperature=config.temperature,
                            top_p=config.top_p,
                            top_k=config.top_k,
                            repetition_penalty=config.repetition_penalty,
                        )
                    )
                    use_sample = bool((decoded[0] == 0).all().item())
                    selected = sampled[0] if use_sample else decoded[0]
                    pattern = self.official.handle_pattern(
                        selected, token_ids, "hybrid"
                    )
                    accepted = [int(token) for token in pattern["tokens"]]
                    step = {
                        "index": len(steps),
                        "mode": "pbd",
                        "q_len": pbd_prefix_len + self.pbd_query_len,
                        "history_len": history_len,
                        "pattern": str(pattern["type"]),
                        "accepted_token_ids": accepted,
                    }
                    steps.append(step)
                    del probabilities, confidence, sampled, decoded, selected, pbd_logits, logits

                    remaining = effective_total_limit - int(generated.numel())
                    if not accepted:
                        raise RuntimeError("PBD decoder accepted no tokens")
                    if len(accepted) > remaining:
                        response.extend(accepted[:remaining])
                        generated = torch.cat(
                            (
                                generated,
                                torch.tensor(
                                    accepted[:remaining], device=self.device,
                                    dtype=torch.long,
                                ),
                            )
                        )
                        stop_reason = limit_stop_reason
                        break

                    if pattern["type"] == "im_end":
                        response.extend(accepted)
                        generated = torch.cat(
                            (
                                generated,
                                torch.tensor(
                                    accepted, device=self.device, dtype=torch.long
                                ),
                            )
                        )
                        stop_reason = "im_end"
                        break

                    if pattern["type"] == "error_box":
                        ar_logits, new_keys, new_values = self._run_causal(
                            accepted,
                            keys,
                            values,
                            history_len,
                            quantized,
                        )
                        next_keys, next_values = self._retain_prefix(
                            keys, values, new_keys, new_values, len(accepted)
                        )
                        del keys, values, new_keys, new_values
                        keys, values = next_keys, next_values
                        history_len += len(accepted)
                        ar_logits = ar_logits[:, -1:]
                    response.extend(accepted)
                    generated = torch.cat(
                        (
                            generated,
                            torch.tensor(
                                accepted, device=self.device, dtype=torch.long
                            ),
                        )
                    )
                    mode = "ar_q1" if pattern["type"] == "error_box" else "pbd"
                    if mode == "pbd":
                        pending_pbd = accepted
                        del ar_logits
                        ar_logits = None
                    else:
                        pending_pbd = []
                    continue

                if ar_logits is None:
                    raise RuntimeError("AR mode has no logits from the accepted prefix")
                if history_len >= self.cache_len:
                    stop_reason = "cache_limit_before_ar"
                    break
                probabilities, confidence, sampled, _ = self.official.sample_tokens(
                    ar_logits[:, -1:, :],
                    generated.unsqueeze(0),
                    token_ids,
                    generation_mode="hybrid",
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                )
                token = int(sampled[0, 0].item())
                ar_fallback_sample_calls += 1
                steps.append(
                    {
                        "index": len(steps),
                        "mode": "ar_q1",
                        "history_len": history_len,
                        "accepted_token_ids": [token],
                    }
                )
                del probabilities, confidence, sampled, ar_logits
                ar_logits = None
                response.append(token)
                generated = torch.cat(
                    (
                        generated,
                        torch.tensor([token], device=self.device, dtype=torch.long),
                    )
                )
                is_coordinate = (
                    token_ids["coord_start_token_id"]
                    <= token
                    <= token_ids["coord_end_token_id"]
                )
                if not (
                    token == token_ids["box_end_token_id"]
                    or is_coordinate
                    or token == token_ids["none_token_id"]
                ):
                    stop_reason = "ar_non_coordinate"
                    break
                if token == token_ids["box_end_token_id"]:
                    # Upstream folds this token's causal K/V update into the
                    # following PBD window rather than running a standalone q1.
                    mode = "pbd"
                    pending_pbd = [token]
                    del ar_logits
                    ar_logits = None
                else:
                    ar_logits, new_keys, new_values = self._run_causal(
                        [token], keys, values, history_len, quantized
                    )
                    next_keys, next_values = self._retain_prefix(
                        keys, values, new_keys, new_values, 1
                    )
                    del keys, values, new_keys, new_values
                    keys, values = next_keys, next_values
                    history_len += 1
                    q1_commit_calls += 1
                    ar_logits = ar_logits[:, -1:]
                    mode = "ar_q1"
        finally:
            self.emulator.set_enabled(False)
            del keys, values
            if ar_logits is not None:
                del ar_logits

        return {
            "method": "official_hybrid_semantics_fused_prefix_mtp",
            "quantized": quantized,
            "seed": seed,
            "generation_config": config.as_dict(),
            "response_token_ids": response,
            "stop_reason": stop_reason,
            "generated_token_count": len(response),
            "prompt_token_count": prompt_len,
            "requested_new_token_limit": config.max_new_tokens,
            "requested_total_token_limit": requested_total_limit,
            "effective_total_token_limit": effective_total_limit,
            "tokenizer_model_max_length": self.model_max_length,
            "generation_limit_overshoot": max(
                0, prompt_len + len(response) - effective_total_limit
            ),
            "final_history_len": history_len,
            "pbd_calls": pbd_calls,
            "pbd_fused_prefix_calls": pbd_fused_prefix_calls,
            "q1_commit_calls": q1_commit_calls,
            "ar_fallback_sample_calls": ar_fallback_sample_calls,
            "steps": steps,
        }
