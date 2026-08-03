"""Language Float and eager-QDQ helpers for LocateAnything validation."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from compiler.scripts.common.coordinates import LanguageCoordinateAuditor
from compiler.scripts.common.quantization import QuantizationEmulator, tensor_comparison


CHUNK_SIZE = 1024
CACHE_LEN = 4096
PBD_QUERY_LEN = 6
AR_QUERY_LEN = 1
IMAGE_TOKEN_ID = 151665
VOCAB_SIZE = 152681

LANGUAGE_BC_GRAPHS = ("prefill", "decode", "decode_ar")
DECODER_WEIGHT_BITS = 8
LM_HEAD_WEIGHT_BITS = 8
DYNAMIC_A8_PATTERNS = (
    r"layers\.\d+\.self_attn\.qk_matmul\.(?:x|y)_fake_quant",
    r"layers\.\d+\.self_attn\.wv_matmul\.(?:x|y)_fake_quant",
    r"layers\.\d+\.self_attn\.cache_(?:k|v)_fq",
)

EMULATION_VERSION = 4
SCHEME = (
    "Decoder Linear W8 + lm_head W8 per-output-channel weights + "
    "per-row dynamic A8 Linear inputs + "
    "per-row dynamic symmetric A8 QK/WV operands and KV cache"
)
LIMITATION = (
    "Ideal PyTorch QDQ; HBDK accumulation, reciprocal LUT, scheduling, and "
    "target-specific rounding are not modeled"
)


def language_linear_weight_bits(module_name: str, default_bits: int) -> int:
    override = os.environ.get("LA_LANGUAGE_LINEAR_BITS", "").strip()
    if override:
        if override not in {"4", "8"}:
            raise ValueError("LA_LANGUAGE_LINEAR_BITS must be 4 or 8")
        return int(override)
    pattern = os.environ.get("LA_LANGUAGE_W8_REGEX", "").strip()
    if pattern and re.search(pattern, module_name):
        return 8
    return default_bits


def language_quantization_policy() -> dict[str, Any]:
    return {
        "decoder_weight_bits": DECODER_WEIGHT_BITS,
        "lm_head_weight_bits": LM_HEAD_WEIGHT_BITS,
        "linear_bits_override": os.environ.get(
            "LA_LANGUAGE_LINEAR_BITS", ""
        ).strip() or None,
        "w8_regex": os.environ.get("LA_LANGUAGE_W8_REGEX", "").strip() or None,
        "dynamic_a8_patterns": list(DYNAMIC_A8_PATTERNS),
    }


def create_language_model(
    model_path: Path,
    output_dir: Path,
    device: str,
) -> tuple[Any, Any, Any]:
    import torch
    from leap_llm.apis.model.locateanything_language import LocateAnythingLanguageApi
    from leap_llm.models.locateanything.hidden_rotation import load_hidden_rotation

    output_dir.mkdir(parents=True, exist_ok=True)
    api = LocateAnythingLanguageApi(
        str(model_path),
        str(output_dir),
        chunk_size=CHUNK_SIZE,
        cache_len=CACHE_LEN,
        decode_seq_len=PBD_QUERY_LEN,
        device=device,
        w_bits=DECODER_WEIGHT_BITS,
        lm_head_w_bits=LM_HEAD_WEIGHT_BITS,
        apply_hidden_rotation=True,
        export_only=True,
    )
    model = api.text_model.to(device=device, dtype=torch.float16).eval()
    model.compile_mode(False)
    rotation, _source = load_hidden_rotation(None, model.config.hidden_size)
    return api, model, rotation


def load_payload(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "projected_visual_features",
        "prediction_token_ids",
        "target_token_ids",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: missing Language fields {missing}")
    return payload


def resolve_language_bc_paths(model_path: Path, *, converted: bool) -> dict[str, Path]:
    """Resolve all graphs from a directory, or exactly one graph from a BC file."""

    model_path = model_path.resolve()
    conversion = "_convert" if converted else ""

    def suffix(graph: str) -> str:
        return f".{graph}{conversion}.bc"

    def bundle(base: Path) -> dict[str, Path]:
        return {graph: Path(f"{base}{suffix(graph)}") for graph in LANGUAGE_BC_GRAPHS}

    if model_path.is_dir():
        marker = suffix("prefill")
        candidates: list[dict[str, Path]] = []
        for prefill in sorted(model_path.rglob(f"*{marker}")):
            base = Path(str(prefill)[: -len(marker)])
            paths = bundle(base)
            if all(path.is_file() for path in paths.values()):
                candidates.append(paths)
        if not candidates:
            kind = "converted" if converted else "exported"
            raise FileNotFoundError(f"no complete {kind} Language BC bundle under {model_path}")
        if len(candidates) != 1:
            prefixes = [str(paths["prefill"])[: -len(marker)] for paths in candidates]
            raise ValueError(
                f"multiple Language BC bundles under {model_path}; pass any BC from one bundle: "
                f"{prefixes}"
            )
        return candidates[0]

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    matches = [
        (graph, kind, f".{graph}{kind}.bc")
        for graph in LANGUAGE_BC_GRAPHS
        for kind in ("", "_convert")
        if model_path.name.endswith(f".{graph}{kind}.bc")
    ]
    if not matches:
        raise ValueError(
            "Language BC filename must end in .prefill.bc, .decode.bc, "
            ".decode_ar.bc, or the corresponding _convert.bc suffix"
        )
    graph, kind, _ = max(matches, key=lambda item: len(item[2]))
    expected_kind = "_convert" if converted else ""
    if kind != expected_kind:
        expected = "converted" if converted else "exported"
        actual = "converted" if kind else "exported"
        raise ValueError(f"--mode selects {expected} BC, but {model_path.name} is {actual}")
    return {graph: model_path}


def _descriptor_shape(descriptor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in descriptor.type.shape)


def _descriptor_dtype(descriptor: Any):
    import numpy as np

    return np.dtype(descriptor.type.np_dtype)


def _as_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _coerce_bc_input(value: Any, descriptor: Any):
    import numpy as np

    array = _as_numpy(value)
    expected = _descriptor_shape(descriptor)
    if array.shape != expected:
        actual_without_singletons = tuple(size for size in array.shape if size != 1)
        expected_without_singletons = tuple(size for size in expected if size != 1)
        if array.size != int(np.prod(expected)) or actual_without_singletons != expected_without_singletons:
            raise ValueError(
                f"Language BC input {descriptor.name} expects {expected}, got {array.shape}"
            )
        array = array.reshape(expected)
    return np.ascontiguousarray(array.astype(_descriptor_dtype(descriptor), copy=False))


def language_bc_input_semantic(index: int, num_layers: int) -> str:
    if index == 0:
        return "inputs_embeds"
    if index == 1:
        return "position_ids"
    if index == 2:
        return "attention_mask"
    cache_index = index - 3
    if cache_index < num_layers:
        return f"layers.{cache_index}.key_cache"
    if cache_index < 2 * num_layers:
        return f"layers.{cache_index - num_layers}.value_cache"
    return f"input.{index}"


def language_bc_output_semantic(index: int, num_layers: int) -> str:
    if index == 0:
        return "logits"
    cache_index = index - 1
    if cache_index < num_layers:
        return f"layers.{cache_index}.new_key"
    if cache_index < 2 * num_layers:
        return f"layers.{cache_index - num_layers}.new_value"
    return f"output.{index}"


@dataclass
class LanguageBCArtifact:
    graph: str
    path: Path
    function: Any
    inputs: list[Any]
    outputs: list[Any]

    def describe(self, num_layers: int) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "path": str(self.path),
            "inputs": [
                {
                    "index": index,
                    "name": str(descriptor.name),
                    "semantic": language_bc_input_semantic(index, num_layers),
                    "shape": list(_descriptor_shape(descriptor)),
                    "dtype": str(_descriptor_dtype(descriptor)),
                }
                for index, descriptor in enumerate(self.inputs)
            ],
            "outputs": [
                {
                    "index": index,
                    "name": str(descriptor.name),
                    "semantic": language_bc_output_semantic(index, num_layers),
                    "shape": list(_descriptor_shape(descriptor)),
                    "dtype": str(_descriptor_dtype(descriptor)),
                }
                for index, descriptor in enumerate(self.outputs)
            ],
        }

    def run_outputs(self, values: list[Any]) -> list[Any]:
        import numpy as np

        if len(values) != len(self.inputs):
            raise ValueError(
                f"Language BC graph {self.graph} expects {len(self.inputs)} inputs, "
                f"got {len(values)}"
            )
        if not self.outputs:
            raise ValueError(f"Language BC graph {self.graph} has no outputs")
        feed = {
            str(descriptor.name): _coerce_bc_input(value, descriptor)
            for descriptor, value in zip(self.inputs, values, strict=True)
        }
        # BC functions accept ``inputs=`` while HBM graph simulation accepts
        # the same mapping as its first positional argument.  Keep the
        # descriptor-driven packing shared across both validation stages.
        try:
            raw = self.function.feed(inputs=feed)
        except TypeError as error:
            if "unexpected keyword argument 'inputs'" not in str(error):
                raise
            raw = self.function.feed(feed)
        missing = [
            str(descriptor.name)
            for descriptor in self.outputs
            if str(descriptor.name) not in raw
        ]
        if missing:
            raise KeyError(
                f"Language BC graph {self.graph} did not return outputs {missing}"
            )
        return [np.asarray(raw[str(descriptor.name)]) for descriptor in self.outputs]

    def run_logits(self, values: list[Any]):
        return self.run_outputs(values)[0]


def load_language_bc_artifacts(
    model_path: Path,
    *,
    converted: bool,
    loader: Callable[[str, Path, str], tuple[Any, list[Any], list[Any]]],
) -> dict[str, LanguageBCArtifact]:
    paths = resolve_language_bc_paths(model_path, converted=converted)
    artifacts: dict[str, LanguageBCArtifact] = {}
    for graph, path in paths.items():
        function, inputs, outputs = loader("bc", path, graph)
        artifacts[graph] = LanguageBCArtifact(graph, path, function, inputs, outputs)
    return artifacts


def _tensor_output(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"unsupported Language boundary output: {type(value).__name__}")


class LanguageBoundaryCapture:
    """Compare cumulative hidden states at all decoder-layer boundaries."""

    def __init__(self, model: Any, enabled: bool) -> None:
        self.enabled = enabled
        self.phase = "off"
        self.stage = "unassigned"
        self.active_tokens: int | None = None
        self.references: dict[tuple[str, str], Any] = {}
        self.rows: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        if not enabled:
            return
        modules = [(f"layers.{index}", layer) for index, layer in enumerate(model.layers)]
        # Logits are compared separately; retaining a 1024 x 152681 hook output
        # would add hundreds of MiB to every reference pass.
        modules.append(("norm", model.norm))
        for name, module in modules:
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def callback(_module: Any, _inputs: Any, output: Any) -> None:
            value = _tensor_output(output)
            if self.active_tokens is not None and value.ndim >= 2:
                value = value[:, : self.active_tokens]
            key = (self.stage, name)
            if self.phase == "reference":
                self.references[key] = value.detach().clone()
            elif self.phase == "candidate":
                reference = self.references.get(key)
                if reference is None:
                    raise RuntimeError(f"missing Language float boundary {self.stage}/{name}")
                self.rows.append({
                    "stage": self.stage,
                    "module": name,
                    "kind": "boundary",
                    "comparison": tensor_comparison(reference, value),
                })

        return callback

    def begin_reference(self, stage: str, active_tokens: int | None = None) -> None:
        self.stage = stage
        self.active_tokens = active_tokens
        self.phase = "reference" if self.enabled else "off"

    def begin_candidate(self, stage: str, active_tokens: int | None = None) -> None:
        self.stage = stage
        self.active_tokens = active_tokens
        self.phase = "candidate" if self.enabled else "off"

    def finish_sample(self) -> None:
        self.phase = "off"
        self.stage = "unassigned"
        self.active_tokens = None
        self.references.clear()
        self.rows = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.references.clear()


def _compact_logits(stage: str, logits: Any, active_len: int) -> Any:
    if stage == "prefill":
        return logits[:, active_len - 1].detach()
    return logits.detach()


def _unravel_position(index: int, shape: tuple[int, ...]) -> list[int]:
    position = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        position[axis] = index % shape[axis]
        index //= shape[axis]
    return position


def logits_comparison(
    reference: Any,
    candidate: Any,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare logits and retain the decision evidence hidden by cosine."""

    comparison = tensor_comparison(reference, candidate)
    if comparison.get("status") != "compared":
        return comparison
    if reference.ndim < 1 or reference.shape[-1] < 2:
        return comparison
    if top_k < 1:
        raise ValueError("top_k must be positive")

    left = reference.detach().float()
    right = candidate.detach().float()
    vocab_size = int(left.shape[-1])
    decision_shape = tuple(int(value) for value in left.shape[:-1])
    flat_left = left.reshape(-1, vocab_size)
    flat_right = right.reshape(-1, vocab_size)
    retained = min(top_k, vocab_size)

    reference_scores, reference_ids = flat_left.topk(retained, dim=-1)
    candidate_scores, candidate_ids = flat_right.topk(retained, dim=-1)
    reference_top1 = reference_ids[:, 0]
    candidate_top1 = candidate_ids[:, 0]
    reference_margin = reference_scores[:, 0] - reference_scores[:, 1]
    candidate_margin = candidate_scores[:, 0] - candidate_scores[:, 1]
    topk_overlap = (
        reference_ids.unsqueeze(-1) == candidate_ids.unsqueeze(-2)
    ).any(dim=-1).float().mean(dim=-1)

    reference_choice_in_reference = flat_left.gather(1, reference_top1[:, None])[:, 0]
    candidate_choice_in_reference = flat_left.gather(1, candidate_top1[:, None])[:, 0]
    reference_choice_in_candidate = flat_right.gather(1, reference_top1[:, None])[:, 0]
    candidate_choice_in_candidate = flat_right.gather(1, candidate_top1[:, None])[:, 0]
    reference_choice_gap = reference_choice_in_reference - candidate_choice_in_reference
    candidate_choice_gap = reference_choice_in_candidate - candidate_choice_in_candidate
    reference_rank_in_candidate = (
        (flat_right > reference_choice_in_candidate[:, None]).sum(dim=-1) + 1
    )
    candidate_rank_in_reference = (
        (flat_left > candidate_choice_in_reference[:, None]).sum(dim=-1) + 1
    )

    comparison.update({
        "top1_flip_rate": float((reference_top1 != candidate_top1).float().mean().item()),
        "topk": retained,
        "topk_overlap": float(topk_overlap.mean().item()),
        "reference_top1_margin": float(reference_margin.mean().item()),
        "candidate_top1_margin": float(candidate_margin.mean().item()),
        "reference_top1_rank_in_candidate": float(
            reference_rank_in_candidate.float().mean().item()
        ),
        "candidate_top1_rank_in_reference": float(
            candidate_rank_in_reference.float().mean().item()
        ),
    })

    cpu_values = {
        "reference_scores": reference_scores.cpu(),
        "reference_ids": reference_ids.cpu(),
        "candidate_scores": candidate_scores.cpu(),
        "candidate_ids": candidate_ids.cpu(),
        "reference_margin": reference_margin.cpu(),
        "candidate_margin": candidate_margin.cpu(),
        "topk_overlap": topk_overlap.cpu(),
        "reference_choice_gap": reference_choice_gap.cpu(),
        "candidate_choice_gap": candidate_choice_gap.cpu(),
        "reference_rank_in_candidate": reference_rank_in_candidate.cpu(),
        "candidate_rank_in_reference": candidate_rank_in_reference.cpu(),
    }
    comparison["decisions"] = [
        {
            "position": _unravel_position(index, decision_shape),
            "top1_flip": bool(
                cpu_values["reference_ids"][index, 0]
                != cpu_values["candidate_ids"][index, 0]
            ),
            "reference_top1_margin": float(cpu_values["reference_margin"][index]),
            "candidate_top1_margin": float(cpu_values["candidate_margin"][index]),
            "reference_choice_gap": float(cpu_values["reference_choice_gap"][index]),
            "candidate_choice_gap": float(cpu_values["candidate_choice_gap"][index]),
            "reference_top1_rank_in_candidate": int(
                cpu_values["reference_rank_in_candidate"][index]
            ),
            "candidate_top1_rank_in_reference": int(
                cpu_values["candidate_rank_in_reference"][index]
            ),
            "topk_overlap": float(cpu_values["topk_overlap"][index]),
            "reference_topk": [
                {
                    "token_id": int(token_id),
                    "logit": float(score),
                }
                for token_id, score in zip(
                    cpu_values["reference_ids"][index].tolist(),
                    cpu_values["reference_scores"][index].tolist(),
                )
            ],
            "candidate_topk": [
                {
                    "token_id": int(token_id),
                    "logit": float(score),
                }
                for token_id, score in zip(
                    cpu_values["candidate_ids"][index].tolist(),
                    cpu_values["candidate_scores"][index].tolist(),
                )
            ],
        }
        for index in range(flat_left.shape[0])
    ]
    return comparison


def _clone_cache_tensors(caches: list[Any]) -> list[Any]:
    return [value.clone() for value in caches]


def _compare_outputs(
    stage: str,
    reference: tuple[Any, list[Any], list[Any]],
    candidate: tuple[Any, list[Any], list[Any]],
    active_len: int,
) -> tuple[dict[str, Any], Any]:
    reference_logits, reference_keys, reference_values = reference
    candidate_logits, candidate_keys, candidate_values = candidate
    compact_reference = _compact_logits(stage, reference_logits, active_len)
    compact_candidate = _compact_logits(stage, candidate_logits, active_len)
    rows = [{
        "stage": stage,
        "module": "logits",
        "kind": "output",
        "comparison": logits_comparison(compact_reference, compact_candidate),
    }]
    if stage == "pbd_q6":
        rows.extend(
            {
                "stage": stage,
                "module": f"logits.token_{index}",
                "kind": "output",
                "comparison": logits_comparison(
                    compact_reference[:, index], compact_candidate[:, index]
                ),
            }
            for index in range(compact_reference.shape[1])
        )
    valid_tokens = active_len if stage == "prefill" else reference_keys[0].shape[1]
    for index, (ref_key, cand_key, ref_value, cand_value) in enumerate(
        zip(reference_keys, candidate_keys, reference_values, candidate_values)
    ):
        rows.append({
            "stage": stage,
            "module": f"layers.{index}.key",
            "kind": "kv_output",
            "comparison": tensor_comparison(
                ref_key[:, :valid_tokens], cand_key[:, :valid_tokens]
            ),
        })
        rows.append({
            "stage": stage,
            "module": f"layers.{index}.value",
            "kind": "kv_output",
            "comparison": tensor_comparison(
                ref_value[:, :valid_tokens], cand_value[:, :valid_tokens]
            ),
        })
    return {"rows": rows, "reference_logits": compact_reference}, compact_candidate


def _select_tokens(payload: dict[str, Any], mode: str, length: int) -> list[int]:
    from leap_llm.apis.calibration.locateanything_replay import select_decode_tokens

    return select_decode_tokens(payload, mode, length)


def _select_pbd_tokens(payload: dict[str, Any], model: Any) -> list[int]:
    from leap_llm.apis.calibration.locateanything_replay import select_pbd_tokens

    return select_pbd_tokens(
        payload,
        PBD_QUERY_LEN,
        int(model.config.text_mask_token_id),
    )


@dataclass
class LanguageRunResult:
    outputs: dict[str, Any]
    comparisons: list[dict[str, Any]]
    boundaries: list[dict[str, Any]]
    operators: list[dict[str, Any]]


class LanguageEagerRunner:
    def __init__(
        self,
        model: Any,
        rotation: Any,
        device: str,
        *,
        quantized: bool,
        capture_boundaries: bool,
        capture_operators: bool,
        rescue_policy: Any | None = None,
        dynamic_quantizer_patterns: list[str] | None = None,
    ) -> None:
        import torch

        self.model = model
        self.rotation = rotation
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.quantized = quantized
        self.emulator = (
            QuantizationEmulator(
                model,
                capture_operators=capture_operators,
                dynamic_attention_quantizers=False,
                linear_weight_bits=language_linear_weight_bits,
                rescue_policy=rescue_policy,
            )
            if quantized else None
        )
        if self.emulator is not None:
            self.emulator.set_dynamic_quantizer_patterns(
                list(
                    DYNAMIC_A8_PATTERNS
                    if dynamic_quantizer_patterns is None
                    else dynamic_quantizer_patterns
                )
            )
        self.boundaries = LanguageBoundaryCapture(model, capture_boundaries and quantized)
        num_kv = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.zero_caches = [torch.zeros(
            (1, CACHE_LEN, num_kv, head_dim),
            device=self.device,
            dtype=self.dtype,
        ) for _ in range(model.config.num_hidden_layers * 2)]

    def _prefill_inputs(self, payload: dict[str, Any]):
        from leap_llm.apis.calibration.locateanything_replay import build_prefill_inputs

        return build_prefill_inputs(
            self.model,
            payload,
            self.rotation,
            chunk_size=CHUNK_SIZE,
            cache_len=CACHE_LEN,
            image_token_id=IMAGE_TOKEN_ID,
            device=self.device,
            dtype=self.dtype,
        )

    def _decode_inputs(
        self,
        token_ids: list[int],
        *,
        q_len: int,
        past_len: int,
        is_pbd: bool,
    ):
        from leap_llm.apis.calibration.locateanything_replay import build_decode_inputs

        return build_decode_inputs(
            self.model,
            token_ids,
            q_len=q_len,
            past_len=past_len,
            cache_len=CACHE_LEN,
            is_pbd=is_pbd,
            device=self.device,
            dtype=self.dtype,
        )

    def _right_aligned(self, keys: list[Any], values: list[Any], active_len: int):
        from leap_llm.apis.calibration.locateanything_replay import build_right_aligned_caches

        return build_right_aligned_caches(
            keys, values, active_len=active_len, cache_len=CACHE_LEN
        )

    def _pair(
        self,
        stage: str,
        inputs: tuple[Any, Any, Any],
        caches: list[Any],
        active_len: int,
    ) -> tuple[
        tuple[Any, list[Any], list[Any]],
        tuple[Any, list[Any], list[Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        Any,
    ]:
        assert self.emulator is not None
        self.emulator.set_stage(stage)
        self.emulator.set_enabled(False)
        self.boundaries.begin_reference(stage, active_len)
        reference = self.model(*inputs, *_clone_cache_tensors(caches))
        self.boundaries.begin_candidate(stage, active_len)
        self.emulator.set_enabled(True)
        candidate = self.model(*inputs, *_clone_cache_tensors(caches))
        comparison, compact_candidate = _compare_outputs(
            stage, reference, candidate, active_len
        )
        return (
            reference,
            candidate,
            comparison["rows"],
            list(self.emulator.rows),
            compact_candidate,
        )

    def set_rescue_policy(self, rescue_policy: Any | None) -> None:
        if self.emulator is None:
            raise RuntimeError("Float runner has no quantization rescue policy")
        self.emulator.set_rescue_policy(rescue_policy)

    def set_dynamic_quantizer_patterns(self, patterns: list[str]) -> dict[str, Any]:
        if self.emulator is None:
            raise RuntimeError("Float runner has no quantization emulator")
        return self.emulator.set_dynamic_quantizer_patterns(patterns)

    def audit_coordinates(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.emulator is None:
            raise RuntimeError("coordinate audit requires a quantized eager runner")
        return LanguageCoordinateAuditor(
            self.model,
            self.rotation,
            self.device,
            self.dtype,
            self.zero_caches,
            self.emulator,
            self.boundaries,
            chunk_size=CHUNK_SIZE,
            cache_len=CACHE_LEN,
            pbd_query_len=PBD_QUERY_LEN,
            image_token_id=IMAGE_TOKEN_ID,
            batch_size=1 if self.emulator.capture_operators else 16,
        ).run(payload)

    def run(self, payload: dict[str, Any]) -> LanguageRunResult:
        import torch

        outputs: dict[str, Any] = {}
        comparisons: list[dict[str, Any]] = []
        operator_rows: list[dict[str, Any]] = []
        prefill_inputs = self._prefill_inputs(payload)
        embeds, positions, mask, active_len = prefill_inputs
        with torch.no_grad():
            if not self.quantized:
                logits, keys, values = self.model(
                    embeds, positions, mask, *self.zero_caches
                )
                outputs["prefill_logits"] = _compact_logits(
                    "prefill", logits, active_len
                ).float().cpu().numpy()
                cache_keys, cache_values = self._right_aligned(keys, values, active_len)
            else:
                reference, candidate, rows, stage_operators, compact = self._pair(
                    "prefill", (embeds, positions, mask), self.zero_caches, active_len
                )
                comparisons.extend(rows)
                operator_rows.extend(stage_operators)
                outputs["prefill_logits"] = compact.float().cpu().numpy()
                ref_logits, ref_keys, ref_values = reference
                cand_logits, _, _ = candidate
                reference_cache = self._right_aligned(ref_keys, ref_values, active_len)
                del ref_logits, cand_logits

            pbd_tokens = _select_pbd_tokens(payload, self.model)
            pbd_inputs = self._decode_inputs(
                pbd_tokens,
                q_len=PBD_QUERY_LEN,
                past_len=active_len,
                is_pbd=True,
            )
            if not self.quantized:
                pbd_logits, _, _ = self.model(
                    *pbd_inputs, *(cache_keys + cache_values)
                )
                outputs["pbd_logits"] = pbd_logits.float().cpu().numpy()
            else:
                reference, candidate, rows, stage_operators, compact = self._pair(
                    "pbd_q6",
                    pbd_inputs,
                    reference_cache[0] + reference_cache[1],
                    PBD_QUERY_LEN,
                )
                comparisons.extend(rows)
                operator_rows.extend(stage_operators)
                outputs["pbd_logits"] = compact.float().cpu().numpy()
                del reference, candidate

            if "slow" in payload["prediction_token_ids"]:
                ar_tokens = _select_tokens(payload, "slow", AR_QUERY_LEN)
                ar_inputs = self._decode_inputs(
                    ar_tokens,
                    q_len=AR_QUERY_LEN,
                    past_len=active_len,
                    is_pbd=False,
                )
                if not self.quantized:
                    ar_logits, _, _ = self.model(
                        *ar_inputs, *(cache_keys + cache_values)
                    )
                    outputs["ar_logits"] = ar_logits.float().cpu().numpy()
                else:
                    reference, candidate, rows, stage_operators, compact = self._pair(
                        "ar_q1",
                        ar_inputs,
                        reference_cache[0] + reference_cache[1],
                        AR_QUERY_LEN,
                    )
                    comparisons.extend(rows)
                    operator_rows.extend(stage_operators)
                    outputs["ar_logits"] = compact.float().cpu().numpy()
                    del reference, candidate

        if self.emulator is not None:
            self.emulator.set_enabled(False)
        boundary_rows = list(self.boundaries.rows)
        self.boundaries.finish_sample()
        return LanguageRunResult(outputs, comparisons, boundary_rows, operator_rows)

    def close(self) -> None:
        self.boundaries.close()
        if self.emulator is not None:
            self.emulator.close()
        self.zero_caches.clear()


def _compact_bc_logits(graph: str, logits: Any, active_len: int):
    import numpy as np

    value = np.asarray(logits)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"Language BC {graph} logits must be [1,q,vocab], got {value.shape}")
    if value.shape[-1] < VOCAB_SIZE:
        raise ValueError(
            f"Language BC {graph} logits width {value.shape[-1]} is below vocab {VOCAB_SIZE}"
        )
    value = value[..., :VOCAB_SIZE]
    if graph == "prefill":
        if value.shape[1] == 1:
            return np.ascontiguousarray(value[:, 0])
        if active_len < 1 or active_len > value.shape[1]:
            raise ValueError(
                f"Language BC prefill active length {active_len} is outside q={value.shape[1]}"
            )
        return np.ascontiguousarray(value[:, active_len - 1])
    expected_q = PBD_QUERY_LEN if graph == "decode" else AR_QUERY_LEN
    if value.shape[1] != expected_q:
        raise ValueError(
            f"Language BC {graph} logits q={value.shape[1]}, expected {expected_q}"
        )
    return np.ascontiguousarray(value)


@dataclass
class LanguageBCRunResult:
    outputs: dict[str, Any]
    execution: dict[str, Any]
    timings: dict[str, float]


class LanguageBCRunner:
    """Execute independent Language BC graphs with Float-prefill decode caches."""

    def __init__(
        self,
        model: Any,
        rotation: Any,
        device: str,
        artifacts: dict[str, LanguageBCArtifact],
        cache_provider: LanguageBCArtifact | None = None,
    ) -> None:
        import torch

        unknown = sorted(set(artifacts) - set(LANGUAGE_BC_GRAPHS))
        if not artifacts or unknown:
            raise ValueError(f"invalid Language BC graph selection: {sorted(artifacts)}")
        self.model = model
        self.rotation = rotation
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.artifacts = artifacts
        self.cache_provider = cache_provider
        self.num_layers = int(model.config.num_hidden_layers)
        self.expected_inputs = 3 + 2 * self.num_layers
        for graph, artifact in artifacts.items():
            if len(artifact.inputs) != self.expected_inputs:
                raise ValueError(
                    f"Language BC graph {graph} has {len(artifact.inputs)} inputs; "
                    f"expected {self.expected_inputs}"
                )
        if self.cache_provider is not None:
            if self.cache_provider.graph != "prefill":
                raise ValueError("Language BC cache provider must be the prefill graph")
            if len(self.cache_provider.inputs) != self.expected_inputs:
                raise ValueError(
                    "Language BC prefill cache provider has "
                    f"{len(self.cache_provider.inputs)} inputs; expected {self.expected_inputs}"
                )
        num_kv = int(model.config.num_key_value_heads)
        head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
        zero = torch.zeros(
            (1, CACHE_LEN, num_kv, head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        # The eager model never mutates cache inputs, so one shared zero tensor is sufficient.
        self.zero_caches = [zero] * (2 * self.num_layers)
        self._bc_zero_arrays: dict[tuple[tuple[int, ...], str], Any] = {}

    def describe_artifacts(self) -> dict[str, Any]:
        return {
            graph: artifact.describe(self.num_layers)
            for graph, artifact in self.artifacts.items()
        }

    def _prefill_inputs(self, payload: dict[str, Any]):
        from leap_llm.apis.calibration.locateanything_replay import build_prefill_inputs

        return build_prefill_inputs(
            self.model,
            payload,
            self.rotation,
            chunk_size=CHUNK_SIZE,
            cache_len=CACHE_LEN,
            image_token_id=IMAGE_TOKEN_ID,
            device=self.device,
            dtype=self.dtype,
        )

    def _decode_inputs(
        self,
        token_ids: list[int],
        *,
        q_len: int,
        past_len: int,
        is_pbd: bool,
    ):
        from leap_llm.apis.calibration.locateanything_replay import build_decode_inputs

        return build_decode_inputs(
            self.model,
            token_ids,
            q_len=q_len,
            past_len=past_len,
            cache_len=CACHE_LEN,
            is_pbd=is_pbd,
            device=self.device,
            dtype=self.dtype,
        )

    def _right_aligned(self, keys: list[Any], values: list[Any], active_len: int):
        from leap_llm.apis.calibration.locateanything_replay import build_right_aligned_caches

        return build_right_aligned_caches(
            keys, values, active_len=active_len, cache_len=CACHE_LEN
        )

    def _bc_zeros(self, artifact: LanguageBCArtifact) -> list[Any]:
        zeros: list[Any] = []
        for descriptor in artifact.inputs[3:]:
            key = (_descriptor_shape(descriptor), str(_descriptor_dtype(descriptor)))
            if key not in self._bc_zero_arrays:
                import numpy as np

                self._bc_zero_arrays[key] = np.zeros(key[0], dtype=_descriptor_dtype(descriptor))
            zeros.append(self._bc_zero_arrays[key])
        return zeros

    def _cache_arrays(
        self,
        keys: list[Any],
        values: list[Any],
        artifact: LanguageBCArtifact,
    ) -> list[Any]:
        caches = [*keys, *values]
        descriptors = artifact.inputs[3:]
        if len(caches) != len(descriptors):
            raise ValueError(
                f"Float Prefill produced {len(caches)} caches, BC expects {len(descriptors)}"
            )
        return [
            _coerce_bc_input(cache, descriptor)
            for cache, descriptor in zip(caches, descriptors, strict=True)
        ]

    def _right_aligned_bc_caches(
        self,
        updates: list[Any],
        active_len: int,
        artifact: LanguageBCArtifact,
    ) -> list[Any]:
        import numpy as np

        descriptors = artifact.inputs[3:]
        if len(updates) != len(descriptors):
            raise ValueError(
                f"Language BC Prefill produced {len(updates)} cache updates, "
                f"Decode expects {len(descriptors)} caches"
            )
        caches: list[Any] = []
        for update, descriptor in zip(updates, descriptors, strict=True):
            array = _as_numpy(update)
            expected = _descriptor_shape(descriptor)
            if array.ndim != 4 or len(expected) != 4:
                raise ValueError(
                    f"Language BC cache {descriptor.name} must be rank 4, "
                    f"got update={array.shape} input={expected}"
                )
            if (
                array.shape[0] != expected[0]
                or array.shape[2:] != expected[2:]
                or active_len < 1
                or active_len > array.shape[1]
                or active_len > expected[1]
            ):
                raise ValueError(
                    f"Language BC cache {descriptor.name} cannot right-align "
                    f"update={array.shape}, active_len={active_len}, input={expected}"
                )
            cache = np.zeros(expected, dtype=_descriptor_dtype(descriptor))
            cache[:, -active_len:] = array[:, :active_len].astype(
                cache.dtype, copy=False
            )
            caches.append(np.ascontiguousarray(cache))
        return caches

    def _execute(
        self,
        graph: str,
        inputs: tuple[Any, Any, Any],
        caches: list[Any],
        active_len: int,
    ) -> tuple[Any, float]:
        started = time.monotonic()
        logits = self.artifacts[graph].run_logits([*inputs, *caches])
        elapsed = time.monotonic() - started
        return _compact_bc_logits(graph, logits, active_len), elapsed

    def run(self, payload: dict[str, Any]) -> LanguageBCRunResult:
        import numpy as np
        import torch

        outputs: dict[str, Any] = {}
        execution: dict[str, Any] = {}
        timings: dict[str, float] = {}
        embeds, positions, mask, active_len = self._prefill_inputs(payload)
        prefill_inputs = (embeds, positions, mask)

        with torch.no_grad():
            cache_keys = cache_values = None
            cache_artifact = self.artifacts.get("decode") or self.artifacts.get("decode_ar")
            compiled_prefill = self.artifacts.get("prefill") or self.cache_provider
            quantized_cache = bool(
                cache_artifact is not None
                and not np.issubdtype(
                    _descriptor_dtype(cache_artifact.inputs[3]), np.floating
                )
            )
            compiled_prefill_outputs = None
            if compiled_prefill is not None and (
                "prefill" in self.artifacts or quantized_cache
            ):
                compiled_started = time.monotonic()
                compiled_prefill_outputs = compiled_prefill.run_outputs(
                    [
                        *prefill_inputs,
                        *self._bc_zeros(compiled_prefill),
                    ]
                )
                timings["compiled_prefill_seconds"] = (
                    time.monotonic() - compiled_started
                )

            if (
                ("decode" in self.artifacts or "decode_ar" in self.artifacts)
                and not quantized_cache
            ):
                float_started = time.monotonic()
                _, keys, values = self.model(*prefill_inputs, *self.zero_caches)
                cache_keys, cache_values = self._right_aligned(keys, values, active_len)
                timings["float_prefill_cache_seconds"] = time.monotonic() - float_started

            if "prefill" in self.artifacts:
                assert compiled_prefill_outputs is not None
                prefill_output = _compact_bc_logits(
                    "prefill", compiled_prefill_outputs[0], active_len
                )
                outputs["prefill_logits"] = prefill_output
                timings["prefill_bc_seconds"] = timings["compiled_prefill_seconds"]
                execution["prefill"] = {
                    "graph": "prefill",
                    "output": "prefill_logits",
                    "cache_source": "zero",
                    "active_len": active_len,
                    "logits_selection": "last active prompt token",
                }

            decode_caches = None
            if cache_artifact is not None:
                if quantized_cache:
                    if compiled_prefill_outputs is None:
                        raise ValueError(
                            "quantized Language Decode cache requires the matching "
                            "compiled Prefill graph; refusing to cast Float KV directly"
                        )
                    decode_caches = self._right_aligned_bc_caches(
                        compiled_prefill_outputs[1:], active_len, cache_artifact
                    )
                else:
                    assert cache_keys is not None and cache_values is not None
                    decode_caches = self._cache_arrays(
                        cache_keys, cache_values, cache_artifact
                    )

            if "decode" in self.artifacts:
                assert decode_caches is not None
                pbd_inputs = self._decode_inputs(
                    _select_pbd_tokens(payload, self.model),
                    q_len=PBD_QUERY_LEN,
                    past_len=active_len,
                    is_pbd=True,
                )
                pbd_output, elapsed = self._execute(
                    "decode", pbd_inputs, decode_caches, active_len
                )
                outputs["pbd_logits"] = pbd_output
                timings["pbd_bc_seconds"] = elapsed
                execution["pbd_q6"] = {
                    "graph": "decode",
                    "output": "pbd_logits",
                    "cache_source": (
                        "compiled_prefill" if quantized_cache else "float_prefill"
                    ),
                    "q_len": PBD_QUERY_LEN,
                    "past_len": active_len,
                    "purpose": (
                        "replay compiled Prefill-to-Decode KV contract"
                        if quantized_cache
                        else "isolate PBD BC error from Prefill BC error"
                    ),
                }

            if "decode_ar" in self.artifacts and "slow" in payload["prediction_token_ids"]:
                assert decode_caches is not None
                ar_inputs = self._decode_inputs(
                    _select_tokens(payload, "slow", AR_QUERY_LEN),
                    q_len=AR_QUERY_LEN,
                    past_len=active_len,
                    is_pbd=False,
                )
                ar_output, elapsed = self._execute(
                    "decode_ar", ar_inputs, decode_caches, active_len
                )
                outputs["ar_logits"] = ar_output
                timings["ar_bc_seconds"] = elapsed
                execution["ar_q1"] = {
                    "graph": "decode_ar",
                    "output": "ar_logits",
                    "cache_source": (
                        "compiled_prefill" if quantized_cache else "float_prefill"
                    ),
                    "q_len": AR_QUERY_LEN,
                    "past_len": active_len,
                    "purpose": (
                        "replay compiled Prefill-to-Decode KV contract"
                        if quantized_cache
                        else "isolate AR BC error from Prefill BC error"
                    ),
                }

        return LanguageBCRunResult(outputs, execution, timings)

    def close(self) -> None:
        self.zero_caches.clear()
        self._bc_zero_arrays.clear()
