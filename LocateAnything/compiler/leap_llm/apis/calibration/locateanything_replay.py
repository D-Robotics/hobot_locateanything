"""Activation-statistics replay helpers for LocateAnything calibration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


REQUIRED_TENSOR_FIELDS = {
    "prompt_input_ids",
    "prompt_attention_mask",
    "vision_input",
    "projected_visual_features",
    "prediction_token_ids",
    "target_token_ids",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_generated_manifest(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = path.resolve()
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if record.get("status") != "complete":
                raise ValueError(f"{path}:{line_number}: record is not complete")
            tensor_path = (path.parent / record["tensor_file"]).resolve()
            if not tensor_path.is_file():
                raise FileNotFoundError(tensor_path)
            record["_tensor_path"] = tensor_path
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"empty generated manifest: {path}")
    return records


def load_tensor_payload(record: dict[str, Any]) -> dict[str, Any]:
    tensor_path = Path(record["_tensor_path"])
    if sha256_file(tensor_path) != record["tensor_sha256"]:
        raise ValueError(f"tensor SHA256 mismatch: {tensor_path}")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    missing = sorted(REQUIRED_TENSOR_FIELDS - set(payload))
    if missing:
        raise ValueError(f"{tensor_path}: missing fields {missing}")
    return payload


def build_prefill_inputs(
    text_model: Any,
    payload: dict[str, Any],
    rotation: torch.Tensor,
    *,
    chunk_size: int,
    cache_len: int,
    image_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -32768.0,
    suffix_token_ids: Iterable[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    input_ids = payload["prompt_input_ids"].reshape(-1).to(torch.long)
    active_len = int(payload["prompt_attention_mask"].reshape(-1).sum().item())
    if active_len != input_ids.numel():
        raise ValueError("D3 prompt must be unpadded before fixed-profile replay")
    prompt_len = active_len
    suffix = [int(token_id) for token_id in suffix_token_ids or ()]
    if suffix:
        input_ids = torch.cat(
            (input_ids, torch.tensor(suffix, dtype=torch.long)), dim=0
        )
        active_len += len(suffix)
    if active_len > chunk_size:
        raise ValueError(f"active length {active_len} exceeds chunk size {chunk_size}")
    image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    image_mask[:prompt_len] = input_ids[:prompt_len] == image_token_id
    image_count = int(image_mask.sum().item())
    projected = payload["projected_visual_features"].reshape(-1, rotation.shape[0]).float()
    if image_count != projected.shape[0]:
        raise ValueError(
            f"image placeholder count {image_count} != projected tokens {projected.shape[0]}"
        )

    ids_device = input_ids.to(device)
    active = text_model.embed_tokens(ids_device).to(dtype=dtype)
    rotated_visual = (projected @ rotation.float()).to(device=device, dtype=dtype)
    active[image_mask.to(device)] = rotated_visual

    inputs_embeds = torch.zeros(
        (1, chunk_size, active.shape[-1]), device=device, dtype=dtype
    )
    inputs_embeds[0, :active_len] = active
    position_ids = torch.arange(chunk_size, device=device, dtype=torch.int32).view(1, 1, -1)

    current_start = cache_len - chunk_size
    mask = torch.full(
        (1, 1, chunk_size, cache_len), mask_value, device=device, dtype=dtype
    )
    for query in range(chunk_size):
        visible = min(query + 1, active_len)
        if visible:
            mask[:, :, query, current_start : current_start + visible] = 0
        if query >= active_len:
            mask[:, :, query, current_start + query] = 0
    return inputs_embeds, position_ids, mask, active_len


def build_right_aligned_caches(
    new_keys: Iterable[torch.Tensor],
    new_values: Iterable[torch.Tensor],
    *,
    active_len: int,
    cache_len: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for new_key, new_value in zip(new_keys, new_values):
        key_cache = torch.zeros(
            (new_key.shape[0], cache_len, new_key.shape[2], new_key.shape[3]),
            device=new_key.device,
            dtype=new_key.dtype,
        )
        value_cache = torch.zeros_like(key_cache)
        key_cache[:, -active_len:] = new_key[:, :active_len]
        value_cache[:, -active_len:] = new_value[:, :active_len]
        keys.append(key_cache)
        values.append(value_cache)
    return keys, values


def build_decode_inputs(
    text_model: Any,
    token_ids: list[int],
    *,
    q_len: int,
    past_len: int,
    cache_len: int,
    is_pbd: bool,
    pbd_prefix_len: int = 0,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -32768.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(token_ids) < q_len:
        raise ValueError(f"need {q_len} decode tokens, got {len(token_ids)}")
    if pbd_prefix_len < 0 or pbd_prefix_len > q_len:
        raise ValueError(
            f"invalid PBD prefix length {pbd_prefix_len} for q_len {q_len}"
        )
    if pbd_prefix_len and not is_pbd:
        raise ValueError("PBD prefix requires is_pbd=True")
    ids = torch.tensor(token_ids[:q_len], device=device, dtype=torch.long)
    embeds = text_model.embed_tokens(ids).to(dtype=dtype).unsqueeze(0)
    positions = torch.arange(past_len, past_len + q_len, device=device, dtype=torch.int32)
    if is_pbd:
        # The suffix is the upstream MTP window: duplicate its anchor at the
        # anchor position, then place the five mask tokens after it. A leading
        # prefix consists of real tokens accepted in the prior MTP round.
        positions[pbd_prefix_len:] -= 1
    position_ids = positions.view(1, 1, q_len)

    mask = torch.full((1, 1, q_len, cache_len), mask_value, device=device, dtype=dtype)
    current_start = cache_len - q_len
    history_start = current_start - past_len
    if history_start < 0:
        raise ValueError(f"past_len {past_len} does not fit cache_len {cache_len}")
    mask[:, :, :, history_start:current_start] = 0
    for query in range(q_len):
        mask[:, :, query, current_start : current_start + query + 1] = 0
    if is_pbd:
        # Only the trailing PBD window is bidirectional. The optional prefix
        # remains causal so its K/V rows are identical to ordinary AR decode.
        window_start = current_start + pbd_prefix_len
        mask[:, :, pbd_prefix_len:, window_start : current_start + q_len] = 0
    if is_pbd and past_len:
        # Upstream masks [-block_size-1] for the trailing MTP rows. With a
        # fused accepted prefix this column belongs to the prefix, not to the
        # cached history immediately before the whole q_len window.
        previous_round_tail = current_start + pbd_prefix_len - 1
        mask[:, :, pbd_prefix_len:, previous_round_tail] = mask_value
    return embeds, position_ids, mask


def select_decode_tokens(payload: dict[str, Any], mode: str, q_len: int) -> list[int]:
    predictions = payload["prediction_token_ids"]
    candidates = predictions.get(mode)
    values = candidates.reshape(-1).tolist() if torch.is_tensor(candidates) else []
    target = payload["target_token_ids"].reshape(-1).tolist()
    combined = [int(value) for value in values + target]
    if not combined:
        raise ValueError(f"no tokens available for {mode} replay")
    while len(combined) < q_len:
        combined.extend(combined[: q_len - len(combined)])
    return combined[:q_len]


def select_pbd_tokens(
    payload: dict[str, Any], q_len: int, text_mask_token_id: int
) -> list[int]:
    """Build the native PBD window: history anchor followed by mask tokens."""

    if q_len < 1:
        raise ValueError("PBD q_len must be positive")
    input_ids = payload["prompt_input_ids"].reshape(-1)
    attention_mask = payload["prompt_attention_mask"].reshape(-1)
    active_len = int(attention_mask.sum().item())
    if active_len < 1 or active_len > input_ids.numel():
        raise ValueError(f"invalid active prompt length for PBD: {active_len}")
    anchor = int(input_ids[active_len - 1].item())
    return [anchor, *([int(text_mask_token_id)] * (q_len - 1))]


def select_replay_prefix_tokens(
    payload: dict[str, Any], prefix_len: int
) -> list[int]:
    """Select a deterministic, box-aligned pending prefix for decode replay."""

    if prefix_len < 1:
        raise ValueError("replay prefix length must be positive")
    target = payload["target_token_ids"].reshape(-1).tolist()
    target = [int(value) for value in target]
    special = payload.get("special_token_ids") or {}
    box_start = int(special.get("<box>", 151668))
    starts = [
        index
        for index, token in enumerate(target)
        if token == box_start and index + prefix_len <= len(target)
    ]
    if starts:
        identity = str(payload.get("bundle_id") or "").encode("utf-8")
        choice = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
        start = starts[choice % len(starts)]
        return target[start : start + prefix_len]
    return select_decode_tokens(payload, "hybrid", prefix_len)


_LOG_HISTOGRAM_MIN_EXPONENT = -32.0
_LOG_HISTOGRAM_MAX_EXPONENT = 32.0
_LOG_HISTOGRAM_BINS = 512


def _input_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _input_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _input_tensors(nested)


class _StreamingActivationStatistics:
    """Fixed-memory statistics for one activation point in one graph stage."""

    def __init__(self) -> None:
        self.execution_count = 0
        self.observed_elements = 0
        self.finite_elements = 0
        self.nonfinite_count = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.absmax = -math.inf
        self.max_hit_count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.zero_abs_count = 0
        self.histogram = torch.zeros(_LOG_HISTOGRAM_BINS, dtype=torch.int64)
        self.fixed_clipping_range: float | None = None
        self.fixed_clipping_count = 0

    def record_execution(self) -> None:
        self.execution_count += 1

    def update(self, value: torch.Tensor, fixed_clipping_range: float | None = None) -> None:
        value = value.detach()
        total = value.numel()
        self.observed_elements += total
        if not total:
            return

        finite_mask = torch.isfinite(value)
        finite_count = int(finite_mask.sum().item())
        self.finite_elements += finite_count
        self.nonfinite_count += total - finite_count
        if not finite_count:
            return

        finite = value[finite_mask].float()
        batch_min = float(finite.min().item())
        batch_max = float(finite.max().item())
        absolute = finite.abs()
        batch_absmax = float(absolute.max().item())
        batch_max_hits = int((absolute == batch_absmax).sum().item())

        self.minimum = min(self.minimum, batch_min)
        self.maximum = max(self.maximum, batch_max)
        if batch_absmax > self.absmax:
            self.absmax = batch_absmax
            self.max_hit_count = batch_max_hits
        elif batch_absmax == self.absmax:
            self.max_hit_count += batch_max_hits

        batch_mean_tensor = finite.mean()
        batch_mean = float(batch_mean_tensor.item())
        batch_m2 = float(((finite - batch_mean_tensor) ** 2).sum().item())
        previous_count = self.finite_elements - finite_count
        if previous_count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            combined_count = previous_count + finite_count
            delta = batch_mean - self.mean
            self.mean += delta * finite_count / combined_count
            self.m2 += (
                batch_m2
                + delta * delta * previous_count * finite_count / combined_count
            )

        positive = absolute[absolute > 0]
        self.zero_abs_count += finite_count - positive.numel()
        if positive.numel():
            width = _LOG_HISTOGRAM_MAX_EXPONENT - _LOG_HISTOGRAM_MIN_EXPONENT
            indices = torch.floor(
                (torch.log2(positive) - _LOG_HISTOGRAM_MIN_EXPONENT)
                * (_LOG_HISTOGRAM_BINS / width)
            ).to(torch.int64)
            indices.clamp_(0, _LOG_HISTOGRAM_BINS - 1)
            counts = torch.bincount(indices, minlength=_LOG_HISTOGRAM_BINS)
            self.histogram += counts.detach().to(device="cpu", dtype=torch.int64)

        if fixed_clipping_range is not None:
            clipping_range = float(fixed_clipping_range)
            if self.fixed_clipping_range is None:
                self.fixed_clipping_range = clipping_range
            elif not math.isclose(self.fixed_clipping_range, clipping_range):
                raise ValueError("fixed clipping range changed during activation replay")
            self.fixed_clipping_count += int((absolute > clipping_range).sum().item())

    def merge(self, other: "_StreamingActivationStatistics") -> None:
        self.execution_count += other.execution_count
        self.observed_elements += other.observed_elements
        self.nonfinite_count += other.nonfinite_count
        if not other.finite_elements:
            return
        if not self.finite_elements:
            self.finite_elements = other.finite_elements
            self.minimum = other.minimum
            self.maximum = other.maximum
            self.absmax = other.absmax
            self.max_hit_count = other.max_hit_count
            self.mean = other.mean
            self.m2 = other.m2
            self.zero_abs_count = other.zero_abs_count
            self.histogram += other.histogram
            self.fixed_clipping_range = other.fixed_clipping_range
            self.fixed_clipping_count = other.fixed_clipping_count
            return

        first_count = self.finite_elements
        combined_count = first_count + other.finite_elements
        delta = other.mean - self.mean
        self.mean += delta * other.finite_elements / combined_count
        self.m2 += (
            other.m2
            + delta * delta * first_count * other.finite_elements / combined_count
        )
        self.finite_elements = combined_count
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        if other.absmax > self.absmax:
            self.absmax = other.absmax
            self.max_hit_count = other.max_hit_count
        elif other.absmax == self.absmax:
            self.max_hit_count += other.max_hit_count
        self.zero_abs_count += other.zero_abs_count
        self.histogram += other.histogram
        if self.fixed_clipping_range is None:
            self.fixed_clipping_range = other.fixed_clipping_range
        elif (
            other.fixed_clipping_range is not None
            and not math.isclose(self.fixed_clipping_range, other.fixed_clipping_range)
        ):
            raise ValueError("cannot merge different fixed clipping ranges")
        self.fixed_clipping_count += other.fixed_clipping_count

    def approximate_abs_quantile(self, probability: float) -> float | None:
        if not self.finite_elements:
            return None
        rank = max(1, math.ceil(probability * self.finite_elements))
        if rank <= self.zero_abs_count:
            return 0.0
        cumulative = self.zero_abs_count
        width = _LOG_HISTOGRAM_MAX_EXPONENT - _LOG_HISTOGRAM_MIN_EXPONENT
        for index, count in enumerate(self.histogram.tolist()):
            cumulative += count
            if cumulative >= rank:
                upper_exponent = _LOG_HISTOGRAM_MIN_EXPONENT + (
                    (index + 1) * width / _LOG_HISTOGRAM_BINS
                )
                return min(2.0**upper_exponent, self.absmax)
        return self.absmax

    def as_dict(self, finalized_range: float | None = None) -> dict[str, Any]:
        clipping_rate = None
        clipping_rate_exact = False
        if self.finite_elements and finalized_range is not None:
            if (
                self.fixed_clipping_range is not None
                and self.fixed_clipping_range == finalized_range
            ):
                clipping_rate = self.fixed_clipping_count / self.finite_elements
                clipping_rate_exact = True
            elif self.absmax <= finalized_range or math.isclose(
                self.absmax, finalized_range, rel_tol=1e-6, abs_tol=1e-12
            ):
                # An absmax range derived from this same stream cannot clip it.
                clipping_rate = 0.0
                clipping_rate_exact = True
        return {
            "min": self.minimum if self.finite_elements else None,
            "max": self.maximum if self.finite_elements else None,
            "absmax": self.absmax if self.finite_elements else None,
            "mean": self.mean if self.finite_elements else None,
            "std": (
                math.sqrt(max(self.m2 / self.finite_elements, 0.0))
                if self.finite_elements else None
            ),
            "p99_abs": self.approximate_abs_quantile(0.99),
            "p999_abs": self.approximate_abs_quantile(0.999),
            "observed_elements": self.observed_elements,
            "finite_elements": self.finite_elements,
            "nonfinite_count": self.nonfinite_count,
            "execution_count": self.execution_count,
            "max_hit_rate": (
                self.max_hit_count / self.finite_elements
                if self.finite_elements else None
            ),
            "clipping_range_abs": finalized_range,
            "clipping_rate": clipping_rate,
            "clipping_rate_exact": clipping_rate_exact,
        }


class ActivationTracker:
    """Track bounded-memory activation statistics and scale execution coverage."""

    def __init__(
        self,
        model: Any,
        component: str = "unknown",
        fixed_clipping_ranges: Mapping[str, float] | None = None,
    ) -> None:
        self.component = component
        self.stage = "unassigned"
        self.executions: dict[str, Counter[str]] = defaultdict(Counter)
        self._statistics: dict[tuple[str, str, str], _StreamingActivationStatistics] = {}
        self._fixed_clipping_ranges = dict(fixed_clipping_ranges or {})
        self._tracked_modules: dict[str, Any] = {}
        self._handles = []
        for name, module in model.named_modules():
            if hasattr(module, "absmax") or hasattr(module, "summax_hidden"):
                self._tracked_modules[name] = module
                self._handles.append(module.register_forward_hook(self._activation_hook(name)))

    def _activation_hook(self, name: str):
        def hook(_module, inputs, _output):
            key = (self.component, self.stage, name)
            statistics = self._statistics.setdefault(key, _StreamingActivationStatistics())
            statistics.record_execution()
            self.executions[name][self.stage] += 1
            fixed_range = self._fixed_clipping_ranges.get(name)
            for value in _input_tensors(inputs):
                statistics.update(value, fixed_range)
        return hook

    @staticmethod
    def _finalized_range(module: Any) -> float | None:
        if not hasattr(module, "absmax"):
            return None
        value = float(torch.as_tensor(module.absmax).detach().cpu().item())
        return value if math.isfinite(value) and value >= 0 else None

    def _aggregate(self, module_name: str) -> _StreamingActivationStatistics:
        result = _StreamingActivationStatistics()
        for (component, _stage, name), statistics in self._statistics.items():
            if component == self.component and name == module_name:
                result.merge(statistics)
        return result

    def activation_statistics(self, model: Any | None = None) -> list[dict[str, Any]]:
        modules = dict(model.named_modules()) if model is not None else self._tracked_modules
        rows: list[dict[str, Any]] = []
        for (component, stage, name), statistics in sorted(self._statistics.items()):
            module = modules.get(name)
            finalized_range = self._finalized_range(module) if module is not None else None
            row = {
                "component": component,
                "graph_stage": stage,
                "module_name": name,
                "kind": (
                    "ConstFakeQuant" if module is not None and hasattr(module, "absmax")
                    else type(module).__name__ if module is not None else "unknown"
                ),
                **statistics.as_dict(finalized_range),
            }
            rows.append(row)
        return rows

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def snapshot(self, model: Any) -> dict[str, Any]:
        activation_points: dict[str, Any] = {}
        for name, module in model.named_modules():
            if name not in self._tracked_modules:
                continue
            statistics = self._aggregate(name)
            finalized_range = self._finalized_range(module)
            diagnostics = statistics.as_dict(finalized_range)
            for metric_name in (
                "min", "max", "absmax", "mean", "std", "p99_abs", "p999_abs"
            ):
                diagnostics[f"activation_{metric_name}"] = diagnostics.pop(metric_name)
            if hasattr(module, "absmax"):
                value = float(torch.as_tensor(module.absmax).detach().cpu().item())
                activation_points[name] = {
                    "kind": "ConstFakeQuant",
                    "absmax": value,
                    "executions": dict(self.executions[name]),
                    **diagnostics,
                }
            else:
                raw = module.summax_hidden
                activation_points[name] = {
                    "kind": type(module).__name__,
                    "summax_hidden": (
                        None if raw is None else float(torch.as_tensor(raw).cpu().item())
                    ),
                    "scale": float(module.scale),
                    "executions": dict(self.executions[name]),
                    **diagnostics,
                }
        return activation_points


# Compatibility for callers outside the reorganized calibration entry point.
ObserverTracker = ActivationTracker


def compare_snapshots(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    first_samples: int | None = None,
    second_samples: int | None = None,
) -> dict[str, Any]:
    layers = []
    for name in sorted(set(first) | set(second)):
        before = first.get(name, {})
        after = second.get(name, {})
        metric = "absmax" if "absmax" in after else "scale"
        before_value = before.get(metric)
        final_value = after.get(metric)
        if before_value is None or final_value is None:
            relative = None
        else:
            relative = abs(final_value - before_value) / max(abs(final_value), 1e-12)
        layers.append({
            "name": name,
            "kind": after.get("kind", before.get("kind")),
            "metric": metric,
            "before_value": before_value,
            "final_value": final_value,
            "relative_drift": relative,
        })
    finite = [row["relative_drift"] for row in layers if row["relative_drift"] is not None]
    ordered = sorted(finite)
    p95_index = math.ceil(0.95 * len(ordered)) - 1 if ordered else None
    return {
        "from_samples": first_samples,
        "to_samples": second_samples,
        "layers": layers,
        "max_relative_drift": max(finite, default=None),
        "mean_relative_drift": sum(finite) / len(finite) if finite else None,
        "p95_relative_drift": ordered[p95_index] if p95_index is not None else None,
        "outliers_over_10pct": [
            row["name"] for row in layers
            if (row["relative_drift"] or 0) > 0.1
        ],
    }


def apply_scale_manifest(
    model: Any,
    manifest_path: Path,
    group: str,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Restore audited activation-scale state before switching to compile mode."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sample_count is None:
        sample_count = int(manifest.get("sample_count", 0))
    if sample_count <= 0:
        raise ValueError("scale manifest has no positive sample_count")
    if manifest.get("sample_count") != sample_count:
        raise ValueError(
            f"scale manifest sample_count={manifest.get('sample_count')} != {sample_count}"
        )
    try:
        snapshot = manifest[group][str(sample_count)]
    except KeyError as exc:
        raise ValueError(f"scale manifest has no {group}/{sample_count} snapshot") from exc
    modules = dict(model.named_modules())
    missing_modules = sorted(set(snapshot) - set(modules))
    retired_attention_points = [
        name
        for name in missing_modules
        if snapshot[name].get("kind") == "ConstFakeQuant"
        and re.fullmatch(
            r"blocks\.\d+\.(?:qk_matmul|wv_matmul)\.[xy]_fake_quant",
            name,
        )
    ]
    unexpected_missing = sorted(set(missing_modules) - set(retired_attention_points))
    if unexpected_missing:
        raise ValueError(
            f"scale manifest references unknown modules: {unexpected_missing[:3]}"
        )
    applied = 0
    for name, value in snapshot.items():
        if name not in modules:
            continue
        module = modules[name]
        if value.get("kind") == "ConstFakeQuant":
            absmax = float(value["absmax"])
            if absmax <= 0:
                raise ValueError(f"invalid zero absmax for {name}")
            module.absmax.copy_(torch.tensor(absmax, device=module.absmax.device))
        else:
            scale = float(value["scale"])
            summax = value.get("summax_hidden")
            if scale <= 0 or summax is None:
                raise ValueError(f"invalid RMSNorm scale for {name}")
            module.scale = scale
            module.summax_hidden = torch.tensor(float(summax), device=module.weight.device)
            module.i_scale.copy_(torch.tensor(1.0 / scale, device=module.i_scale.device))
            module.i_scale_pow.copy_(
                torch.tensor(1.0 / (scale * scale), device=module.i_scale_pow.device)
            )
        applied += 1
    return {
        "group": group,
        "sample_count": sample_count,
        "applied_modules": applied,
        "ignored_dynamic_attention_activation_points": len(retired_attention_points),
        # Deprecated compatibility key for existing build consumers.
        "ignored_dynamic_attention_observers": len(retired_attention_points),
        "generated_manifest_sha256": manifest.get("generated_manifest_sha256"),
    }
