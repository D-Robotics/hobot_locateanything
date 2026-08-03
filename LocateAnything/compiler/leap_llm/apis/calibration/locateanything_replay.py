"""Activation-statistics replay helpers for LocateAnything calibration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

import torch


REQUIRED_TENSOR_FIELDS = {
    "prompt_input_ids",
    "prompt_attention_mask",
    "vision_input",
    "projected_visual_features",
    "prediction_token_ids",
    "target_token_ids",
}

DECODE_CONTEXT_POLICY = "bundle_hash_base_plus_detection_target_tail_v2"
DECODE_CONTEXT_PENDING_TOKENS = 6
DECODE_DEPTH_BUCKETS = ("zero", "1_31", "32_127", "128_plus")
BASE_CONTEXT_ROLE = "base"
SUPPLEMENTAL_CONTEXT_ROLES = ("target_tail",)


def decode_depth_bucket(suffix_len: int) -> str:
    if suffix_len < 0:
        raise ValueError("decode suffix length cannot be negative")
    if suffix_len == 0:
        return "zero"
    if suffix_len < 32:
        return "1_31"
    if suffix_len < 128:
        return "32_127"
    return "128_plus"


class DecodeReplayContext(NamedTuple):
    """One deterministic history used to replay every Language graph variant."""

    context_id: str
    context_role: str
    bundle_id: str
    token_source: str
    selection_slot: int
    boundary_token_id: int | None
    prompt_len: int
    max_suffix_len: int
    suffix_token_ids: tuple[int, ...]
    pending_token_ids: tuple[int, ...]
    anchor_token_id: int
    target_usable_suffix_len: int
    eligible_target_offsets: tuple[int, ...]
    required_target_offsets: tuple[int, ...]

    @property
    def suffix_len(self) -> int:
        return len(self.suffix_token_ids)

    @property
    def past_len(self) -> int:
        return self.prompt_len + self.suffix_len

    @property
    def depth_bucket(self) -> str:
        return decode_depth_bucket(self.suffix_len)

    def coverage_record(self, task: str) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "context_role": self.context_role,
            "bundle_id": self.bundle_id,
            "task": str(task),
            "token_source": self.token_source,
            "selection_slot": self.selection_slot,
            "boundary_token_id": self.boundary_token_id,
            "prompt_len": self.prompt_len,
            "max_suffix_len": self.max_suffix_len,
            "offset": self.suffix_len,
            "suffix_len": self.suffix_len,
            "past_len": self.past_len,
            "depth_bucket": self.depth_bucket,
            "target_usable_suffix_len": self.target_usable_suffix_len,
            "eligible_target_offsets": list(self.eligible_target_offsets),
            "required_target_offsets": list(self.required_target_offsets),
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
        raise ValueError("prepared prompt must be unpadded before fixed-profile replay")
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


def append_cache_updates(
    keys: Iterable[torch.Tensor],
    values: Iterable[torch.Tensor],
    new_keys: Iterable[torch.Tensor],
    new_values: Iterable[torch.Tensor],
    *,
    accepted: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Append accepted causal K/V rows to fixed-length right-aligned caches."""

    if accepted <= 0:
        raise ValueError("accepted cache rows must be positive")
    key_pairs = list(zip(keys, new_keys, strict=True))
    value_pairs = list(zip(values, new_values, strict=True))
    if len(key_pairs) != len(value_pairs):
        raise ValueError("key and value cache layer counts differ")
    for cache, update in (*key_pairs, *value_pairs):
        if accepted > cache.shape[1] or accepted > update.shape[1]:
            raise ValueError(
                f"cannot append {accepted} rows from cache/update shapes "
                f"{tuple(cache.shape)}/{tuple(update.shape)}"
            )
    updated_keys = [
        torch.cat((cache[:, accepted:], update[:, :accepted]), dim=1)
        for cache, update in key_pairs
    ]
    updated_values = [
        torch.cat((cache[:, accepted:], update[:, :accepted]), dim=1)
        for cache, update in value_pairs
    ]
    return updated_keys, updated_values


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


def replay_sequential_ar_q1(
    text_model: Any,
    token_ids: Iterable[int],
    keys: Iterable[torch.Tensor],
    values: Iterable[torch.Tensor],
    *,
    active_len: int,
    cache_len: int,
    device: torch.device,
    dtype: torch.dtype,
    input_builder: Any = build_decode_inputs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Replay accepted tokens through q1 while committing each K/V update."""

    cache_keys = list(keys)
    cache_values = list(values)
    replay_tokens = [int(token_id) for token_id in token_ids]
    if not replay_tokens:
        raise ValueError("sequential AR replay requires at least one token")
    for token_offset, token_id in enumerate(replay_tokens):
        embeds, positions, mask = input_builder(
            text_model,
            [token_id],
            q_len=1,
            past_len=active_len + token_offset,
            cache_len=cache_len,
            is_pbd=False,
            device=device,
            dtype=dtype,
        )
        logits, new_keys, new_values = text_model(
            embeds,
            positions,
            mask,
            *(cache_keys + cache_values),
        )
        cache_keys, cache_values = append_cache_updates(
            cache_keys,
            cache_values,
            new_keys,
            new_values,
            accepted=1,
        )
        del logits, new_keys, new_values, embeds, positions, mask
    return cache_keys, cache_values


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


def _flat_token_ids(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(token) for token in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(token) for token in value]
    return []


def _structural_offsets(
    token_ids: list[int],
    *,
    structural_token_ids: set[int],
    max_suffix_len: int,
    pending_tokens: int,
) -> list[int]:
    latest = min(max_suffix_len, len(token_ids) - pending_tokens)
    if latest < 0:
        return []
    offsets = {0}
    offsets.update(
        index
        for index, token_id in enumerate(token_ids[: latest + 1])
        if token_id in structural_token_ids
    )
    return sorted(offsets)


def decode_context_id(bundle_id: str, token_source: str, offset: int) -> str:
    """Return a stable identity keyed only by bundle, token source, and offset."""

    identity = json.dumps(
        {
            "bundle_id": str(bundle_id),
            "offset": int(offset),
            "token_source": str(token_source),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def required_detection_target_offsets(
    eligible_positive_offsets: Iterable[int],
    *,
    usable_suffix_len: int,
) -> tuple[int, ...]:
    """Select one target-tail context for a long Detection output."""

    offsets = sorted({int(offset) for offset in eligible_positive_offsets if int(offset) > 0})
    if usable_suffix_len < 32 or not offsets:
        return ()
    return (offsets[-1],)


def select_decode_replay_context(
    payload: dict[str, Any],
    *,
    chunk_size: int,
    pending_tokens: int = DECODE_CONTEXT_PENDING_TOKENS,
    task: str | None = None,
) -> DecodeReplayContext:
    """Select the stable base boundary shared by all Decode graph variants.

    Slot zero preserves the prompt boundary. The remaining four hash slots pick
    progressively deeper structural boundaries. Prediction and target streams
    are both eligible; the bundle hash selects their preference without making
    replay depend on manifest order or resume position.
    """

    if pending_tokens < 1:
        raise ValueError("pending token count must be positive")
    prompt_ids = _flat_token_ids(payload.get("prompt_input_ids"))
    attention_mask = _flat_token_ids(payload.get("prompt_attention_mask"))
    prompt_len = sum(attention_mask)
    if prompt_len < 1 or prompt_len > len(prompt_ids):
        raise ValueError(f"invalid prompt length for Decode replay: {prompt_len}")
    max_suffix_len = chunk_size - prompt_len
    if max_suffix_len < 0:
        raise ValueError(
            f"prompt length {prompt_len} exceeds Prefill chunk size {chunk_size}"
        )

    bundle_id = str(payload.get("bundle_id") or "")
    task = str(task if task is not None else payload.get("task") or "")
    digest = hashlib.sha256(bundle_id.encode("utf-8")).digest()
    selection_slot = digest[0] % 5
    special = payload.get("special_token_ids") or {}
    structural_token_ids = {
        int(special.get("<ref>", 151672)),
        int(special.get("<box>", 151668)),
    }
    streams = [
        ("target", _flat_token_ids(payload.get("target_token_ids"))),
        (
            "prediction:hybrid",
            _flat_token_ids((payload.get("prediction_token_ids") or {}).get("hybrid")),
        ),
    ]
    target_ids = streams[0][1]
    target_usable_suffix_len = 0
    eligible_target_offsets: tuple[int, ...] = ()
    if task == "detection" and target_ids:
        target_usable_suffix_len = max(
            0, min(max_suffix_len, len(target_ids) - pending_tokens)
        )
        eligible_target_offsets = tuple(
            offset
            for offset in _structural_offsets(
                target_ids,
                structural_token_ids=structural_token_ids,
                max_suffix_len=max_suffix_len,
                pending_tokens=pending_tokens,
            )
            if offset > 0
        )
    required_target_offsets = required_detection_target_offsets(
        eligible_target_offsets,
        usable_suffix_len=target_usable_suffix_len,
    )
    streams = [entry for entry in streams if entry[1]]
    if streams:
        pivot = digest[1] % len(streams)
        streams = streams[pivot:] + streams[:pivot]

    candidates: list[tuple[str, list[int], list[int]]] = []
    for source, token_ids in streams:
        offsets = _structural_offsets(
            token_ids,
            structural_token_ids=structural_token_ids,
            max_suffix_len=max_suffix_len,
            pending_tokens=pending_tokens,
        )
        if offsets:
            candidates.append((source, token_ids, offsets))

    if selection_slot:
        deep_candidates = [
            candidate for candidate in candidates if any(candidate[2])
        ]
        if deep_candidates:
            candidates = deep_candidates

    if candidates:
        token_source, token_ids, offsets = candidates[0]
        selected_token_ids = token_ids
        positive_offsets = [offset for offset in offsets if offset > 0]
        if selection_slot and positive_offsets:
            if len(positive_offsets) == 1:
                offset = positive_offsets[0]
            else:
                index = round(
                    (selection_slot - 1) * (len(positive_offsets) - 1) / 3
                )
                offset = positive_offsets[min(index, len(positive_offsets) - 1)]
        else:
            offset = 0 if 0 in offsets else offsets[0]
        pending = token_ids[offset : offset + pending_tokens]
        boundary_token_id = int(token_ids[offset])
    else:
        token_source = "fallback:hybrid+target"
        selected_token_ids = []
        offset = 0
        pending = select_decode_tokens(payload, "hybrid", pending_tokens)
        boundary_token_id = None

    suffix = tuple(int(token) for token in selected_token_ids[:offset])
    pending_tuple = tuple(int(token) for token in pending)
    if len(pending_tuple) != pending_tokens:
        raise ValueError(
            f"Decode replay needs {pending_tokens} pending tokens, got {len(pending_tuple)}"
        )
    anchor_token_id = suffix[-1] if suffix else int(prompt_ids[prompt_len - 1])
    return DecodeReplayContext(
        context_id=decode_context_id(bundle_id, token_source, offset),
        context_role=BASE_CONTEXT_ROLE,
        bundle_id=bundle_id,
        token_source=token_source,
        selection_slot=selection_slot,
        boundary_token_id=boundary_token_id,
        prompt_len=prompt_len,
        max_suffix_len=max_suffix_len,
        suffix_token_ids=suffix,
        pending_token_ids=pending_tuple,
        anchor_token_id=anchor_token_id,
        target_usable_suffix_len=target_usable_suffix_len,
        eligible_target_offsets=eligible_target_offsets,
        required_target_offsets=required_target_offsets,
    )


def select_decode_replay_contexts(
    payload: dict[str, Any],
    *,
    task: str,
    chunk_size: int,
    pending_tokens: int = DECODE_CONTEXT_PENDING_TOKENS,
) -> tuple[DecodeReplayContext, ...]:
    """Return one base context plus a bounded target-tail Detection context."""

    base = select_decode_replay_context(
        payload,
        chunk_size=chunk_size,
        pending_tokens=pending_tokens,
        task=task,
    )
    contexts = [base]
    seen = {base.context_id}
    target_ids = _flat_token_ids(payload.get("target_token_ids"))
    planned = tuple(zip(SUPPLEMENTAL_CONTEXT_ROLES, base.required_target_offsets))

    for context_role, offset in planned:
        context_identity = decode_context_id(base.bundle_id, "target", offset)
        if context_identity in seen:
            continue
        pending = tuple(int(token) for token in target_ids[offset : offset + pending_tokens])
        if len(pending) != pending_tokens:
            raise ValueError(
                f"{base.bundle_id}: target context at offset {offset} has "
                f"{len(pending)} pending tokens"
            )
        suffix = tuple(int(token) for token in target_ids[:offset])
        contexts.append(
            DecodeReplayContext(
                context_id=context_identity,
                context_role=context_role,
                bundle_id=base.bundle_id,
                token_source="target",
                selection_slot=-1,
                boundary_token_id=int(target_ids[offset]),
                prompt_len=base.prompt_len,
                max_suffix_len=base.max_suffix_len,
                suffix_token_ids=suffix,
                pending_token_ids=pending,
                anchor_token_id=int(suffix[-1]),
                target_usable_suffix_len=base.target_usable_suffix_len,
                eligible_target_offsets=base.eligible_target_offsets,
                required_target_offsets=base.required_target_offsets,
            )
        )
        seen.add(context_identity)

    if len(contexts) > 2:
        raise AssertionError(f"{base.bundle_id}: Decode context cap exceeded")
    return tuple(contexts)


def _numeric_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)

    def percentile(probability: float) -> int:
        index = max(0, math.ceil(probability * len(ordered)) - 1)
        return ordered[index]

    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def summarize_decode_context_coverage(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_samples: int,
    cache_len: int,
    max_query_len: int = 12,
) -> dict[str, Any]:
    """Summarize and validate the contexts used for Language activation replay."""

    contexts = sorted(
        (dict(record) for record in records),
        key=lambda row: (str(row.get("bundle_id") or ""), str(row.get("context_id") or "")),
    )
    errors: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        grouped[str(context.get("bundle_id") or "")].append(context)
    if len(grouped) != expected_samples:
        errors.append(
            f"Decode context sample_count={len(grouped)} expected={expected_samples}"
        )
    if "" in grouped:
        errors.append("Decode context has an empty bundle_id")

    context_ids = [str(context.get("context_id") or "") for context in contexts]
    if not all(context_ids):
        errors.append("Decode context has an empty context_id")
    if len(set(context_ids)) != len(context_ids):
        errors.append("Decode context_id values are not unique")

    depth_counts = Counter({bucket: 0 for bucket in DECODE_DEPTH_BUCKETS})
    source_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    task_depth_counts: dict[str, Counter[str]] = defaultdict(Counter)
    suffix_lengths: list[int] = []
    past_lengths: list[int] = []
    for context in contexts:
        bundle_id = str(context.get("bundle_id") or "")
        context_id = str(context.get("context_id") or "")
        context_role = str(context.get("context_role") or "")
        suffix_len = int(context.get("suffix_len", -1))
        prompt_len = int(context.get("prompt_len", -1))
        past_len = int(context.get("past_len", -1))
        max_suffix_len = int(context.get("max_suffix_len", -1))
        bucket = str(context.get("depth_bucket") or "")
        source = str(context.get("token_source") or "")
        task = str(context.get("task") or "unknown")
        if context_id != decode_context_id(bundle_id, source, suffix_len):
            errors.append(f"{bundle_id}: context_id does not match source/offset")
        if context_role not in {BASE_CONTEXT_ROLE, *SUPPLEMENTAL_CONTEXT_ROLES}:
            errors.append(f"{bundle_id}: invalid context_role={context_role}")
        if suffix_len < 0 or suffix_len > max_suffix_len:
            errors.append(f"{bundle_id}: invalid suffix_len={suffix_len}")
        if past_len != prompt_len + suffix_len:
            errors.append(f"{bundle_id}: past_len does not match prompt+suffix")
        if past_len < 1 or past_len + max_query_len > cache_len:
            errors.append(f"{bundle_id}: past_len={past_len} exceeds cache profile")
        if bucket != decode_depth_bucket(max(suffix_len, 0)):
            errors.append(f"{bundle_id}: inconsistent depth bucket")
        if not source:
            errors.append(f"{bundle_id}: missing token source")
        if suffix_len >= 0:
            suffix_lengths.append(suffix_len)
        if past_len >= 0:
            past_lengths.append(past_len)
        if bucket in DECODE_DEPTH_BUCKETS:
            depth_counts[bucket] += 1
            task_depth_counts[task][bucket] += 1
        source_counts[source] += 1
        role_counts[context_role] += 1

    base_context_count = 0
    supplemental_context_count = 0
    eligible_long_detection_count = 0
    required_target_context_count = 0
    covered_required_target_context_count = 0
    missing_required_target_contexts: list[str] = []
    for bundle_id, bundle_contexts in sorted(grouped.items()):
        if len(bundle_contexts) > 2:
            errors.append(f"{bundle_id}: has {len(bundle_contexts)} contexts; maximum is 2")
        bases = [
            context
            for context in bundle_contexts
            if context.get("context_role") == BASE_CONTEXT_ROLE
        ]
        base_context_count += len(bases)
        supplemental_context_count += len(bundle_contexts) - len(bases)
        if len(bases) != 1:
            errors.append(f"{bundle_id}: expected exactly one base context, got {len(bases)}")
            continue
        base = bases[0]
        task = str(base.get("task") or "")
        try:
            target_usable_suffix_len = int(base["target_usable_suffix_len"])
            eligible_offsets = tuple(int(value) for value in base["eligible_target_offsets"])
            required_offsets = tuple(int(value) for value in base["required_target_offsets"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{bundle_id}: invalid target offset metadata")
            continue
        if tuple(sorted(set(eligible_offsets))) != eligible_offsets or any(
            offset <= 0 for offset in eligible_offsets
        ):
            errors.append(f"{bundle_id}: eligible_target_offsets are not sorted unique positives")
        if (
            target_usable_suffix_len < 0
            or target_usable_suffix_len > int(base.get("max_suffix_len", -1))
        ):
            errors.append(f"{bundle_id}: invalid target_usable_suffix_len")
        if eligible_offsets and eligible_offsets[-1] > target_usable_suffix_len:
            errors.append(f"{bundle_id}: eligible target offset exceeds usable suffix")
        if task != "detection" and target_usable_suffix_len != 0:
            errors.append(f"{bundle_id}: non-Detection context declares target suffix")
        if task == "detection" and target_usable_suffix_len >= 32 and not eligible_offsets:
            errors.append(f"{bundle_id}: long Detection target has no structural boundary")
        expected_required = (
            required_detection_target_offsets(
                eligible_offsets,
                usable_suffix_len=target_usable_suffix_len,
            )
            if task == "detection"
            else ()
        )
        if required_offsets != expected_required:
            errors.append(
                f"{bundle_id}: required target offsets {required_offsets} "
                f"do not match deterministic plan {expected_required}"
            )
        if expected_required:
            eligible_long_detection_count += 1
        required_target_context_count += len(expected_required)
        expected_roles = dict(zip(expected_required, SUPPLEMENTAL_CONTEXT_ROLES))
        for context in bundle_contexts:
            role = str(context.get("context_role") or "")
            if context.get("task") != task:
                errors.append(f"{bundle_id}: context task differs from base context")
            if tuple(context.get("eligible_target_offsets") or ()) != eligible_offsets:
                errors.append(f"{bundle_id}: context eligible offsets differ from base")
            if context.get("target_usable_suffix_len") != target_usable_suffix_len:
                errors.append(f"{bundle_id}: context usable suffix differs from base")
            if tuple(context.get("required_target_offsets") or ()) != required_offsets:
                errors.append(f"{bundle_id}: context required offsets differ from base")
            if role == BASE_CONTEXT_ROLE:
                continue
            offset = int(context.get("offset", -1))
            if task != "detection" or context.get("token_source") != "target":
                errors.append(
                    f"{bundle_id}: supplemental context must be Detection target replay"
                )
            if offset not in expected_required:
                errors.append(f"{bundle_id}: unexpected supplemental offset={offset}")
            elif role != expected_roles[offset]:
                errors.append(
                    f"{bundle_id}: offset {offset} has role={role}, "
                    f"expected={expected_roles[offset]}"
                )
        target_offsets = {
            int(context.get("offset", -1))
            for context in bundle_contexts
            if context.get("token_source") == "target"
        }
        for offset in expected_required:
            if offset in target_offsets:
                covered_required_target_context_count += 1
            else:
                missing = f"{bundle_id}:target:{offset}"
                missing_required_target_contexts.append(missing)
                errors.append(f"{bundle_id}: missing required target context offset={offset}")

    nonzero_count = sum(value for key, value in depth_counts.items() if key != "zero")
    deep_count = depth_counts["32_127"] + depth_counts["128_plus"]
    if contexts and depth_counts["zero"] == 0:
        errors.append("Decode context coverage has no prompt-boundary samples")
    if contexts and nonzero_count == 0:
        errors.append("Decode context coverage has no nonzero history suffix")
    if contexts and deep_count == 0:
        errors.append("Decode context coverage has no suffix of at least 32 tokens")
    detection_depths = task_depth_counts.get("detection", Counter())
    if detection_depths and not (
        detection_depths["32_127"] + detection_depths["128_plus"]
    ):
        errors.append("Detection Decode coverage has no suffix of at least 32 tokens")

    selection_sha256 = hashlib.sha256(
        json.dumps(contexts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "policy": DECODE_CONTEXT_POLICY,
        "sample_count": len(grouped),
        "language_context_count": len(contexts),
        "base_context_count": base_context_count,
        "supplemental_context_count": supplemental_context_count,
        "eligible_long_detection_sample_count": eligible_long_detection_count,
        "required_target_context_count": required_target_context_count,
        "covered_required_target_context_count": covered_required_target_context_count,
        "missing_required_target_contexts": missing_required_target_contexts,
        "selection_sha256": selection_sha256,
        "suffix_len": _numeric_summary(suffix_lengths),
        "past_len": _numeric_summary(past_lengths),
        "depth_buckets": dict(depth_counts),
        "token_sources": dict(sorted(source_counts.items())),
        "context_roles": dict(sorted(role_counts.items())),
        "task_depth_buckets": {
            task: {bucket: counts.get(bucket, 0) for bucket in DECODE_DEPTH_BUCKETS}
            for task, counts in sorted(task_depth_counts.items())
        },
        "contexts": contexts,
        "errors": errors,
        "passed": not errors,
    }


def select_pbd_tokens(
    payload: dict[str, Any],
    q_len: int,
    text_mask_token_id: int,
    *,
    anchor_token_id: int | None = None,
) -> list[int]:
    """Build the native PBD window: history anchor followed by mask tokens."""

    if q_len < 1:
        raise ValueError("PBD q_len must be positive")
    input_ids = payload["prompt_input_ids"].reshape(-1)
    attention_mask = payload["prompt_attention_mask"].reshape(-1)
    active_len = int(attention_mask.sum().item())
    if active_len < 1 or active_len > input_ids.numel():
        raise ValueError(f"invalid active prompt length for PBD: {active_len}")
    anchor = (
        int(anchor_token_id)
        if anchor_token_id is not None
        else int(input_ids[active_len - 1].item())
    )
    return [anchor, *([int(text_mask_token_id)] * (q_len - 1))]


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

    @property
    def tracked_module_count(self) -> int:
        """Number of model modules whose static activation state is tracked."""
        return len(self._tracked_modules)

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

    required_scale_modules: dict[str, str] = {}
    for name, module in modules.items():
        if hasattr(module, "absmax"):
            # Disabled fake-quant modules do not consume a calibrated range in
            # their build path and therefore do not require a manifest entry.
            if getattr(module, "quantized", True):
                required_scale_modules[name] = "ConstFakeQuant"
        elif all(
            hasattr(module, attribute)
            for attribute in (
                "scale", "summax_hidden", "weight", "i_scale", "i_scale_pow"
            )
        ):
            required_scale_modules[name] = "RMSNorm"

    missing_scales = sorted(set(required_scale_modules) - set(snapshot))
    if missing_scales:
        raise ValueError(
            "scale manifest is missing current quantized modules: "
            f"{missing_scales[:3]}"
        )

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
        expected_kind = required_scale_modules.get(name)
        manifest_kind = value.get("kind")
        if expected_kind is None:
            raise ValueError(
                f"scale manifest references non-quantized module: {name}"
            )
        if expected_kind == "ConstFakeQuant" and manifest_kind != "ConstFakeQuant":
            raise ValueError(
                f"scale manifest kind mismatch for {name}: "
                f"expected ConstFakeQuant, got {manifest_kind!r}"
            )
        if expected_kind == "RMSNorm" and manifest_kind == "ConstFakeQuant":
            raise ValueError(
                f"scale manifest kind mismatch for {name}: "
                f"expected RMSNorm-compatible scale, got ConstFakeQuant"
            )
        if expected_kind == "ConstFakeQuant":
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
