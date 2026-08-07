"""Coordinate-focused Language validation for LocateAnything hybrid decoding."""

from __future__ import annotations

import math
from typing import Any, Iterable


AUDIT_VERSION = 4
COORD_START_TOKEN_ID = 151677
COORD_END_TOKEN_ID = 152677
BOX_START_TOKEN_ID = 151668
BOX_END_TOKEN_ID = 151669
REF_START_TOKEN_ID = 151672
REF_END_TOKEN_ID = 151673
NONE_TOKEN_ID = 4064
NULL_TOKEN_ID = 152678
IM_END_TOKEN_ID = 151645
TEXT_MASK_TOKEN_ID = 151676
PIXELS_PER_COORDINATE = 672.0 / 1000.0
BBOX_KEEP_K = 4
REF_KEEP_K = 5
AR_FALLBACK_MAX_NEW_TOKENS = 8


def _flat_ints(value: Any) -> list[int]:
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return [int(item) for item in value.reshape(-1).tolist()]
    return [int(item) for item in value]


def token_ids_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    special = payload.get("special_token_ids", {})
    return {
        "box_start_token_id": int(special.get("<box>", BOX_START_TOKEN_ID)),
        "box_end_token_id": int(special.get("</box>", BOX_END_TOKEN_ID)),
        "ref_start_token_id": int(special.get("<ref>", REF_START_TOKEN_ID)),
        "ref_end_token_id": int(special.get("</ref>", REF_END_TOKEN_ID)),
        "coord_start_token_id": COORD_START_TOKEN_ID,
        "coord_end_token_id": COORD_END_TOKEN_ID,
        "none_token_id": NONE_TOKEN_ID,
        "null_token_id": int(special.get("<null>", NULL_TOKEN_ID)),
        "im_end_token_id": int(special.get("<|im_end|>", IM_END_TOKEN_ID)),
        "default_mask_token_id": int(
            special.get("<text_mask>", TEXT_MASK_TOKEN_ID)
        ),
    }


def _coordinate_values(tokens: Iterable[int], token_ids: dict[str, int]) -> list[int]:
    start = token_ids["coord_start_token_id"]
    end = token_ids["coord_end_token_id"]
    values = [int(token) for token in tokens]
    if not all(start <= token <= end for token in values):
        return []
    return [token - start for token in values]


def extract_coordinate_probes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Locate every box emitted by the saved upstream hybrid response."""

    predictions = payload.get("prediction_token_ids", {})
    hybrid = _flat_ints(predictions.get("hybrid", []))
    token_ids = token_ids_from_payload(payload)
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]
    none = token_ids["none_token_id"]
    probes: list[dict[str, Any]] = []
    for offset, token in enumerate(hybrid):
        if token != box_start:
            continue
        window = hybrid[offset : offset + 6]
        kind = "invalid"
        group = list(window)
        coordinates: list[int] = []
        if len(window) >= 3 and window[1] == none and window[2] == box_end:
            kind = "empty"
            group = window[:3]
        elif len(window) >= 6 and window[5] == box_end:
            coordinates = _coordinate_values(window[1:5], token_ids)
            if len(coordinates) == 4:
                kind = "box"
                group = window[:6]
        elif len(window) >= 4 and window[3] == box_end:
            coordinates = _coordinate_values(window[1:3], token_ids)
            if len(coordinates) == 2:
                kind = "point"
                group = window[:4]
        probes.append(
            {
                "index": len(probes),
                "response_offset": offset,
                "source_kind": kind,
                "source_tokens": group,
                "source_coordinate_values": coordinates,
                "anchor_token_id": (
                    hybrid[offset - 1]
                    if offset
                    else int(payload["prompt_input_ids"].reshape(-1)[-1].item())
                ),
            }
        )
    return probes


def _apply_repetition_penalty(
    logits: Any,
    generated_token_ids: list[list[int]],
    penalty: float,
) -> Any:
    import torch

    if penalty == 1.0:
        return logits
    for batch_index, tokens in enumerate(generated_token_ids):
        if not tokens:
            continue
        ids = torch.tensor(
            sorted(set(tokens)), device=logits.device, dtype=torch.long
        )
        values = logits[batch_index, :, ids]
        adjusted = torch.where(values < 0, values * penalty, values / penalty)
        logits[batch_index, :, ids] = adjusted
    return logits


def _nucleus_summary(
    logits: Any,
    generated_token_ids: list[list[int]],
    token_ids: dict[str, int],
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> dict[str, Any]:
    import torch

    if temperature <= 0:
        raise ValueError("coordinate audit temperature must be positive")
    work = logits.float().div(float(temperature))
    _apply_repetition_penalty(work, generated_token_ids, repetition_penalty)
    log_normalizer = torch.logsumexp(work, dim=-1, keepdim=True)
    vocab_size = int(work.shape[-1])
    width = min(256, vocab_size)
    while True:
        values, ids = torch.topk(work, k=width, dim=-1)
        raw_probabilities = torch.exp(values - log_normalizer)
        covered_mass = raw_probabilities.sum(dim=-1)
        if bool((covered_mass >= top_p).all()) or width == vocab_size:
            break
        if width >= 8192:
            minimum = float(covered_mass.min().item())
            raise RuntimeError(
                f"top-{width} covers only {minimum:.6f} probability mass"
            )
        width = min(width * 2, vocab_size, 8192)

    cumulative = raw_probabilities.cumsum(dim=-1)
    keep_count = (cumulative < top_p).sum(dim=-1) + 1
    keep_count = keep_count.clamp(max=width)
    positions = torch.arange(width, device=work.device)
    keep_mask = positions.view(1, 1, -1) < keep_count.unsqueeze(-1)
    kept_values = values.masked_fill(~keep_mask, -torch.inf)
    probabilities = torch.softmax(kept_values, dim=-1)
    probabilities = probabilities.masked_fill(~keep_mask, 0)

    special_names = (
        "box_start_token_id",
        "box_end_token_id",
        "ref_start_token_id",
        "ref_end_token_id",
        "none_token_id",
        "null_token_id",
        "im_end_token_id",
    )
    special_ids = torch.tensor(
        [token_ids[name] for name in special_names],
        device=work.device,
        dtype=torch.long,
    )
    special_probabilities = (
        probabilities.unsqueeze(-1)
        * (ids.unsqueeze(-1) == special_ids.view(1, 1, 1, -1))
    ).sum(dim=-2)
    retained = min(REF_KEEP_K, width)
    top_ids = ids[..., :retained]
    top_probabilities = probabilities[..., :retained]
    top_logits = values[..., :retained]
    top_valid = keep_mask[..., :retained]
    top_ids = top_ids.masked_fill(~top_valid, -1)
    top_logits = top_logits.masked_fill(~top_valid, -torch.inf)
    kept_log_normalizer = torch.logsumexp(kept_values, dim=-1)
    del values, ids, raw_probabilities, cumulative, kept_values
    return {
        "top_ids": top_ids.detach().cpu(),
        "top_probabilities": top_probabilities.detach().cpu(),
        "top_logits": top_logits.detach().cpu(),
        "special_probabilities": special_probabilities.detach().cpu(),
        "special_names": special_names,
        "nucleus_width": width,
        "covered_mass_min": float(covered_mass.min().item()),
        "_adjusted_logits": work,
        "_keep_count": keep_count,
        "_kept_log_normalizer": kept_log_normalizer,
    }


def _release_summary(summary: dict[str, Any]) -> None:
    for name in ("_adjusted_logits", "_keep_count", "_kept_log_normalizer"):
        summary.pop(name, None)


def _is_coordinate(token: int, token_ids: dict[str, int]) -> bool:
    return token_ids["coord_start_token_id"] <= token <= token_ids["coord_end_token_id"]


def _top_candidates(
    summary: dict[str, Any],
    batch_index: int,
    position: int,
    token_ids: dict[str, int],
    *,
    count: int = BBOX_KEEP_K,
) -> list[dict[str, Any]]:
    ids = summary["top_ids"][batch_index, position, :count].tolist()
    probabilities = summary["top_probabilities"][batch_index, position, :count].tolist()
    logits = summary["top_logits"][batch_index, position, :count].tolist()
    result: list[dict[str, Any]] = []
    for rank, (token, probability, logit) in enumerate(
        zip(ids, probabilities, logits, strict=True), 1
    ):
        token = int(token)
        if token < 0:
            continue
        coordinate = (
            token - token_ids["coord_start_token_id"]
            if _is_coordinate(token, token_ids)
            else None
        )
        result.append(
            {
                "rank": rank,
                "token_id": token,
                "coordinate": coordinate,
                "is_coordinate": coordinate is not None,
                "decoder_probability": float(probability),
                "adjusted_logit": float(logit),
            }
        )
    return result


def _token_distribution_evidence(
    summary: dict[str, Any],
    batch_index: int,
    position: int,
    token: int | None,
) -> dict[str, Any] | None:
    import torch

    if token is None or token < 0:
        return None
    adjusted = summary["_adjusted_logits"][batch_index, position]
    if token >= adjusted.shape[-1]:
        return None
    target = adjusted[token]
    full_rank = int((adjusted > target).sum().item()) + 1
    keep_count = int(summary["_keep_count"][batch_index, position].item())
    retained = full_rank <= keep_count
    probability = (
        float(
            torch.exp(
                target - summary["_kept_log_normalizer"][batch_index, position]
            ).item()
        )
        if retained
        else 0.0
    )
    top_ids = summary["top_ids"][batch_index, position, :BBOX_KEEP_K].tolist()
    top4_rank = next(
        (rank for rank, candidate in enumerate(top_ids, 1) if int(candidate) == token),
        None,
    )
    return {
        "token_id": int(token),
        "full_rank": full_rank,
        "top4_rank": top4_rank,
        "top_p_retained": retained,
        "decoder_probability": probability,
        "adjusted_logit": float(target.item()),
    }


def _handle_pattern(tokens: list[int], token_ids: dict[str, int]) -> dict[str, Any]:
    null = token_ids["null_token_id"]
    im_end = token_ids["im_end_token_id"]
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]
    none = token_ids["none_token_id"]
    coord_start = token_ids["coord_start_token_id"]
    coord_end = token_ids["coord_end_token_id"]
    ref_end = token_ids["ref_end_token_id"]
    if tokens[0] in {null, im_end}:
        return {"type": "im_end", "tokens": [im_end], "fallback": False}
    if tokens[:2] == [box_start, none]:
        return {
            "type": "empty",
            "tokens": [box_start, none, box_end],
            "fallback": False,
        }
    if tokens[0] == box_start:
        coordinate_end = 1
        for token in tokens[1:5]:
            if coord_start <= token <= coord_end:
                coordinate_end += 1
            else:
                break
        if coordinate_end == 5 and tokens[5] == box_end:
            return {"type": "box", "tokens": tokens, "fallback": False}
        if coordinate_end == 3 and tokens[3] == box_end:
            return {
                "type": "point",
                "tokens": tokens[:4],
                "fallback": False,
            }
        return {
            "type": "error_box",
            "tokens": tokens[:coordinate_end],
            "fallback": True,
        }
    trimmed = list(tokens)
    if null in trimmed:
        trimmed = trimmed[: trimmed.index(null)]
    if len(trimmed) >= 2 and trimmed[-1] == trimmed[-2] == ref_end:
        trimmed = trimmed[:-1]
    return {"type": "ref_object", "tokens": trimmed, "fallback": False}


def _decode_one(
    summary: dict[str, Any],
    batch_index: int,
    token_ids: dict[str, int],
) -> dict[str, Any]:
    top_ids = summary["top_ids"][batch_index].tolist()
    top_probabilities = summary["top_probabilities"][batch_index].tolist()
    special = summary["special_probabilities"][batch_index].tolist()
    special_names = summary["special_names"]
    special_by_position = [
        dict(zip(special_names, row, strict=True)) for row in special
    ]
    raw_top1 = [int(row[0]) for row in top_ids]
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]
    none = token_ids["none_token_id"]
    null = token_ids["null_token_id"]
    im_end = token_ids["im_end_token_id"]
    coord_start = token_ids["coord_start_token_id"]
    coord_end = token_ids["coord_end_token_id"]

    frame = "illegal_box"
    if special_by_position[0]["box_start_token_id"] >= 0.7:
        if (
            special_by_position[1]["none_token_id"] > 0.2
            and special_by_position[2]["box_end_token_id"] > 0.2
            and special_by_position[3]["null_token_id"] > 0.1
            and special_by_position[4]["null_token_id"] > 0.1
        ):
            frame = "empty_box"
    end_score = sum(
        special_by_position[5][name]
        for name in ("box_end_token_id", "null_token_id", "im_end_token_id")
    )
    if frame != "empty_box" and end_score >= 0.2:
        frame = "legal_box"

    decoded: list[int] | None = None
    candidates: list[dict[str, Any]] = []
    if frame == "empty_box":
        decoded = [box_start, none, box_end, null, null, null]
    elif frame == "legal_box":
        coordinates: list[int] = []
        for position in range(1, 5):
            valid = [
                (int(token), float(probability))
                for token, probability in zip(
                    top_ids[position][:BBOX_KEEP_K],
                    top_probabilities[position][:BBOX_KEEP_K],
                    strict=True,
                )
                if coord_start <= int(token) <= coord_end
            ]
            candidates.append(
                {
                    "position": position,
                    "candidates": [
                        {
                            "token_id": token,
                            "coordinate": token - coord_start,
                            "probability": probability,
                        }
                        for token, probability in valid
                    ],
                }
            )
            if not valid:
                break
            token, probability = valid[0]
            spread = max(item[0] for item in valid) - min(item[0] for item in valid)
            abnormal = probability < 0.9 and len(valid) > 1 and spread > 60
            coordinates.append(0 if abnormal else token)
        if len(coordinates) == 4:
            decoded = [box_start, *coordinates, box_end]

    if decoded is None:
        if special_by_position[0]["ref_start_token_id"] >= 0.6:
            text_tokens: list[int] = []
            for position in range(1, 6):
                valid = [
                    int(token)
                    for token in top_ids[position]
                    if int(token) >= 0
                    and not coord_start <= int(token) <= coord_end
                ]
                if not valid:
                    text_tokens = []
                    break
                text_tokens.append(valid[0])
            if text_tokens:
                decoded = [token_ids["ref_start_token_id"], *text_tokens]
        if decoded is None:
            decoded = raw_top1

    pattern = _handle_pattern(decoded, token_ids)
    coordinate_values = _coordinate_values(
        pattern["tokens"][1:-1]
        if pattern["type"] in {"box", "point"}
        else [],
        token_ids,
    )
    return {
        "type": pattern["type"],
        "tokens": [int(token) for token in pattern["tokens"]],
        "coordinate_values": coordinate_values,
        "fallback": bool(pattern["fallback"]),
        "raw_top1": raw_top1,
        "frame": frame,
        "frame_probabilities": {
            "box_start": special_by_position[0]["box_start_token_id"],
            "box_end_or_terminal": end_score,
        },
        "coordinate_candidates": candidates,
        "position_top4": [
            {
                "position": position,
                "candidates": _top_candidates(
                    summary, batch_index, position, token_ids
                ),
            }
            for position in range(1, min(5, len(top_ids)))
        ],
        "nucleus_width": summary["nucleus_width"],
        "covered_mass_min": summary["covered_mass_min"],
    }


def _decoded_coordinate_token(
    decoded: dict[str, Any], position: int, token_ids: dict[str, int]
) -> int | None:
    tokens = decoded.get("tokens", [])
    if position >= len(tokens):
        return None
    token = int(tokens[position])
    return token if _is_coordinate(token, token_ids) else None


def _coordinate_delta(
    reference_token: int | None,
    candidate_token: int | None,
    token_ids: dict[str, int],
) -> dict[str, float]:
    if reference_token is None or candidate_token is None:
        return {"comparable": 0.0}
    start = token_ids["coord_start_token_id"]
    reference = reference_token - start
    candidate = candidate_token - start
    delta = candidate - reference
    return {
        "comparable": 1.0,
        "token_exact": float(delta == 0),
        "token_delta": float(delta),
        "token_abs_delta": float(abs(delta)),
        "pixel_delta": float(delta * PIXELS_PER_COORDINATE),
        "pixel_abs_delta": float(abs(delta) * PIXELS_PER_COORDINATE),
    }


def position_diagnostics(
    probe: dict[str, Any],
    reference: dict[str, Any],
    candidate: dict[str, Any],
    reference_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    batch_index: int,
    token_ids: dict[str, int],
) -> list[dict[str, Any]]:
    source_values = probe.get("source_coordinate_values", [])
    if len(source_values) not in {2, 4}:
        return []
    source_tokens = probe.get("source_tokens", [])
    rows: list[dict[str, Any]] = []
    for position in range(1, len(source_values) + 1):
        source_token = (
            int(source_tokens[position])
            if position < len(source_tokens)
            and _is_coordinate(int(source_tokens[position]), token_ids)
            else None
        )
        reference_token = _decoded_coordinate_token(reference, position, token_ids)
        candidate_token = _decoded_coordinate_token(candidate, position, token_ids)
        float_in_candidate = _token_distribution_evidence(
            candidate_summary, batch_index, position, reference_token
        )
        if float_in_candidate is not None:
            selected = _token_distribution_evidence(
                candidate_summary, batch_index, position, candidate_token
            )
            top1 = _token_distribution_evidence(
                candidate_summary,
                batch_index,
                position,
                int(candidate_summary["top_ids"][batch_index, position, 0].item()),
            )
            float_logit = float(float_in_candidate["adjusted_logit"])
            float_in_candidate["selected_minus_float_logit_margin"] = (
                None
                if selected is None
                else float(selected["adjusted_logit"]) - float_logit
            )
            float_in_candidate["top1_minus_float_logit_margin"] = (
                None
                if top1 is None
                else float(top1["adjusted_logit"]) - float_logit
            )
        comparison = _coordinate_delta(reference_token, candidate_token, token_ids)
        if float_in_candidate is not None:
            comparison.update(
                {
                    "float_token_top4_hit": float(
                        float_in_candidate["top4_rank"] is not None
                    ),
                    "float_token_rank_in_quantized": float(
                        float_in_candidate["full_rank"]
                    ),
                    "float_token_probability_in_quantized": float(
                        float_in_candidate["decoder_probability"]
                    ),
                }
            )
            for target, source in (
                (
                    "selected_minus_float_logit_margin",
                    "selected_minus_float_logit_margin",
                ),
                ("top1_minus_float_logit_margin", "top1_minus_float_logit_margin"),
            ):
                if float_in_candidate[source] is not None:
                    comparison[target] = float(float_in_candidate[source])
        rows.append(
            {
                "position": position,
                "source": {
                    "token_id": source_token,
                    "coordinate": int(source_values[position - 1]),
                    "in_float": _token_distribution_evidence(
                        reference_summary, batch_index, position, source_token
                    ),
                    "in_quantized_eager": _token_distribution_evidence(
                        candidate_summary, batch_index, position, source_token
                    ),
                },
                "float": {
                    "selected_token_id": reference_token,
                    "coordinate": (
                        None
                        if reference_token is None
                        else reference_token - token_ids["coord_start_token_id"]
                    ),
                    "raw_top1_token_id": int(
                        reference_summary["top_ids"][batch_index, position, 0].item()
                    ),
                    "top4": _top_candidates(
                        reference_summary, batch_index, position, token_ids
                    ),
                },
                "quantized_eager": {
                    "selected_token_id": candidate_token,
                    "coordinate": (
                        None
                        if candidate_token is None
                        else candidate_token - token_ids["coord_start_token_id"]
                    ),
                    "raw_top1_token_id": int(
                        candidate_summary["top_ids"][batch_index, position, 0].item()
                    ),
                    "top4": _top_candidates(
                        candidate_summary, batch_index, position, token_ids
                    ),
                },
                "float_token_in_quantized": float_in_candidate,
                "comparison": comparison,
            }
        )
    return rows


def _decode_pbd_logits_with_summary(
    logits: Any,
    generated_token_ids: list[list[int]],
    token_ids: dict[str, int],
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = _nucleus_summary(
        logits,
        generated_token_ids,
        token_ids,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    decoded = [
        _decode_one(summary, index, token_ids) for index in range(int(logits.shape[0]))
    ]
    return decoded, summary


def decode_pbd_logits(
    logits: Any,
    generated_token_ids: list[list[int]],
    token_ids: dict[str, int],
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> list[dict[str, Any]]:
    decoded, summary = _decode_pbd_logits_with_summary(
        logits,
        generated_token_ids,
        token_ids,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    _release_summary(summary)
    return decoded


def _box_iou(reference: list[int], candidate: list[int]) -> float:
    rx1, ry1, rx2, ry2 = reference
    cx1, cy1, cx2, cy2 = candidate
    intersection_width = max(0, min(rx2, cx2) - max(rx1, cx1))
    intersection_height = max(0, min(ry2, cy2) - max(ry1, cy1))
    intersection = intersection_width * intersection_height
    reference_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    candidate_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    union = reference_area + candidate_area - intersection
    return float(intersection / union) if union else float(reference == candidate)


def compare_decoded_coordinates(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float]:
    reference_values = reference["coordinate_values"]
    candidate_values = candidate["coordinate_values"]
    comparison: dict[str, float] = {
        "structure_agreement": float(reference["type"] == candidate["type"]),
        "float_valid": float(reference["type"] in {"box", "point", "empty"}),
        "candidate_valid": float(candidate["type"] in {"box", "point", "empty"}),
        "float_fallback": float(reference["fallback"]),
        "candidate_fallback": float(candidate["fallback"]),
    }
    if not reference_values or len(reference_values) != len(candidate_values):
        return comparison
    errors = [abs(left - right) for left, right in zip(reference_values, candidate_values)]
    comparison.update(
        coordinate_token_exact=float(reference_values == candidate_values),
        coordinate_mae=float(sum(errors) / len(errors)),
        coordinate_max_abs=float(max(errors)),
        pixel_mae=float(sum(errors) / len(errors) * PIXELS_PER_COORDINATE),
        pixel_max_abs=float(max(errors) * PIXELS_PER_COORDINATE),
    )
    if len(reference_values) == 4:
        comparison["box_iou"] = _box_iou(reference_values, candidate_values)
    elif len(reference_values) == 2:
        comparison["point_distance_pixels"] = math.dist(
            reference_values, candidate_values
        ) * PIXELS_PER_COORDINATE
    return comparison


def source_decoded(probe: dict[str, Any]) -> dict[str, Any]:
    kind = str(probe["source_kind"])
    return {
        "type": kind,
        "tokens": [int(token) for token in probe["source_tokens"]],
        "coordinate_values": [
            int(value) for value in probe["source_coordinate_values"]
        ],
        "fallback": kind == "invalid",
    }


def coordinate_metric_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in audit.get("decisions", []):
        source_kind = str(decision["source"]["kind"])
        primary = decision.get("resolved_comparison", decision["comparison"])
        rows.append(
            {
                "reference_sequence": 20_000 + int(decision["index"]),
                "semantic_group": "COORDINATE OUTPUT",
                "semantic_operation": source_kind,
                "module": f"coordinates/{source_kind}",
                "shape": [len(decision["source"]["coordinate_values"])],
                "status": "matched",
                "comparison": primary,
            }
        )
        rows.extend(
            {
                "reference_sequence": 21_000 + int(decision["index"]) * 4 + offset,
                "semantic_group": "COORDINATE POSITION",
                "semantic_operation": f"{source_kind}.position_{item['position']}",
                "module": f"coordinates/{source_kind}/position_{item['position']}",
                "shape": [1],
                "status": "matched",
                "comparison": item["comparison"],
            }
            for offset, item in enumerate(decision.get("position_diagnostics", []))
        )
    return rows


class LanguageCoordinateAuditor:
    """Compare Float and quantized PBD coordinates under identical histories."""

    def __init__(
        self,
        model: Any,
        rotation: Any,
        device: Any,
        dtype: Any,
        zero_caches: list[Any],
        emulator: Any,
        boundaries: Any | None = None,
        *,
        chunk_size: int,
        cache_len: int,
        pbd_query_len: int,
        image_token_id: int,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.rotation = rotation
        self.device = device
        self.dtype = dtype
        self.zero_caches = zero_caches
        self.emulator = emulator
        self.boundaries = boundaries
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        self.pbd_query_len = pbd_query_len
        self.image_token_id = image_token_id
        self.batch_size = batch_size

    def _begin_boundary_capture(
        self, phase: str, stage: str, active_tokens: int
    ) -> None:
        if self.boundaries is None:
            return
        if phase == "reference":
            self.boundaries.begin_reference(stage, active_tokens)
        else:
            self.boundaries.begin_candidate(stage, active_tokens)

    def _pause_boundary_capture(self) -> None:
        if self.boundaries is not None:
            self.boundaries.phase = "off"

    def _run_ar_token(
        self,
        token: int,
        history_keys: list[Any],
        history_values: list[Any],
        *,
        quantized: bool,
    ) -> tuple[Any, list[Any], list[Any]]:
        import torch

        past_len = int(history_keys[0].shape[1])
        token_tensor = torch.tensor(
            [[token]], device=self.device, dtype=torch.long
        )
        embeds = self.model.embed_tokens(token_tensor).to(dtype=self.dtype)
        positions = torch.tensor(
            [[[past_len]]], device=self.device, dtype=torch.int32
        )
        attention = torch.zeros(
            (1, 1, 1, past_len + 1), device=self.device, dtype=self.dtype
        )
        caches: list[Any] = []
        for history in [*history_keys, *history_values]:
            padding = torch.zeros(
                (1, 1, *history.shape[2:]),
                device=history.device,
                dtype=history.dtype,
            )
            caches.append(torch.cat((padding, history), dim=1))
        self.emulator.set_stage("ar_q1")
        self.emulator.set_enabled(quantized)
        self._pause_boundary_capture()
        with torch.no_grad():
            logits, new_keys, new_values = self.model(
                embeds, positions, attention, *caches
            )
        updated_keys = [
            torch.cat((history, value), dim=1)
            for history, value in zip(history_keys, new_keys, strict=True)
        ]
        updated_values = [
            torch.cat((history, value), dim=1)
            for history, value in zip(history_values, new_values, strict=True)
        ]
        del caches, new_keys, new_values
        return logits, updated_keys, updated_values

    def _run_ar_fallback(
        self,
        probe: dict[str, Any],
        initial: dict[str, Any],
        source_keys: list[Any],
        source_values: list[Any],
        prompt_tokens: list[int],
        hybrid: list[int],
        prompt_len: int,
        token_ids: dict[str, int],
        *,
        quantized: bool,
    ) -> dict[str, Any] | None:
        if not initial["fallback"]:
            return None
        history_len = prompt_len + int(probe["response_offset"])
        history_keys = [value[:, :history_len].clone() for value in source_keys]
        history_values = [value[:, :history_len].clone() for value in source_values]
        generated = prompt_tokens + hybrid[: int(probe["response_offset"])]
        accepted_prefix = [int(token) for token in initial["tokens"]]
        logits = None
        replay: list[dict[str, Any]] = []
        for token in accepted_prefix:
            position = len(generated)
            logits, history_keys, history_values = self._run_ar_token(
                token,
                history_keys,
                history_values,
                quantized=quantized,
            )
            generated.append(token)
            replay.append({"token_id": token, "position_id": position})

        if logits is None:
            raise RuntimeError("AR fallback has no accepted PBD prefix to replay")

        resolved_tokens = list(accepted_prefix)
        steps: list[dict[str, Any]] = []
        termination = "max_new_tokens"
        for step_index in range(AR_FALLBACK_MAX_NEW_TOKENS):
            summary = _nucleus_summary(
                logits,
                [generated],
                token_ids,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            selected = int(summary["top_ids"][0, 0, 0].item())
            source_index = len(resolved_tokens)
            source_token = (
                int(probe["source_tokens"][source_index])
                if source_index < len(probe["source_tokens"])
                else None
            )
            selected_evidence = _token_distribution_evidence(
                summary, 0, 0, selected
            )
            source_evidence = _token_distribution_evidence(
                summary, 0, 0, source_token
            )
            comparison: dict[str, float] = {
                "token_exact": float(source_token == selected)
                if source_token is not None
                else 0.0
            }
            if (
                source_token is not None
                and _is_coordinate(source_token, token_ids)
                and _is_coordinate(selected, token_ids)
            ):
                comparison.update(
                    _coordinate_delta(source_token, selected, token_ids)
                )
            steps.append(
                {
                    "step": step_index,
                    "output_index": source_index,
                    "selected_token_id": selected,
                    "source_token_id": source_token,
                    "selected": selected_evidence,
                    "source_token_in_distribution": source_evidence,
                    "top4": _top_candidates(summary, 0, 0, token_ids),
                    "comparison": comparison,
                }
            )
            _release_summary(summary)
            del logits
            resolved_tokens.append(selected)
            generated.append(selected)
            if selected == token_ids["box_end_token_id"]:
                termination = "box_end"
                break
            if not (
                _is_coordinate(selected, token_ids)
                or selected == token_ids["none_token_id"]
            ):
                termination = "non_coordinate"
                break
            logits, history_keys, history_values = self._run_ar_token(
                selected,
                history_keys,
                history_values,
                quantized=quantized,
            )

        padded = resolved_tokens[:6] + [token_ids["null_token_id"]] * max(
            0, 6 - len(resolved_tokens)
        )
        pattern = _handle_pattern(padded[:6], token_ids)
        coordinate_values = _coordinate_values(
            pattern["tokens"][1:-1]
            if pattern["type"] in {"box", "point"}
            else [],
            token_ids,
        )
        resolved = {
            "type": pattern["type"],
            "tokens": [int(token) for token in pattern["tokens"]],
            "coordinate_values": coordinate_values,
            "fallback": bool(pattern["fallback"]),
        }
        del history_keys, history_values
        return {
            "method": "deterministic_greedy_q1_after_pbd_fallback",
            "accepted_pbd_prefix": accepted_prefix,
            "prefix_replay": replay,
            "steps": steps,
            "termination": termination,
            "resolved": resolved,
        }

    def _prefill(self, payload: dict[str, Any], suffix: list[int]):
        from pipeline.replay import build_prefill_inputs

        prompt_len = int(payload["prompt_attention_mask"].reshape(-1).sum().item())
        active_len = prompt_len + len(suffix)
        inputs = build_prefill_inputs(
            self.model,
            payload,
            self.rotation,
            chunk_size=max(self.chunk_size, active_len),
            cache_len=self.cache_len,
            image_token_id=self.image_token_id,
            device=self.device,
            dtype=self.dtype,
            suffix_token_ids=suffix,
        )
        return inputs, prompt_len

    def _pbd_inputs(self, probes: list[dict[str, Any]], prompt_len: int):
        import torch

        past_lengths = [prompt_len + int(probe["response_offset"]) for probe in probes]
        max_past = max(past_lengths)
        cache_width = max_past + self.pbd_query_len
        anchors = torch.tensor(
            [probe["anchor_token_id"] for probe in probes],
            device=self.device,
            dtype=torch.long,
        )
        masks = torch.full(
            (len(probes), self.pbd_query_len - 1),
            TEXT_MASK_TOKEN_ID,
            device=self.device,
            dtype=torch.long,
        )
        tokens = torch.cat((anchors.unsqueeze(1), masks), dim=1)
        embeds = self.model.embed_tokens(tokens).to(dtype=self.dtype)
        base = torch.tensor(past_lengths, device=self.device, dtype=torch.int32)
        offsets = torch.arange(
            self.pbd_query_len, device=self.device, dtype=torch.int32
        ) - 1
        positions = (base.unsqueeze(1) + offsets.unsqueeze(0)).unsqueeze(1)
        attention = torch.full(
            (len(probes), 1, self.pbd_query_len, cache_width),
            -32768.0,
            device=self.device,
            dtype=self.dtype,
        )
        current_start = cache_width - self.pbd_query_len
        for index, past_len in enumerate(past_lengths):
            attention[index, :, :, current_start - past_len : current_start] = 0
            attention[index, :, :, current_start:] = 0
            attention[index, :, :, current_start - 1] = -32768.0
        return (embeds, positions, attention), past_lengths, cache_width

    @staticmethod
    def _prefix_caches(
        keys: list[Any],
        values: list[Any],
        past_lengths: list[int],
        cache_width: int,
    ) -> list[Any]:
        import torch

        result: list[Any] = []
        for source in [*keys, *values]:
            target = torch.zeros(
                (len(past_lengths), cache_width, *source.shape[2:]),
                device=source.device,
                dtype=source.dtype,
            )
            for index, past_len in enumerate(past_lengths):
                target[index, -past_len:] = source[0, :past_len]
            result.append(target)
        return result

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        import torch

        probes = extract_coordinate_probes(payload)
        token_ids = token_ids_from_payload(payload)
        hybrid = _flat_ints(payload["prediction_token_ids"].get("hybrid", []))
        if not probes:
            return {
                "version": AUDIT_VERSION,
                "method": "teacher_forced_hybrid_pbd_q6_with_ar_q1_fallback",
                "decision_count": 0,
                "decisions": [],
            }
        max_prefix = max(int(probe["response_offset"]) for probe in probes)
        prefill_inputs, prompt_len = self._prefill(payload, hybrid[:max_prefix])
        if prompt_len + max_prefix >= self.cache_len - self.pbd_query_len:
            raise ValueError("coordinate history does not fit the configured KV cache")

        self.emulator.set_stage("prefill")
        self.emulator.set_enabled(False)
        self._begin_boundary_capture("reference", "coordinate_prefill", prompt_len + max_prefix)
        with torch.no_grad():
            reference_logits, reference_keys, reference_values = self.model(
                *prefill_inputs[:3], *self.zero_caches
            )
        del reference_logits
        self.emulator.set_enabled(True)
        self._begin_boundary_capture("candidate", "coordinate_prefill", prompt_len + max_prefix)
        with torch.no_grad():
            candidate_logits, candidate_keys, candidate_values = self.model(
                *prefill_inputs[:3], *self.zero_caches
            )
        del candidate_logits
        prefill_operator_rows = [
            {**row, "stage": "coordinate_prefill"}
            for row in self.emulator.rows
        ]

        prompt_tokens = _flat_ints(payload["prompt_input_ids"])
        decisions: list[dict[str, Any]] = []
        pbd_operator_rows: list[dict[str, Any]] = []
        try:
            for start in range(0, len(probes), self.batch_size):
                batch = probes[start : start + self.batch_size]
                pbd_inputs, past_lengths, cache_width = self._pbd_inputs(
                    batch, prompt_len
                )
                generated = [
                    prompt_tokens + hybrid[: int(probe["response_offset"])]
                    for probe in batch
                ]

                reference_caches = self._prefix_caches(
                    reference_keys, reference_values, past_lengths, cache_width
                )
                self.emulator.set_stage("pbd_q6")
                self.emulator.set_enabled(False)
                self._begin_boundary_capture(
                    "reference", "coordinate_pbd_q6", self.pbd_query_len
                )
                with torch.no_grad():
                    reference_batch_logits, _, _ = self.model(
                        *pbd_inputs, *reference_caches
                    )
                del reference_caches

                candidate_caches = self._prefix_caches(
                    candidate_keys, candidate_values, past_lengths, cache_width
                )
                self.emulator.set_enabled(True)
                self._begin_boundary_capture(
                    "candidate", "coordinate_pbd_q6", self.pbd_query_len
                )
                with torch.no_grad():
                    candidate_batch_logits, _, _ = self.model(
                        *pbd_inputs, *candidate_caches
                    )
                del candidate_caches
                batch_operator_rows = [
                    {
                        **row,
                        "stage": "coordinate_pbd_q6",
                        "decision_indexes": [
                            int(probe["index"]) for probe in batch
                        ],
                    }
                    for row in self.emulator.rows
                ]
                pbd_operator_rows.extend(batch_operator_rows)

                reference_decoded, reference_summary = _decode_pbd_logits_with_summary(
                    reference_batch_logits,
                    generated,
                    token_ids,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
                candidate_decoded, candidate_summary = _decode_pbd_logits_with_summary(
                    candidate_batch_logits,
                    generated,
                    token_ids,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

                decision_start = len(decisions)
                for batch_index, (probe, reference, candidate) in enumerate(
                    zip(batch, reference_decoded, candidate_decoded, strict=True)
                ):
                    source = source_decoded(probe)
                    decisions.append(
                        {
                            "index": int(probe["index"]),
                            "response_offset": int(probe["response_offset"]),
                            "source": {
                                "kind": probe["source_kind"],
                                "tokens": probe["source_tokens"],
                                "coordinate_values": probe[
                                    "source_coordinate_values"
                                ],
                            },
                            "float": reference,
                            "quantized_eager": candidate,
                            "comparison": compare_decoded_coordinates(
                                reference, candidate
                            ),
                            "source_to_float": compare_decoded_coordinates(
                                source, reference
                            ),
                            "source_to_quantized_eager": compare_decoded_coordinates(
                                source, candidate
                            ),
                            "position_diagnostics": position_diagnostics(
                                probe,
                                reference,
                                candidate,
                                reference_summary,
                                candidate_summary,
                                batch_index,
                                token_ids,
                            ),
                        }
                    )
                _release_summary(reference_summary)
                _release_summary(candidate_summary)
                del reference_batch_logits, candidate_batch_logits

                for probe, decision in zip(
                    batch, decisions[decision_start:], strict=True
                ):
                    reference_ar = self._run_ar_fallback(
                        probe,
                        decision["float"],
                        reference_keys,
                        reference_values,
                        prompt_tokens,
                        hybrid,
                        prompt_len,
                        token_ids,
                        quantized=False,
                    )
                    candidate_ar = self._run_ar_fallback(
                        probe,
                        decision["quantized_eager"],
                        candidate_keys,
                        candidate_values,
                        prompt_tokens,
                        hybrid,
                        prompt_len,
                        token_ids,
                        quantized=True,
                    )
                    resolved_reference = (
                        reference_ar["resolved"]
                        if reference_ar is not None
                        else decision["float"]
                    )
                    resolved_candidate = (
                        candidate_ar["resolved"]
                        if candidate_ar is not None
                        else decision["quantized_eager"]
                    )
                    source = source_decoded(probe)
                    decision["ar_q1"] = {
                        "float": reference_ar,
                        "quantized_eager": candidate_ar,
                    }
                    decision["resolved"] = {
                        "float": resolved_reference,
                        "quantized_eager": resolved_candidate,
                    }
                    decision["resolved_comparison"] = compare_decoded_coordinates(
                        resolved_reference, resolved_candidate
                    )
                    decision["source_to_resolved_float"] = (
                        compare_decoded_coordinates(source, resolved_reference)
                    )
                    decision["source_to_resolved_quantized_eager"] = (
                        compare_decoded_coordinates(source, resolved_candidate)
                    )
            boundary_rows = (
                list(self.boundaries.rows) if self.boundaries is not None else []
            )
        finally:
            self.emulator.set_enabled(False)
            if self.boundaries is not None:
                self.boundaries.finish_sample()
            del reference_keys, reference_values, candidate_keys, candidate_values

        return {
            "version": AUDIT_VERSION,
            "method": "teacher_forced_hybrid_pbd_q6_with_ar_q1_fallback",
            "decoder": {
                "generation_mode": "hybrid",
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "sampling": "greedy_for_audit",
                "bbox_keep_k": BBOX_KEEP_K,
                "ref_keep_k": REF_KEEP_K,
                "ar_fallback": {
                    "q_len": 1,
                    "sampling": "deterministic_greedy_for_audit",
                    "max_new_tokens": AR_FALLBACK_MAX_NEW_TOKENS,
                    "accepted_pbd_prefix": "replayed_one_token_at_a_time",
                },
            },
            "cache_sources": {
                "float": "float_teacher_forced_prefill",
                "quantized_eager": "quantized_teacher_forced_prefill",
            },
            "comparison_scopes": {
                "comparison": "pbd_q6_float_vs_quantized_eager",
                "resolved_comparison": "post_fallback_float_vs_quantized_eager",
                "source": "saved_upstream_sampled_hybrid_output_not_ground_truth",
            },
            "boundary_scope": "prefill_and_pbd_q6_only",
            "coordinate_token_range": [
                token_ids["coord_start_token_id"],
                token_ids["coord_end_token_id"],
            ],
            "pixel_scale": PIXELS_PER_COORDINATE,
            "decision_count": len(decisions),
            "decisions": decisions,
            "boundaries": boundary_rows,
            "operators": {
                "prefill": prefill_operator_rows,
                "pbd_q6": pbd_operator_rows,
            },
        }
