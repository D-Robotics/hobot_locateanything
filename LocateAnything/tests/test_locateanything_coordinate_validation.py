from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "compiler/scripts/common/coordinates.py"
)
SPEC = importlib.util.spec_from_file_location("locateanything_coordinate_validation", SCRIPT)
coordinate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(coordinate)

UPSTREAM_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/upstream_generate_utils.py"
)
UPSTREAM_SPEC = importlib.util.spec_from_file_location(
    "locateanything_upstream_generate_utils", UPSTREAM_SCRIPT
)
upstream = importlib.util.module_from_spec(UPSTREAM_SPEC)
assert UPSTREAM_SPEC.loader is not None
UPSTREAM_SPEC.loader.exec_module(upstream)


def payload_with_hybrid(tokens: list[int]) -> dict:
    return {
        "prompt_input_ids": torch.tensor([[10, 11]]),
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
        "prediction_token_ids": {"hybrid": torch.tensor(tokens)},
        "special_token_ids": {
            "<box>": coordinate.BOX_START_TOKEN_ID,
            "</box>": coordinate.BOX_END_TOKEN_ID,
            "<ref>": coordinate.REF_START_TOKEN_ID,
            "</ref>": coordinate.REF_END_TOKEN_ID,
            "<null>": coordinate.NULL_TOKEN_ID,
            "<text_mask>": coordinate.TEXT_MASK_TOKEN_ID,
            "<|im_end|>": coordinate.IM_END_TOKEN_ID,
        },
    }


def test_extract_coordinate_probes_preserves_box_and_point_context():
    c = coordinate.COORD_START_TOKEN_ID
    tokens = [
        coordinate.REF_START_TOKEN_ID,
        42,
        coordinate.REF_END_TOKEN_ID,
        coordinate.BOX_START_TOKEN_ID,
        c + 10,
        c + 20,
        c + 30,
        c + 40,
        coordinate.BOX_END_TOKEN_ID,
        coordinate.BOX_START_TOKEN_ID,
        c + 50,
        c + 60,
        coordinate.BOX_END_TOKEN_ID,
    ]

    probes = coordinate.extract_coordinate_probes(payload_with_hybrid(tokens))

    assert [probe["source_kind"] for probe in probes] == ["box", "point"]
    assert probes[0]["response_offset"] == 3
    assert probes[0]["source_coordinate_values"] == [10, 20, 30, 40]
    assert probes[1]["source_coordinate_values"] == [50, 60]


def logits_for_tokens(tokens: list[int]) -> torch.Tensor:
    logits = torch.full((1, 6, coordinate.COORD_END_TOKEN_ID + 4), -30.0)
    for position, token in enumerate(tokens):
        logits[0, position, token] = 30.0
    return logits


def upstream_greedy_decode(
    logits: torch.Tensor, generated: list[list[int]], token_ids: dict[str, int]
) -> list[dict]:
    history = torch.tensor(generated, dtype=torch.long)
    work = upstream.apply_repetition_penalty(logits.clone(), history, 1.1)
    work = upstream.top_p_logits(work / 0.7, 0.9)
    probabilities = torch.softmax(work, dim=-1)
    type_map = {"coord_box": "box", "point_box": "point", "empty_box": "empty"}
    decoded = []
    for batch_index in range(logits.shape[0]):
        tokens = upstream.decode_bbox_avg(
            work[batch_index],
            probabilities[batch_index],
            token_ids,
            keep_k=4,
            generation_mode="hybrid",
        )
        if tokens is None:
            tokens = upstream.decode_ref(
                work[batch_index], probabilities[batch_index], token_ids, keep_k=5
            )
        if tokens is None:
            tokens = probabilities[batch_index].argmax(dim=-1)
        pattern = upstream.handle_pattern(tokens, token_ids, "hybrid")
        decoded.append(
            {
                "type": type_map.get(pattern["type"], pattern["type"]),
                "tokens": [int(token) for token in pattern["tokens"]],
            }
        )
    return decoded


def test_decode_pbd_logits_reproduces_standard_coordinate_box():
    c = coordinate.COORD_START_TOKEN_ID
    expected = [
        coordinate.BOX_START_TOKEN_ID,
        c + 100,
        c + 200,
        c + 300,
        c + 400,
        coordinate.BOX_END_TOKEN_ID,
    ]
    payload = payload_with_hybrid(expected)

    decoded = coordinate.decode_pbd_logits(
        logits_for_tokens(expected), [[]], coordinate.token_ids_from_payload(payload)
    )[0]

    assert decoded["type"] == "box"
    assert decoded["tokens"] == expected
    assert decoded["coordinate_values"] == [100, 200, 300, 400]
    assert decoded["fallback"] is False

    native = upstream_greedy_decode(
        logits_for_tokens(expected), [[]], coordinate.token_ids_from_payload(payload)
    )[0]
    assert decoded["type"] == native["type"]
    assert decoded["tokens"] == native["tokens"]


def test_decode_pbd_logits_reproduces_point_from_raw_pbd_pattern():
    c = coordinate.COORD_START_TOKEN_ID
    expected = [
        coordinate.BOX_START_TOKEN_ID,
        c + 250,
        c + 750,
        coordinate.BOX_END_TOKEN_ID,
        coordinate.NULL_TOKEN_ID,
        coordinate.NULL_TOKEN_ID,
    ]
    payload = payload_with_hybrid(expected)

    decoded = coordinate.decode_pbd_logits(
        logits_for_tokens(expected), [[]], coordinate.token_ids_from_payload(payload)
    )[0]

    assert decoded["type"] == "point"
    assert decoded["coordinate_values"] == [250, 750]

    native = upstream_greedy_decode(
        logits_for_tokens(expected), [[]], coordinate.token_ids_from_payload(payload)
    )[0]
    assert decoded["type"] == native["type"]
    assert decoded["tokens"] == native["tokens"]


def test_bbox_decoder_ignores_coordinate_candidate_below_native_top4():
    c = coordinate.COORD_START_TOKEN_ID
    logits = torch.full((1, 6, coordinate.COORD_END_TOKEN_ID + 4), -30.0)
    logits[0, 0, coordinate.BOX_START_TOKEN_ID] = 30.0
    logits[0, 5, coordinate.BOX_END_TOKEN_ID] = 30.0
    for position in range(1, 5):
        for rank, token in enumerate((10, 11, 12, 13)):
            logits[0, position, token] = 30.0 - rank
        logits[0, position, c + position] = 25.0

    payload = payload_with_hybrid([])
    decoded = coordinate.decode_pbd_logits(
        logits, [[]], coordinate.token_ids_from_payload(payload)
    )[0]

    native = upstream_greedy_decode(
        logits, [[]], coordinate.token_ids_from_payload(payload)
    )[0]

    assert coordinate.BBOX_KEEP_K == 4
    assert decoded["type"] == native["type"]
    assert decoded["tokens"] == native["tokens"]
    assert decoded["type"] != "box"


def test_coordinate_comparison_reports_token_pixel_and_geometry_errors():
    reference = {
        "type": "box",
        "coordinate_values": [100, 200, 500, 600],
        "fallback": False,
    }
    candidate = {
        "type": "box",
        "coordinate_values": [110, 200, 500, 620],
        "fallback": False,
    }

    result = coordinate.compare_decoded_coordinates(reference, candidate)

    assert result["coordinate_token_exact"] == 0
    assert result["coordinate_mae"] == pytest.approx(7.5)
    assert result["coordinate_max_abs"] == 20
    assert result["pixel_max_abs"] == pytest.approx(13.44)
    assert 0 < result["box_iou"] < 1


def test_position_diagnostics_tracks_float_token_rank_and_margin():
    c = coordinate.COORD_START_TOKEN_ID
    reference_tokens = [
        coordinate.BOX_START_TOKEN_ID,
        c + 100,
        c + 200,
        c + 300,
        c + 400,
        coordinate.BOX_END_TOKEN_ID,
    ]
    candidate_tokens = list(reference_tokens)
    candidate_tokens[1] = c + 102
    reference_logits = logits_for_tokens(reference_tokens)
    candidate_logits = logits_for_tokens(candidate_tokens)
    candidate_logits[0, 1, c + 100] = 29.0
    payload = payload_with_hybrid(reference_tokens)
    token_ids = coordinate.token_ids_from_payload(payload)

    reference, reference_summary = coordinate._decode_pbd_logits_with_summary(
        reference_logits,
        [[]],
        token_ids,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    candidate, candidate_summary = coordinate._decode_pbd_logits_with_summary(
        candidate_logits,
        [[]],
        token_ids,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    probe = coordinate.extract_coordinate_probes(payload)[0]

    rows = coordinate.position_diagnostics(
        probe,
        reference[0],
        candidate[0],
        reference_summary,
        candidate_summary,
        0,
        token_ids,
    )
    coordinate._release_summary(reference_summary)
    coordinate._release_summary(candidate_summary)

    first = rows[0]
    assert first["float"]["selected_token_id"] == c + 100
    assert first["quantized_eager"]["selected_token_id"] == c + 102
    assert first["float_token_in_quantized"]["full_rank"] == 2
    assert first["float_token_in_quantized"]["top4_rank"] == 2
    assert first["comparison"]["token_delta"] == 2
    assert first["comparison"]["pixel_abs_delta"] == pytest.approx(1.344)
    assert first["float_token_in_quantized"][
        "selected_minus_float_logit_margin"
    ] == pytest.approx((30.0 - 29.0) / 0.7, abs=1e-5)


class DummyEmulator:
    def set_stage(self, _stage: str) -> None:
        pass

    def set_enabled(self, _enabled: bool) -> None:
        pass


class DummyARModel:
    def __init__(self, transitions: dict[int, int]) -> None:
        self.transitions = transitions
        self.current_token = 0

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.current_token = int(token_ids.reshape(-1)[0].item())
        return torch.zeros((*token_ids.shape, 4), dtype=torch.float32)

    def __call__(self, _embeds, _positions, _attention, *caches):
        vocab = coordinate.COORD_END_TOKEN_ID + 4
        logits = torch.full((1, 1, vocab), -30.0)
        logits[0, 0, self.transitions[self.current_token]] = 30.0
        layers = len(caches) // 2
        keys = [torch.zeros((1, 1, 1, 1)) for _ in range(layers)]
        values = [torch.zeros((1, 1, 1, 1)) for _ in range(layers)]
        return logits, keys, values


def test_ar_fallback_replays_prefix_with_q1_and_closes_box():
    c = coordinate.COORD_START_TOKEN_ID
    box = coordinate.BOX_START_TOKEN_ID
    end = coordinate.BOX_END_TOKEN_ID
    source_tokens = [box, c + 100, c + 200, c + 300, c + 400, end]
    transitions = dict(zip(source_tokens[:-1], source_tokens[1:], strict=True))
    model = DummyARModel(transitions)
    auditor = coordinate.LanguageCoordinateAuditor(
        model,
        rotation=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        zero_caches=[],
        emulator=DummyEmulator(),
        chunk_size=16,
        cache_len=32,
        pbd_query_len=6,
        image_token_id=1,
    )
    payload = payload_with_hybrid(source_tokens)
    probe = coordinate.extract_coordinate_probes(payload)[0]
    token_ids = coordinate.token_ids_from_payload(payload)

    result = auditor._run_ar_fallback(
        probe,
        {
            "type": "error_box",
            "tokens": [box],
            "coordinate_values": [],
            "fallback": True,
        },
        [torch.zeros((1, 2, 1, 1))],
        [torch.zeros((1, 2, 1, 1))],
        [10, 11],
        source_tokens,
        2,
        token_ids,
        quantized=False,
    )

    assert result is not None
    assert result["accepted_pbd_prefix"] == [box]
    assert result["termination"] == "box_end"
    assert result["resolved"]["type"] == "box"
    assert result["resolved"]["coordinate_values"] == [100, 200, 300, 400]
    assert [step["selected_token_id"] for step in result["steps"]] == source_tokens[1:]
