from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "compiler/leap_llm/apis/calibration/locateanything_replay.py"
)
SPEC = importlib.util.spec_from_file_location("locateanything_replay", MODULE_PATH)
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(replay)

ROPE_PATH = (
    Path(__file__).resolve().parents[1]
    / "compiler/leap_llm/models/locateanything/utils/rope_2d.py"
)
ROPE_SPEC = importlib.util.spec_from_file_location("locateanything_rope_2d", ROPE_PATH)
rope_2d = importlib.util.module_from_spec(ROPE_SPEC)
assert ROPE_SPEC.loader is not None
ROPE_SPEC.loader.exec_module(rope_2d)


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 4)


def test_prefill_replaces_all_image_placeholders_and_right_aligns_mask():
    model = TinyText()
    payload = {
        "prompt_input_ids": torch.tensor([[2, 9, 9, 3]]),
        "prompt_attention_mask": torch.ones(1, 4, dtype=torch.long),
        "projected_visual_features": torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]),
    }
    embeds, positions, mask, active = replay.build_prefill_inputs(
        model, payload, torch.eye(4), chunk_size=6, cache_len=10,
        image_token_id=9, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert active == 4
    assert torch.equal(embeds[0, 1:3], payload["projected_visual_features"][0])
    assert positions.shape == (1, 1, 6)
    assert torch.all(mask[0, 0, 3, 4:8] == 0)
    assert torch.all(mask[0, 0, 0, :4] < 0)


def test_prefill_appends_teacher_forced_suffix_without_treating_it_as_visual():
    model = TinyText()
    payload = {
        "prompt_input_ids": torch.tensor([[2, 9, 3]]),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
        "projected_visual_features": torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
    }
    embeds, _, _, active = replay.build_prefill_inputs(
        model,
        payload,
        torch.eye(4),
        chunk_size=6,
        cache_len=10,
        image_token_id=9,
        device=torch.device("cpu"),
        dtype=torch.float32,
        suffix_token_ids=[4, 5],
    )

    assert active == 5
    assert torch.equal(embeds[0, 1], payload["projected_visual_features"][0, 0])
    assert torch.equal(embeds[0, 3:5], model.embed_tokens(torch.tensor([4, 5])))


def test_decode_mask_uses_right_aligned_history_and_pbd_window():
    model = TinyText()
    _, positions, mask = replay.build_decode_inputs(
        model, [1, 2, 3], q_len=3, past_len=4, cache_len=12,
        is_pbd=True, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert positions.tolist() == [[[3, 4, 5]]]
    assert torch.all(mask[:, :, :, 5:8] == 0)
    assert torch.all(mask[:, :, :, 9:12] == 0)
    assert torch.all(mask[:, :, :, 8] < 0)


def test_decode_pbd_prefix_keeps_real_tokens_causal_before_mtp_window():
    model = TinyText()
    _, positions, mask = replay.build_decode_inputs(
        model,
        [20, 21, 21, 15, 15],
        q_len=5,
        past_len=4,
        cache_len=12,
        is_pbd=True,
        pbd_prefix_len=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Prefix positions are normal causal positions. The trailing MTP anchor
    # duplicates the final prefix position, then its masks advance normally.
    assert positions.tolist() == [[[4, 5, 5, 6, 7]]]
    assert torch.all(mask[:, :, 0, 7] == 0)
    assert torch.all(mask[:, :, 0, 8:] < 0)
    assert torch.all(mask[:, :, 1, 7:9] == 0)
    assert torch.all(mask[:, :, 1, 9:] < 0)
    # The duplicated final prefix token is masked for the trailing PBD rows;
    # those rows attend the earlier prefix and their own PBD window.
    assert torch.all(mask[:, :, 2:, 7] == 0)
    assert torch.all(mask[:, :, 2:, 8] < 0)
    assert torch.all(mask[:, :, 2:, 9:12] == 0)
    assert torch.all(mask[:, :, :, 3:7] == 0)


def test_q9_deep_context_positions_and_mask_keep_prefix_causal():
    model = TinyText()
    _, positions, mask = replay.build_decode_inputs(
        model,
        [20, 21, 22, 22, 15, 15, 15, 15, 15],
        q_len=9,
        past_len=20,
        cache_len=64,
        is_pbd=True,
        pbd_prefix_len=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert positions.tolist() == [[[20, 21, 22, 22, 23, 24, 25, 26, 27]]]
    # Cached history occupies [35, 55). Prefix positions remain causal.
    assert torch.all(mask[:, :, :, 35:55] == 0)
    assert torch.all(mask[:, :, 0, 55:56] == 0)
    assert torch.all(mask[:, :, 0, 56:] < 0)
    assert torch.all(mask[:, :, 2, 55:58] == 0)
    assert torch.all(mask[:, :, 2, 58:] < 0)
    # The six-position PBD window is bidirectional and excludes its duplicate.
    assert torch.all(mask[:, :, 3:, 58:64] == 0)
    assert torch.all(mask[:, :, 3:, 57] < 0)


def test_pbd_tokens_match_native_anchor_and_mask_protocol():
    payload = {
        "prompt_input_ids": torch.tensor([[11, 12, 13]]),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
        "prediction_token_ids": {"hybrid": torch.tensor([21, 22, 23])},
        "target_token_ids": torch.tensor([31, 32, 33]),
    }

    assert replay.select_pbd_tokens(payload, 6, 151676) == [
        13,
        151676,
        151676,
        151676,
        151676,
        151676,
    ]

    assert replay.select_pbd_tokens(
        payload, 6, 151676, anchor_token_id=23
    ) == [23, 151676, 151676, 151676, 151676, 151676]


def test_decode_context_is_structurally_deep_and_resume_deterministic():
    ref_start, ref_end = 20, 21
    box_start, box_end = 22, 23
    target = [
        ref_start, 7, ref_end, box_start, 1, 2, 3, 4, box_end,
        ref_start, 8, ref_end, box_start, 5, 6, 7, 8, box_end,
        ref_start, 9, ref_end, box_start, 9, 10, 11, 12, box_end,
    ]
    payload = {
        "prompt_input_ids": torch.tensor([[2, 3, 4]]),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
        "prediction_token_ids": {"hybrid": torch.tensor([], dtype=torch.long)},
        "target_token_ids": torch.tensor(target),
        "special_token_ids": {"<ref>": ref_start, "<box>": box_start},
    }
    for index in range(1000):
        payload["bundle_id"] = f"deep-sample-{index}"
        first = replay.select_decode_replay_context(payload, chunk_size=64)
        if first.selection_slot == 4:
            break
    else:  # pragma: no cover - SHA256 slots make this unreachable in practice.
        raise AssertionError("could not find a tail-selection bundle id")

    # Calling another sample between attempts models a resumed/shuffled replay.
    other = dict(payload, bundle_id="different-sample")
    replay.select_decode_replay_context(other, chunk_size=64)
    second = replay.select_decode_replay_context(payload, chunk_size=64)

    assert first == second
    assert first.token_source == "target"
    assert first.suffix_len > 0
    assert first.pending_token_ids[0] in {ref_start, box_start}
    assert list(first.suffix_token_ids) == target[: first.suffix_len]
    assert list(first.pending_token_ids) == target[first.suffix_len : first.suffix_len + 6]
    assert first.anchor_token_id == target[first.suffix_len - 1]
    assert first.past_len == 3 + first.suffix_len


def test_decode_context_repeats_short_output_only_for_pending_workspace():
    payload = {
        "bundle_id": "short-output",
        "prompt_input_ids": torch.tensor([[2, 3, 4]]),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
        "prediction_token_ids": {"hybrid": torch.tensor([7, 8])},
        "target_token_ids": torch.tensor([9]),
        "special_token_ids": {"<ref>": 20, "<box>": 22},
    }

    context = replay.select_decode_replay_context(payload, chunk_size=8)

    assert context.suffix_token_ids == ()
    assert context.pending_token_ids == (7, 8, 9, 7, 8, 9)
    assert context.past_len == 3
    assert context.token_source == "fallback:hybrid+target"


def test_decode_context_rejects_output_without_any_replay_tokens():
    payload = {
        "bundle_id": "empty-output",
        "prompt_input_ids": torch.tensor([[2, 3, 4]]),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
        "prediction_token_ids": {"hybrid": torch.tensor([], dtype=torch.long)},
        "target_token_ids": torch.tensor([], dtype=torch.long),
        "special_token_ids": {"<ref>": 20, "<box>": 22},
    }

    with __import__("pytest").raises(ValueError, match="no tokens available"):
        replay.select_decode_replay_context(payload, chunk_size=8)


def test_decode_context_at_chunk_limit_keeps_pending_tokens_outside_prefill():
    payload = {
        "bundle_id": "chunk-limit",
        "prompt_input_ids": torch.tensor([[2, 3, 4, 5]]),
        "prompt_attention_mask": torch.ones(1, 4, dtype=torch.long),
        "prediction_token_ids": {"hybrid": torch.tensor([], dtype=torch.long)},
        "target_token_ids": torch.tensor([22, 1, 2, 3, 4, 23]),
        "special_token_ids": {"<ref>": 20, "<box>": 22},
    }

    context = replay.select_decode_replay_context(payload, chunk_size=4)

    assert context.max_suffix_len == 0
    assert context.suffix_token_ids == ()
    assert context.pending_token_ids == (22, 1, 2, 3, 4, 23)
    assert context.past_len == 4


def _context_record(
    bundle_id: str,
    *,
    suffix_len: int,
    source: str = "target",
) -> dict:
    return {
        "bundle_id": bundle_id,
        "task": "detection",
        "token_source": source,
        "selection_slot": 0 if suffix_len == 0 else 4,
        "boundary_token_id": 22,
        "prompt_len": 10,
        "max_suffix_len": 128,
        "offset": suffix_len,
        "suffix_len": suffix_len,
        "past_len": 10 + suffix_len,
        "depth_bucket": replay.decode_depth_bucket(suffix_len),
    }


def test_decode_context_coverage_rejects_all_zero_suffixes():
    coverage = replay.summarize_decode_context_coverage(
        [_context_record("a", suffix_len=0), _context_record("b", suffix_len=0)],
        expected_samples=2,
        cache_len=256,
    )

    assert coverage["passed"] is False
    assert coverage["depth_buckets"]["zero"] == 2
    assert any("no nonzero history suffix" in error for error in coverage["errors"])


def test_decode_context_coverage_accepts_shallow_and_deep_histories():
    rows = [
        _context_record("a", suffix_len=0),
        _context_record("b", suffix_len=64, source="prediction:hybrid"),
    ]
    first = replay.summarize_decode_context_coverage(
        rows, expected_samples=2, cache_len=256
    )
    second = replay.summarize_decode_context_coverage(
        list(reversed(rows)), expected_samples=2, cache_len=256
    )

    assert first["passed"] is True
    assert first["depth_buckets"]["32_127"] == 1
    assert first["token_sources"] == {"prediction:hybrid": 1, "target": 1}
    assert first["selection_sha256"] == second["selection_sha256"]


def test_right_aligned_cache_preserves_only_active_prefill_tokens():
    key = torch.arange(1 * 6 * 2 * 2).reshape(1, 6, 2, 2).float()
    keys, values = replay.build_right_aligned_caches([key], [key + 100], active_len=4, cache_len=9)
    assert torch.equal(keys[0][:, -4:], key[:, :4])
    assert torch.count_nonzero(keys[0][:, :-4]) == 0
    assert torch.equal(values[0][:, -4:], key[:, :4] + 100)


def test_snapshot_convergence_reports_relative_drift():
    result = replay.compare_snapshots(
        {"a": {"kind": "ConstFakeQuant", "absmax": 8.0}},
        {"a": {"kind": "ConstFakeQuant", "absmax": 10.0}},
    )
    assert result["layers"][0]["relative_drift"] == 0.2
    assert result["outliers_over_10pct"] == ["a"]


def test_moonvit_rope_matches_upstream_adjacent_complex_pairs():
    torch.manual_seed(7)
    q = torch.randn(2, 6, 8)
    k = torch.randn(2, 6, 8)
    freqs = rope_2d.gather_freqs_by_grid(
        rope_2d.precompute_freqs_cos_sin(8, 8, dim=8), grid_h=2, grid_w=3
    )
    expected_q, expected_k = rope_2d.apply_rope_real(
        q.transpose(0, 1), k.transpose(0, 1), freqs
    )
    cos = freqs[..., 0].repeat_interleave(2, dim=-1)
    sin = freqs[..., 1].repeat_interleave(2, dim=-1)
    actual_q = q * cos + rope_2d.rotate_adjacent_pairs(q) * sin
    actual_k = k * cos + rope_2d.rotate_adjacent_pairs(k) * sin
    assert torch.allclose(actual_q.transpose(0, 1), expected_q, atol=1e-6)
    assert torch.allclose(actual_k.transpose(0, 1), expected_k, atol=1e-6)


def test_apply_scale_manifest_restores_absmax_and_rmsnorm(tmp_path):
    class FakeQuant(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("absmax", torch.tensor(0.0))

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.scale = 1.0
            self.summax_hidden = None
            self.register_buffer("i_scale", torch.tensor(1.0))
            self.register_buffer("i_scale_pow", torch.tensor(1.0))

    model = nn.Module()
    model.fq = FakeQuant()
    model.norm = Norm()
    manifest = {
        "sample_count": 512,
        "generated_manifest_sha256": "abc",
        "vision": {"512": {
            "fq": {"kind": "ConstFakeQuant", "absmax": 7.5},
            "norm": {"kind": "RMSNorm", "scale": 2.0, "summax_hidden": 11.0},
        }},
    }
    path = tmp_path / "scales.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    result = replay.apply_scale_manifest(model, path, "vision")
    assert result["applied_modules"] == 2
    assert model.fq.absmax.item() == 7.5
    assert model.norm.scale == 2.0
    assert model.norm.i_scale.item() == 0.5
    assert model.norm.i_scale_pow.item() == 0.25


def test_apply_scale_manifest_ignores_only_retired_dynamic_attention_observers(tmp_path):
    model = nn.Module()
    manifest = {
        "sample_count": 820,
        "vision": {
            "820": {
                "blocks.0.qk_matmul.x_fake_quant": {
                    "kind": "ConstFakeQuant",
                    "absmax": 8.0,
                },
                "blocks.0.wv_matmul.y_fake_quant": {
                    "kind": "ConstFakeQuant",
                    "absmax": 4.0,
                },
            }
        },
    }
    path = tmp_path / "scales.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    result = replay.apply_scale_manifest(model, path, "vision")

    assert result["applied_modules"] == 0
    assert result["ignored_dynamic_attention_observers"] == 2


def test_apply_scale_manifest_still_rejects_other_unknown_modules(tmp_path):
    model = nn.Module()
    manifest = {
        "sample_count": 1,
        "vision": {
            "1": {
                "blocks.0.unexpected_fake_quant": {
                    "kind": "ConstFakeQuant",
                    "absmax": 1.0,
                }
            }
        },
    }
    path = tmp_path / "scales.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    with __import__("pytest").raises(ValueError, match="unknown modules"):
        replay.apply_scale_manifest(model, path, "vision")


def test_apply_scale_manifest_rejects_current_quantized_module_without_scale(tmp_path):
    class FakeQuant(nn.Module):
        def __init__(self):
            super().__init__()
            self.quantized = True
            self.register_buffer("absmax", torch.tensor(0.0))

    model = nn.Module()
    model.first = FakeQuant()
    model.added_after_calibration = FakeQuant()
    manifest = {
        "sample_count": 1200,
        "vision": {
            "1200": {
                "first": {"kind": "ConstFakeQuant", "absmax": 2.0},
            }
        },
    }
    path = tmp_path / "scales.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    with __import__("pytest").raises(
        ValueError, match="missing current quantized modules.*added_after_calibration"
    ):
        replay.apply_scale_manifest(model, path, "vision")


def test_apply_scale_manifest_does_not_require_disabled_fake_quant(tmp_path):
    class FakeQuant(nn.Module):
        def __init__(self, quantized):
            super().__init__()
            self.quantized = quantized
            self.register_buffer("absmax", torch.tensor(0.0))

    model = nn.Module()
    model.enabled = FakeQuant(True)
    model.disabled = FakeQuant(False)
    manifest = {
        "sample_count": 1200,
        "language": {
            "1200": {
                "enabled": {"kind": "ConstFakeQuant", "absmax": 3.0},
            }
        },
    }
    path = tmp_path / "scales.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    result = replay.apply_scale_manifest(model, path, "language")

    assert result["applied_modules"] == 1
    assert model.enabled.absmax.item() == 3.0
