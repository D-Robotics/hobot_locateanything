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


def test_replay_prefix_is_box_aligned_and_deterministic():
    payload = {
        "bundle_id": "sample-1",
        "prediction_token_ids": {"hybrid": torch.tensor([21, 22, 23])},
        "target_token_ids": torch.tensor([7, 151668, 101, 102, 103, 104, 151669, 8]),
        "special_token_ids": {"<box>": 151668},
    }

    assert replay.select_replay_prefix_tokens(payload, 4) == [151668, 101, 102, 103]
    assert replay.select_replay_prefix_tokens(payload, 6) == [151668, 101, 102, 103, 104, 151669]


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
