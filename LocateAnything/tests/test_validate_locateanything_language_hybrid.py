from __future__ import annotations

import json

import pytest
import torch

from compiler.scripts.validate import hybrid as validation


def test_official_seed_prefers_payload_then_manifest():
    payload = {"prediction_seeds": {"hybrid": 11}}
    manifest = {"item": {"prediction": {"hybrid": {"seed": 22}}}}

    assert validation.official_seed("item", payload, manifest) == (
        11,
        "payload:prediction_seeds.hybrid",
    )
    assert validation.official_seed("item", {}, manifest) == (
        22,
        "generated.jsonl:prediction.hybrid.seed",
    )


def test_generation_config_uses_payload_metadata():
    config, source = validation.generation_config_from_payload(
        {
            "generation_config": {
                "max_new_tokens": 64,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": None,
                "repetition_penalty": 1.1,
            }
        }
    )

    assert source == "payload:generation_config"
    assert config.max_new_tokens == 64
    assert config.top_k is None


def test_generation_config_defaults_to_official_output_budget():
    config, source = validation.generation_config_from_payload({})

    assert source == validation.GENERATION_CONFIG_SOURCE
    assert config.max_new_tokens == 2048


def test_generation_metadata_prefers_prepare_metadata_over_generation_summary(tmp_path):
    tensor_dir = tmp_path / "generated" / "tensors"
    tensor_dir.mkdir(parents=True)
    (tensor_dir.parent / "prepare_job_metadata.json").write_text(
        json.dumps({"max_new_tokens": 64}), encoding="utf-8"
    )
    (tensor_dir.parent / "generation_summary.json").write_text(
        json.dumps({"max_new_tokens": 96}), encoding="utf-8"
    )

    metadata, provenance = validation.load_generation_metadata(tensor_dir)
    config, source = validation.generation_config_from_payload({}, metadata)

    assert metadata["max_new_tokens"] == 64
    assert config.max_new_tokens == 64
    assert source.startswith("prepare_job_metadata.json:max_new_tokens")
    assert len(provenance) == 2


def test_manifest_discovery_and_sequence_comparison(tmp_path):
    tensor_dir = tmp_path / "generated" / "tensors"
    tensor_dir.mkdir(parents=True)
    manifest = tensor_dir.parent / "generated.jsonl"
    manifest.write_text(
        json.dumps({"bundle_id": "item", "status": "complete"}) + "\n",
        encoding="utf-8",
    )

    path, records = validation.load_generation_manifest(tensor_dir)

    assert path == manifest.resolve()
    assert records["item"]["status"] == "complete"
    assert validation.sequence_comparison([1, 2, 3], [1, 2, 4]) == {
        "exact": False,
        "left_length": 3,
        "right_length": 3,
        "common_prefix_length": 2,
        "common_prefix_rate": 2 / 3,
    }
    assert validation.sequence_comparison([], [])["common_prefix_rate"] == 1.0


def test_manifest_reference_is_fail_closed_and_preserves_raw_answer(tmp_path):
    payload_path = tmp_path / "item.pt"
    payload_path.write_bytes(b"payload")
    digest = validation.sha256(payload_path)
    manifest = {
        "item": {
            "status": "complete",
            "tensor_file": "tensors/item.pt",
            "tensor_sha256": digest,
            "prediction": {
                "hybrid": {"answer": "raw answer", "token_ids": [1, 2]}
            },
        }
    }
    payload = {
        "bundle_id": "item",
        "prediction_token_ids": {"hybrid": torch.tensor([1, 2])},
    }

    assert validation.validate_payload_identity(payload_path, payload, manifest) == "item"
    assert validation.official_saved_reference("item", payload, manifest) == (
        [1, 2],
        "raw answer",
    )


def test_hybrid_control_flow_summary_counts_pbd_and_ar():
    samples = [
        {
            "adapted_float": {
                "steps": [
                    {"mode": "pbd", "q_len": 6, "pattern": "coord_box"},
                    {"mode": "pbd", "q_len": 9, "pattern": "error_box"},
                ],
                "pbd_fused_prefix_calls": 2,
                "q1_commit_calls": 3,
                "ar_fallback_sample_calls": 2,
                "stop_reason": "im_end",
            }
        }
    ]

    summary = validation.hybrid_control_flow_summary(samples, "adapted_float")

    assert summary["pbd_calls"] == 2
    assert summary["pbd_direct_acceptance_rate"] == 0.5
    assert summary["pbd_error_box_rate"] == 0.5
    assert summary["pbd_fused_prefix_calls"] == 2
    assert summary["q1_commit_calls"] == 3
    assert summary["ar_fallback_sample_calls"] == 2


def test_initialization_failure_writes_report(tmp_path):
    input_dir = tmp_path / "input"
    model_path = tmp_path / "model"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    model_path.mkdir()

    with pytest.raises(FileNotFoundError):
        validation.main(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--model_path",
                str(model_path),
            ]
        )

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error"]["type"] == "FileNotFoundError"
