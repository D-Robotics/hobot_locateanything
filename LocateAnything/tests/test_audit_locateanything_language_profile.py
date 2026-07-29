import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "compiler" / "scripts" / "calibration/audit_profile.py"
SPEC = importlib.util.spec_from_file_location("language_profile_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_bundle(root: Path, *, prompt_tokens: int = 580) -> Path:
    tensors = root / "tensors"
    tensors.mkdir(parents=True)
    tensor_path = tensors / "sample.pt"
    input_ids = torch.full((prompt_tokens,), 42, dtype=torch.int64)
    input_ids[:576] = 151665
    torch.save({
        "prompt_input_ids": input_ids,
        "prompt_attention_mask": torch.ones(prompt_tokens, dtype=torch.int64),
        "vision_input": torch.zeros(1, 2304, 588),
        "projected_visual_features": torch.zeros(1, 576, 2048),
        "prediction_token_ids": {"hybrid": torch.arange(6), "slow": torch.arange(1)},
        "target_token_ids": torch.arange(6),
    }, tensor_path)
    digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest = root / "generated.jsonl"
    manifest.write_text(json.dumps({
        "bundle_id": "sample",
        "status": "complete",
        "task": "detection",
        "tensor_file": "tensors/sample.pt",
        "tensor_sha256": digest,
    }) + "\n", encoding="utf-8")
    return manifest


def test_profile_audit_passes_and_reports_headroom(tmp_path):
    manifest = write_bundle(tmp_path / "bundle")
    output = tmp_path / "report"
    args = MODULE.parser().parse_args([
        "--generated-jsonl", str(manifest),
        "--output-dir", str(output),
    ])

    assert MODULE.run(args) == 0
    report = json.loads((output / "language_profile_audit.json").read_text())
    assert report["passed"] is True
    assert report["summary"][0]["max"] == 580
    assert report["summary"][0]["prefill_headroom_min"] == 444
    assert report["summary"][0]["cache_headroom_after_pbd_min"] == 3510


def test_profile_audit_rejects_prompt_over_chunk(tmp_path):
    manifest = write_bundle(tmp_path / "bundle", prompt_tokens=1025)
    args = MODULE.parser().parse_args([
        "--generated-jsonl", str(manifest),
        "--output-dir", str(tmp_path / "report"),
    ])

    try:
        MODULE.run(args)
    except RuntimeError as error:
        assert "failed for 1 samples" in str(error)
    else:
        raise AssertionError("expected profile audit failure")
