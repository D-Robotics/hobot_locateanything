import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "compiler" / "scripts" / "validate/deployment.py"
spec = importlib.util.spec_from_file_location("validate_locateanything_deployment", SCRIPT)
deployment = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(deployment)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixtures(tmp_path):
    tasks = deployment.TASKS
    selected = []
    for index in range(300):
        task = tasks[index % len(tasks)]
        selected.append({
            "bundle_id": f"sample-{index:04d}",
            "task": task,
            "prompt": f"prompt {index}",
            "target_response": "<ref>x</ref><box><1><2><3><4></box>",
            "image_sha256": f"{index:064x}",
        })
    profile = {
        "image_width": 672,
        "image_height": 672,
        "resize_mode": "letterbox",
        "patch_count": 2304,
        "visual_token_count": 576,
        "prefill_limit": 1024,
        "pbd_block_size": 6,
    }
    language_profile = {
        "language_decoder_weight_bits": 8,
        "language_lm_head_weight_bits": 8,
        "text_mask_token_id": 151676,
        "pbd_block_size": 6,
        "pbd_total_query_lengths": list(range(6, 13)),
        "ar_total_query_lengths": list(range(1, 6)),
        "pbd_q6_role": "post_prefill_bootstrap_only",
        "pbd_input_protocol": "accepted_prefix_plus_duplicated_anchor_plus_5_text_masks",
    }
    generated = [
        dict(
            row,
            tensor_file=f"tensors/{row['bundle_id']}.pt",
            fixed_profile=profile,
            prediction={"hybrid": {}, **({"slow": {}} if index < 10 else {})},
        )
        for index, row in enumerate(selected)
    ]
    selected_path = tmp_path / "selected.jsonl"
    generated_path = tmp_path / "generated.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(generated_path, generated)
    counts = {task: 50 for task in tasks}
    scale_path = tmp_path / "scale.json"
    scale_path.write_text(json.dumps({
        "generated_manifest_sha256": deployment.sha256(generated_path),
        "sample_count": 300,
        "checkpoint_samples": 256,
        "task_counts": counts,
        "rotation_source": deployment.DEFAULT_ROTATION_NAME,
        "rotation_file_sha256": None,
        "profile": language_profile,
        "vision": {
            "256": {"fq": {"kind": "ConstFakeQuant", "absmax": 1.0}},
            "300": {"fq": {"kind": "ConstFakeQuant", "absmax": 1.0}},
        },
        "language": {
            "256": {"fq": {"kind": "ConstFakeQuant", "absmax": 1.0}},
            "300": {"fq": {"kind": "ConstFakeQuant", "absmax": 1.0}},
        },
    }), encoding="utf-8")
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(json.dumps({
        "generated_manifest_sha256": deployment.sha256(generated_path),
        "sample_count": 300,
        "checkpoint_samples": 256,
        "task_counts": counts,
        "profile": language_profile,
        "expected_stages": list(deployment.GRAPH_STAGES),
        "stage_sample_counts": {stage: 300 for stage in deployment.GRAPH_STAGES},
        "all_stages_executed": True,
        "observer_audit_passed": True,
        "observer_audit": {
            "vision": {
                "passed": True, "observer_count": 1,
                "unexecuted": [], "zero_absmax": [], "invalid_norm": [],
            },
            "language": {
                "passed": True, "observer_count": 1,
                "unexecuted": [], "zero_absmax": [], "invalid_norm": [],
            },
        },
    }), encoding="utf-8")
    return selected_path, generated_path, scale_path, coverage_path


def run(monkeypatch, paths):
    selected, generated, scale, coverage = paths
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--selected-jsonl", str(selected),
        "--generated-jsonl", str(generated),
        "--scale-manifest", str(scale),
        "--coverage-json", str(coverage),
        "--image-width", "672", "--image-height", "672",
        "--chunk-size", "1024", "--cache-len", "4096",
        "--decode-seq-len", "6",
    ])
    return deployment.main()


def test_preflight_accepts_consistent_chain(tmp_path, monkeypatch):
    assert run(monkeypatch, fixtures(tmp_path)) == 0


def test_preflight_rejects_scale_from_other_generated_manifest(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    scale_path = paths[2]
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    scale["generated_manifest_sha256"] = "0" * 64
    scale_path.write_text(json.dumps(scale), encoding="utf-8")
    assert run(monkeypatch, paths) == 2


def test_preflight_rejects_incomplete_observer_coverage(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    coverage_path = paths[3]
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["observer_audit_passed"] = False
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    assert run(monkeypatch, paths) == 2


def test_preflight_rejects_coverage_from_other_generated_manifest(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    coverage_path = paths[3]
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["generated_manifest_sha256"] = "0" * 64
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    assert run(monkeypatch, paths) == 2


def test_preflight_rejects_missing_full_scale_snapshot(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    scale_path = paths[2]
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    del scale["language"]["300"]
    scale_path.write_text(json.dumps(scale), encoding="utf-8")
    assert run(monkeypatch, paths) == 2


def test_preflight_rejects_wrong_four_path_counts(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    coverage_path = paths[3]
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["stage_sample_counts"]["pbd_q6"] = 299
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    assert run(monkeypatch, paths) == 2


def test_preflight_rejects_stale_language_observer_profile(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    scale_path = paths[2]
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    scale["profile"]["language_decoder_weight_bits"] = 4
    scale_path.write_text(json.dumps(scale), encoding="utf-8")

    assert run(monkeypatch, paths) == 2


def test_preflight_rejects_disabling_d4_hidden_rotation(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    selected, generated, scale, coverage = paths
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--selected-jsonl", str(selected),
        "--generated-jsonl", str(generated),
        "--scale-manifest", str(scale),
        "--coverage-json", str(coverage),
        "--image-width", "672", "--image-height", "672",
        "--chunk-size", "1024", "--cache-len", "4096",
        "--decode-seq-len", "6", "--disable-hidden-rotation",
    ])
    assert deployment.main() == 2
