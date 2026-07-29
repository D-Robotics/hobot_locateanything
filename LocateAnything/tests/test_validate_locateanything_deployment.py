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
        "decode_context_policy": deployment.DECODE_CONTEXT_POLICY,
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
        "decode_context_coverage_passed": True,
        "decode_context_coverage": {
            "policy": deployment.DECODE_CONTEXT_POLICY,
            "sample_count": 300,
            "passed": True,
            "errors": [],
            "suffix_len": {"min": 0, "max": 64},
            "past_len": {"min": 600, "max": 664},
            "depth_buckets": {
                "zero": 60, "1_31": 60, "32_127": 180, "128_plus": 0,
            },
            "token_sources": {"target": 150, "prediction:hybrid": 150},
        },
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


def release_identity_fixture(tmp_path):
    generated_dir = tmp_path / "generated"
    statistics_dir = tmp_path / "statistics"
    model_dir = tmp_path / "model"
    compiler_dir = tmp_path / "compiler"
    for directory in (generated_dir, statistics_dir, model_dir, compiler_dir):
        directory.mkdir()

    selected = tmp_path / "selected.jsonl"
    generated = generated_dir / "generated.jsonl"
    selected.write_text("{}\n" * 1200, encoding="utf-8")
    generated.write_text("{}\n" * 1200, encoding="utf-8")
    prepare_identity = generated_dir / "prepare_run_identity.json"
    generation_summary = generated_dir / "generation_summary.json"

    shard_name = "model-00001-of-00001.safetensors"
    (model_dir / shard_name).write_bytes(b"checkpoint")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard_name}}), encoding="utf-8"
    )
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (compiler_dir / "entry.py").write_text("VALUE = 1\n", encoding="utf-8")
    prepare_source = compiler_dir / "scripts" / "calibration" / "prepare.py"
    prepare_source.parent.mkdir(parents=True)
    prepare_source.write_text("VALUE = 'prepare'\n", encoding="utf-8")

    fixed_profile = {"image_width": 672, "prefill_limit": 1024}
    generation_config = {"max_new_tokens": 1024}
    prepare_identity.write_text(
        json.dumps({
            "schema_version": 1,
            "selected_manifest_sha256": deployment.sha256(selected),
            "checkpoint": deployment.checkpoint_identity(model_dir),
            "tokenizer": deployment.tokenizer_identity(model_dir),
            "prepare_source": deployment.file_identity(
                prepare_source, normalize_text=True
            ),
            "fixed_profile": fixed_profile,
            "generation_config": generation_config,
        }),
        encoding="utf-8",
    )
    generation_summary.write_text(
        json.dumps({
            "selected_manifest_sha256": deployment.sha256(selected),
            "generated_manifest": generated.name,
            "generated_manifest_sha256": deployment.sha256(generated),
            "sample_count": 1200,
            "fixed_profile": fixed_profile,
            "generation_config": generation_config,
            "prepare_run_identity": prepare_identity.name,
            "prepare_run_identity_sha256": deployment.sha256(prepare_identity),
        }),
        encoding="utf-8",
    )

    run_identity = {
        "generated_manifest": deployment.file_identity(generated),
        "selected_manifest": deployment.file_identity(selected),
        "prepare_run_identity": deployment.file_identity(prepare_identity),
        "generation_summary": deployment.file_identity(generation_summary),
        "checkpoint": deployment.checkpoint_identity(model_dir),
        "tokenizer": deployment.tokenizer_identity(model_dir),
        "compiler_source": deployment.source_tree_identity(compiler_dir),
    }
    run_identity_sha = deployment.sha256_json(run_identity)
    scale = {"calibration_run_identity_sha256": run_identity_sha}
    coverage = {"calibration_run_identity_sha256": run_identity_sha}
    scale_path = statistics_dir / "calibration_scale_manifest.json"
    coverage_path = statistics_dir / "calibration_graph_coverage.json"
    scale_path.write_text(json.dumps(scale), encoding="utf-8")
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    convergence = statistics_dir / "scale_convergence.json"
    legacy_convergence = statistics_dir / "scale_convergence_512_vs_1200.json"
    convergence.write_text("{}\n", encoding="utf-8")
    legacy_convergence.write_text("{}\n", encoding="utf-8")
    durable = [scale_path, coverage_path, convergence, legacy_convergence]
    (statistics_dir / "calibration_run_identity.json").write_text(
        json.dumps({
            "status": "complete",
            "identity": run_identity,
            "artifacts": deployment.artifact_identities(durable),
        }),
        encoding="utf-8",
    )
    return {
        "generated": generated,
        "selected": selected,
        "generated_sha": deployment.sha256(generated),
        "scale": scale,
        "scale_path": scale_path,
        "coverage": coverage,
        "model": model_dir,
        "compiler": compiler_dir,
        "prepare_source": prepare_source,
        "shard": model_dir / shard_name,
    }


def release_identity_errors(state):
    return deployment.release_identity_errors(
        expected_samples=1200,
        selected_jsonl=state["selected"],
        generated_jsonl=state["generated"],
        scale_manifest_path=state["scale_path"],
        scale=state["scale"],
        coverage=state["coverage"],
        generated_sha=state["generated_sha"],
        model_path=state["model"],
        compiler_source_root=state["compiler"],
        prepare_source_path=state["prepare_source"],
        enforce_frozen_checkpoint=False,
    )


def test_preflight_accepts_consistent_chain(tmp_path, monkeypatch):
    assert run(monkeypatch, fixtures(tmp_path)) == 0


def test_release_identity_accepts_unchanged_complete_chain(tmp_path):
    assert release_identity_errors(release_identity_fixture(tmp_path)) == []


def test_release_identity_enforces_frozen_checkpoint_without_type_error(tmp_path):
    state = release_identity_fixture(tmp_path)
    errors = deployment.release_identity_errors(
        expected_samples=1200,
        selected_jsonl=state["selected"],
        generated_jsonl=state["generated"],
        scale_manifest_path=state["scale_path"],
        scale=state["scale"],
        coverage=state["coverage"],
        generated_sha=state["generated_sha"],
        model_path=state["model"],
        compiler_source_root=state["compiler"],
        prepare_source_path=state["prepare_source"],
    )

    assert any("checkpoint" in error for error in errors)


def test_release_identity_rejects_modified_scale_artifact(tmp_path):
    state = release_identity_fixture(tmp_path)
    state["scale_path"].write_text('{"changed":true}\n', encoding="utf-8")
    errors = release_identity_errors(state)
    assert any("artifact identity mismatch" in error for error in errors)


def test_release_identity_rejects_modified_checkpoint(tmp_path):
    state = release_identity_fixture(tmp_path)
    state["shard"].write_bytes(b"changed checkpoint")
    errors = release_identity_errors(state)
    assert "calibration identity checkpoint mismatch" in errors


def test_release_identity_rejects_modified_compiler_source(tmp_path):
    state = release_identity_fixture(tmp_path)
    (state["compiler"] / "entry.py").write_text("VALUE = 2\n", encoding="utf-8")
    errors = release_identity_errors(state)
    assert "calibration identity compiler source mismatch" in errors


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


def test_preflight_rejects_all_zero_decode_history(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    coverage_path = paths[3]
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    context = coverage["decode_context_coverage"]
    context["depth_buckets"] = {
        "zero": 300, "1_31": 0, "32_127": 0, "128_plus": 0,
    }
    context["suffix_len"] = {"min": 0, "max": 0}
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")

    assert run(monkeypatch, paths) == 2


def test_preflight_reports_non_numeric_decode_counts(tmp_path, monkeypatch, capsys):
    paths = fixtures(tmp_path)
    coverage_path = paths[3]
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    context = coverage["decode_context_coverage"]
    context["depth_buckets"]["32_127"] = "180"
    context["token_sources"]["target"] = None
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")

    assert run(monkeypatch, paths) == 2
    output = capsys.readouterr().out
    assert "depth_buckets contains invalid counts" in output
    assert "token_sources contains invalid counts" in output


def test_preflight_rejects_stale_language_observer_profile(tmp_path, monkeypatch):
    paths = fixtures(tmp_path)
    scale_path = paths[2]
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    scale["profile"]["language_decoder_weight_bits"] = 4
    scale_path.write_text(json.dumps(scale), encoding="utf-8")

    assert run(monkeypatch, paths) == 2


def test_preflight_rejects_disabling_calibration_hidden_rotation(tmp_path, monkeypatch):
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


def test_release_distribution_gate_requires_frozen_task_and_detection_mix():
    assert deployment.release_distribution_errors(
        1200,
        deployment.RELEASE_TASK_COUNTS,
        deployment.RELEASE_DETECTION_SOURCE_COUNTS,
    ) == []

    wrong_tasks = dict(deployment.RELEASE_TASK_COUNTS)
    wrong_tasks["detection"] -= 1
    wrong_tasks["gui"] += 1
    errors = deployment.release_distribution_errors(
        1200,
        wrong_tasks,
        {"coco_multicategory_detection": 620},
    )
    assert any("release task counts mismatch" in error for error in errors)
    assert any("release Detection source counts mismatch" in error for error in errors)


def test_non_release_fixture_does_not_inherit_release_distribution_gate():
    assert deployment.release_distribution_errors(
        None,
        {task: 50 for task in deployment.TASKS},
        {"missing": 50},
    ) == []


def test_release_checkpoint_gate_requires_512_samples():
    assert deployment.release_convergence_checkpoint_errors(1200, 512) == []
    assert (
        "expected=512"
        in deployment.release_convergence_checkpoint_errors(1200, 256)[0]
    )
    assert deployment.release_convergence_checkpoint_errors(None, 256) == []


def test_release_manifest_gate_requires_and_matches_frozen_sha():
    digest = "a" * 64
    assert deployment.selected_manifest_sha_errors(1200, digest, digest) == []
    assert "is required" in deployment.selected_manifest_sha_errors(1200, None, digest)[0]
    assert "mismatch" in deployment.selected_manifest_sha_errors(1200, "b" * 64, digest)[0]
    assert "is invalid" in deployment.selected_manifest_sha_errors(None, "short", digest)[0]


def test_non_release_manifest_gate_is_optional():
    assert deployment.selected_manifest_sha_errors(None, None, "a" * 64) == []
