from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "compiler" / "scripts" / "calibration" / "preflight.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(tmp_path: Path) -> tuple[Path, dict]:
    image = tmp_path / "images" / "sample.jpg"
    image.parent.mkdir()
    image.write_bytes(b"static-image-fixture")
    record = {
        "bundle_id": "sample-1",
        "image": "images/sample.jpg",
        "image_sha256": sha256(image),
        "metadata": {
            "calibration_source_role": "coco_multicategory_detection",
            "calibration_stratum": "single",
        },
        "prompt": "Locate all the instances that matches the following description: cat.",
        "split": "train",
        "target_response": "<ref>cat</ref><box><10><20><30><40></box>",
        "task": "detection",
    }
    manifest = tmp_path / "selected.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    profile = {
        "manifest_sha256": sha256(manifest),
        "sample_count": 1,
        "task_counts": {"detection": 1},
        "source_role_counts": {"coco_multicategory_detection": 1},
        "coco_stratum_counts": {"single": 1},
    }
    return manifest, profile


def test_manifest_audit_accepts_hash_checked_train_fixture(tmp_path):
    module = load_module()
    manifest, profile = write_manifest(tmp_path)
    report, records = module.audit_manifest(manifest, profile)
    assert report["passed"] is True
    assert report["unique_images"] == 1
    assert report["task_counts"] == {"detection": 1}
    assert records[0]["bundle_id"] == "sample-1"


def test_manifest_audit_rejects_a_modified_image(tmp_path):
    module = load_module()
    manifest, profile = write_manifest(tmp_path)
    (tmp_path / "images" / "sample.jpg").write_bytes(b"modified")
    with pytest.raises(module.PreflightError, match="image SHA256 mismatch"):
        module.audit_manifest(manifest, profile)


def test_manifest_audit_rejects_detection_label_not_requested_by_prompt(tmp_path):
    module = load_module()
    manifest, profile = write_manifest(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["target_response"] = "<ref>dog</ref><box><10><20><30><40></box>"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    profile["manifest_sha256"] = sha256(manifest)

    with pytest.raises(module.PreflightError, match="target references do not match"):
        module.audit_manifest(manifest, profile)


def test_model_audit_accepts_vocab_merges_tokenizer_layout(tmp_path):
    module = load_module()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "vision_config": {"patch_size": 14, "merge_kernel_size": [2, 2]},
        "text_config": {
            "vocab_size": 152681,
            "hidden_size": 2048,
            "block_size": 6,
            "text_mask_token_id": 151676,
        },
        "image_token_index": 151665,
        "coord_start_token_id": 151677,
        "coord_end_token_id": 152677,
    }), encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "processor_config.json").write_text("{}", encoding="utf-8")
    (model / "vocab.json").write_text("{}", encoding="utf-8")
    (model / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (model / "added_tokens.json").write_text(
        json.dumps(module.EXPECTED_TOKEN_IDS), encoding="utf-8"
    )
    (model / "model.safetensors").write_bytes(b"weights")
    checkpoint_sha256 = sha256(model / "model.safetensors")

    report = module.audit_model(model, {
        "patch_size": 14,
        "merge_size": 2,
        "vocab_size": 152681,
        "hidden_size": 2048,
        "image_token_id": 151665,
        "checkpoint_sha256": {"model.safetensors": checkpoint_sha256},
        "checkpoint_index_sha256": None,
    })

    assert report["tokenizer_layout"] == "bpe_vocab_merges"
    assert set(report["tokenizer_files"]) == {
        "vocab.json", "merges.txt", "added_tokens.json", "tokenizer_config.json"
    }
    assert report["checkpoint_shards"] == 1
    assert report["checkpoint_files"]["model.safetensors"]["sha256"] == checkpoint_sha256


def test_dependency_audit_uses_import_and_checks_critical_versions():
    module = load_module()
    versions = {
        distribution: expected or "unconstrained"
        for distribution, expected in module.REQUIRED_MODULES.values()
    }
    imported = []
    report = module.audit_dependencies(
        importer=lambda name: imported.append(name),
        version_lookup=lambda distribution: versions[distribution],
    )
    assert imported == list(module.REQUIRED_MODULES)
    assert report["checks"] == "module_import_then_distribution_version"
    assert "pip check" in report["metadata_notes"]["decord"]


def test_dependency_audit_rejects_missing_and_wrong_versions():
    module = load_module()

    def importer(name):
        if name == "decord":
            raise ImportError("missing")

    def version(distribution):
        if distribution == "transformers":
            return "4.57.1"
        for candidate_distribution, expected in module.REQUIRED_MODULES.values():
            if candidate_distribution == distribution:
                return expected or "unconstrained"
        raise AssertionError(distribution)

    with pytest.raises(module.PreflightError) as error:
        module.audit_dependencies(importer=importer, version_lookup=version)
    assert "missing modules: decord" in str(error.value)
    assert "transformers==4.57.1 (required 4.57.6)" in str(error.value)


def test_processor_contract_expands_exactly_576_visual_tokens():
    module = load_module()

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            image_tokens = text.count("<IMG_CONTEXT>")
            return {"input_ids": [151665] * image_tokens + [7, 8]}

    processor = SimpleNamespace(
        image_processor=SimpleNamespace(patch_size=14, merge_kernel_size=[2, 2]),
        tokenizer=Tokenizer(),
        image_token="<IMG_CONTEXT>",
        image_token_id=151665,
        image_start_token="<img>",
        image_end_token="</img>",
        py_apply_chat_template=lambda messages, **_kwargs: (
            "<image-1>" + messages[0]["content"][1]["text"]
        ),
    )
    profile = {
        "patch_size": 14,
        "merge_size": 2,
        "image_token_id": 151665,
        "visual_tokens": 576,
        "prefill_limit": 1024,
        "max_new_tokens": 1024,
    }
    report = module.audit_processor(
        Path("unused"),
        [{"bundle_id": "sample-1", "prompt": "find cat", "target_response": "<box>None</box>"}],
        profile,
        loader=lambda *_args, **_kwargs: processor,
    )
    assert report["visual_tokens"] == 576
    assert report["max_prefill"]["tokens"] == 578
    assert report["model_weights_loaded"] is False
    assert report["gpu_inference"] is False


def test_flatten_ids_accepts_batch_encoding_mapping():
    module = load_module()
    encoded = UserDict({"input_ids": [[11, 12, 13]]})
    assert module.flatten_ids(encoded) == [11, 12, 13]


def test_runtime_tokenizer_must_match_checkpoint_tokenizer(tmp_path):
    module = load_module()
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")

    class CheckpointTokenizer:
        def __call__(self, value, **_kwargs):
            return UserDict({"input_ids": [ord(character) for character in value]})

    class RuntimeEncoding:
        def __init__(self, value):
            self.ids = [ord(character) for character in value]

    class RuntimeTokenizer:
        def encode(self, value, **_kwargs):
            return RuntimeEncoding(value)

    records = [{"bundle_id": "sample-1", "prompt": "cat", "target_response": "box"}]
    report = module.audit_runtime_tokenizer(
        CheckpointTokenizer(),
        records,
        tokenizer_json,
        loader=lambda _path: RuntimeTokenizer(),
    )
    assert report["texts_checked"] == 2
    assert report["regex_contract"] == "checkpoint_default_matches_runtime_tokenizer_json"


def test_representative_image_check_runs_real_letterbox_processor_contract(tmp_path):
    module = load_module()
    from PIL import Image

    image_path = tmp_path / "wide.jpg"
    Image.new("RGB", (320, 120), (10, 20, 30)).save(image_path)

    class PixelValues:
        shape = (2304, 3, 14, 14)

    class Processor:
        image_token_id = 151665

        def py_apply_chat_template(self, _messages, **_kwargs):
            return "<image-1>find cat"

        def process_vision_info(self, messages):
            prepared = messages[0]["content"][0]["image"]
            assert prepared.size == (672, 672)
            return [prepared], None

        def __call__(self, **_kwargs):
            return {
                "pixel_values": PixelValues(),
                "image_grid_hws": [[48, 48]],
                "input_ids": [[151665] * 576 + [7, 8]],
            }

    profile = {
        "task_counts": {"detection": 1},
        "image_width": 672,
        "image_height": 672,
        "resize_mode": "letterbox",
        "letterbox_fill": 128,
        "patch_size": 14,
        "patch_count": 2304,
        "image_token_id": 151665,
        "visual_tokens": 576,
    }
    report = module.audit_representative_images(
        Processor(),
        [{
            "bundle_id": "sample-1",
            "task": "detection",
            "prompt": "find cat",
            "_preflight_image_path": str(image_path),
        }],
        profile,
    )

    assert report["detection"]["source_size"] == [320, 120]
    assert report["detection"]["prepared_size"] == [672, 672]
    assert report["detection"]["pixel_values_shape"] == [2304, 3, 14, 14]
    assert report["detection"]["image_grid_hws"] == [[48, 48]]
    assert report["detection"]["visual_tokens"] == 576


def test_preflight_source_never_imports_or_loads_automodel():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "AutoModel" not in source
    assert "cuda" not in source.lower()
