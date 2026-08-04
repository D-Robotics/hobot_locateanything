#!/usr/bin/env python3
"""Validate frozen LocateAnything Prepare inputs without loading model weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compiler.configuration import (  # noqa: E402
    ConfigurationFileError,
    load_config_file,
)

RUNTIME_TOKENIZER_JSON = PROJECT_ROOT / "deploy" / "tokenizer" / "tokenizer.json"

REQUIRED_MODULES = {
    "torch": ("torch", None),
    "torchvision": ("torchvision", None),
    "transformers": ("transformers", "4.57.6"),
    "tokenizers": ("tokenizers", "0.22.2"),
    "accelerate": ("accelerate", "1.12.0"),
    "peft": ("peft", "0.12.0"),
    "PIL": ("Pillow", None),
    "numpy": ("numpy", "2.2.6"),
    "cv2": ("opencv-python-headless", None),
    "lmdb": ("lmdb", "2.2.1"),
    "decord": ("decord", "0.6.0"),
    "packaging": ("packaging", None),
    "requests": ("requests", None),
    "tqdm": ("tqdm", None),
}
EXPECTED_TOKEN_IDS = {
    "<IMG_CONTEXT>": 151665,
    "<box>": 151668,
    "</box>": 151669,
    "<ref>": 151672,
    "</ref>": 151673,
    "<text_mask>": 151676,
    "<0>": 151677,
    "<1000>": 152677,
    "<null>": 152678,
    "<switch>": 152679,
}
BOX_RE = re.compile(r"<box>(.*?)</box>")
COORD_RE = re.compile(r"<([0-9]{1,4})>")


class PreflightError(RuntimeError):
    """Raised when a release input violates the static Prepare contract."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read JSON {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise PreflightError(f"{path}:{line_number}: row is not an object")
                value["_line"] = line_number
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read JSONL {path}: {exc}") from exc
    return records


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_contract(path: Path) -> dict[str, Any]:
    try:
        return load_config_file(path)
    except ConfigurationFileError as exc:
        raise PreflightError(f"cannot read release config {path}: {exc}") from exc


def integer_counts(value: Any, name: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PreflightError(f"{name} must be a mapping")
    counts: dict[str, int] = {}
    for key, count in value.items():
        if type(count) is not int or count < 0:
            raise PreflightError(f"{name}.{key} must be a non-negative integer")
        counts[str(key)] = count
    return counts


def calibration_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    calibration = config.get("calibration")
    model = config.get("model")
    language = config.get("language")
    if not all(isinstance(value, dict) for value in (calibration, model, language)):
        raise PreflightError("config requires calibration, model, and language mappings")
    assert isinstance(calibration, dict)
    assert isinstance(model, dict)
    assert isinstance(language, dict)

    task_counts = integer_counts(calibration.get("task_counts"), "calibration.task_counts")
    role_counts = integer_counts(
        calibration.get("source_role_counts"), "calibration.source_role_counts"
    )
    strata = integer_counts(
        calibration.get("coco_stratum_counts"), "calibration.coco_stratum_counts"
    )
    sample_setting = calibration.get("sample_count", "auto")
    samples = None if sample_setting == "auto" else int(sample_setting)
    if samples is not None and samples <= 0:
        raise PreflightError("calibration.sample_count must be positive or auto")
    if task_counts and samples is not None and sum(task_counts.values()) != samples:
        raise PreflightError("calibration.task_counts must sum to sample_count")
    if role_counts and samples is not None and sum(role_counts.values()) != samples:
        raise PreflightError("calibration.source_role_counts must sum to sample_count")
    if strata and role_counts and sum(strata.values()) != role_counts.get("coco_detection"):
        raise PreflightError("COCO strata must sum to the COCO source-role count")

    width = int(model.get("image_width", 0))
    height = int(model.get("image_height", 0))
    resize_mode = str(model.get("resize_mode") or "")
    letterbox_fill = int(model.get("letterbox_fill", -1))
    patch = int(model.get("patch_size", 0))
    merge = int(model.get("spatial_merge", 0))
    if min(width, height, patch, merge) <= 0 or width % patch or height % patch:
        raise PreflightError("invalid fixed Vision dimensions")
    patches = (width // patch) * (height // patch)
    visual_tokens = patches // (merge * merge)
    prefill = int(language.get("chunk_size", 0))
    if patches != 2304 or visual_tokens != 576 or prefill != 1024:
        raise PreflightError("release processor contract requires 2304 patches/576 tokens/q1024")
    if resize_mode != "letterbox" or letterbox_fill != 128:
        raise PreflightError("release image preprocessing requires letterbox with fill=128")

    max_new_tokens = int(calibration.get("max_new_tokens", 0))
    if max_new_tokens != 1024:
        raise PreflightError("release prepare max_new_tokens must be 1024")
    return {
        "sample_count": samples,
        "task_counts": task_counts,
        "source_role_counts": role_counts,
        "coco_stratum_counts": strata,
        "image_width": width,
        "image_height": height,
        "resize_mode": resize_mode,
        "letterbox_fill": letterbox_fill,
        "patch_size": patch,
        "merge_size": merge,
        "patch_count": patches,
        "visual_tokens": visual_tokens,
        "hidden_size": int(model.get("hidden_size", 0)),
        "vocab_size": int(model.get("vocab_size", 0)),
        "image_token_id": int(calibration.get("image_token_id", -1)),
        "prefill_limit": prefill,
        "max_new_tokens": max_new_tokens,
    }


def audit_dependencies(
    importer: Callable[[str], Any] = importlib.import_module,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    missing: list[str] = []
    mismatched: list[str] = []
    versions: dict[str, str | None] = {}
    for module, (distribution, expected) in REQUIRED_MODULES.items():
        try:
            importer(module)
        except Exception:
            missing.append(module)
            versions[module] = None
            continue
        try:
            installed = version_lookup(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = "unknown"
        versions[module] = installed
        if expected is not None and installed != expected:
            mismatched.append(f"{distribution}=={installed} (required {expected})")
    if missing or mismatched:
        parts = []
        if missing:
            parts.append("missing modules: " + ", ".join(missing))
        if mismatched:
            parts.append("version mismatch: " + ", ".join(mismatched))
        raise PreflightError("; ".join(parts))
    return {
        "passed": True,
        "versions": versions,
        "checks": "module_import_then_distribution_version",
        "metadata_notes": {
            "decord": (
                "decord 0.6.0 may be reported as unsupported by pip check on this "
                "platform; successful module import is the operational requirement"
            )
        },
    }


def validate_target(response: str, context: str) -> None:
    if not response:
        raise PreflightError(f"{context}: empty target_response")
    blocks = BOX_RE.findall(response)
    if not blocks:
        raise PreflightError(f"{context}: target_response has no <box> block")
    for body in blocks:
        if body.strip().lower() == "none":
            continue
        coordinates = [int(value) for value in COORD_RE.findall(body)]
        reconstructed = "".join(f"<{value}>" for value in coordinates)
        if len(coordinates) not in {2, 4} or reconstructed != body:
            raise PreflightError(f"{context}: malformed coordinate block {body!r}")
        if any(value > 1000 for value in coordinates):
            raise PreflightError(f"{context}: coordinate outside [0, 1000]")


def audit_manifest(path: Path, profile: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise PreflightError("Pillow is required to decode calibration images") from exc

    path = path.resolve()
    if not path.is_file():
        raise PreflightError(f"selected manifest is not a file: {path}")
    records = read_jsonl(path)
    if profile["sample_count"] is not None and len(records) != profile["sample_count"]:
        raise PreflightError(
            f"selected manifest has {len(records)} records, expected {profile['sample_count']}"
        )

    root = path.parent.resolve()
    tasks: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    image_paths: set[Path] = set()
    bundle_ids: set[str] = set()
    image_bytes = 0
    decoded_images = 0
    image_formats: Counter[str] = Counter()
    null_output_records = 0
    multi_box_records = 0
    max_box_groups = 0
    max_prompt_chars = 0
    for record in records:
        line = record.pop("_line")
        context = f"{path}:{line}"
        task = str(record.get("task") or "")
        prompt = str(record.get("prompt") or "")
        bundle_id = str(record.get("bundle_id") or "")
        image_value = str(record.get("image") or "")
        if not task or not prompt or not bundle_id or not image_value:
            raise PreflightError(f"{context}: missing task/prompt/bundle_id/image")
        if bundle_id in bundle_ids:
            raise PreflightError(f"{context}: duplicate bundle_id {bundle_id}")
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = root / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise PreflightError(f"{context}: image is not a file: {image_path}")
        try:
            with Image.open(image_path) as source_image:
                source_image.load()
                decoded_width, decoded_height = source_image.size
                image_format = str(source_image.format or "unknown").upper()
        except (OSError, ValueError) as exc:
            raise PreflightError(f"{context}: image cannot be decoded: {image_path}") from exc
        recorded_width = record.get("source_width")
        recorded_height = record.get("source_height")
        if (
            recorded_width is not None
            and recorded_height is not None
            and (int(recorded_width), int(recorded_height))
            != (decoded_width, decoded_height)
        ):
            raise PreflightError(
                f"{context}: recorded image size "
                f"{recorded_width}x{recorded_height} != decoded "
                f"{decoded_width}x{decoded_height}"
            )
        record["_preflight_image_path"] = str(image_path)

        target_response = str(record.get("target_response") or "")
        if target_response:
            validate_target(target_response, context)
        box_groups = target_response.count("<box>")
        null_output_records += int("<box>None</box>" in target_response)
        multi_box_records += int(box_groups > 1)
        max_box_groups = max(max_box_groups, box_groups)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        role = str(metadata.get("calibration_source_role") or "unknown")
        tasks[task] += 1
        roles[role] += 1
        if role == "coco_detection":
            stratum = str(metadata.get("calibration_stratum") or "")
            strata[stratum] += 1
        image_paths.add(image_path)
        bundle_ids.add(bundle_id)
        image_bytes += image_path.stat().st_size
        decoded_images += 1
        image_formats[image_format] += 1
        max_prompt_chars = max(max_prompt_chars, len(prompt))

    for label, actual, expected in (
        ("task counts", dict(tasks), profile["task_counts"]),
        ("source-role counts", dict(roles), profile["source_role_counts"]),
        ("COCO strata", dict(strata), profile["coco_stratum_counts"]),
    ):
        if expected and actual != expected:
            raise PreflightError(f"{label} {actual} != {expected}")
    return ({
        "passed": True,
        "manifest": str(path),
        "sample_count": len(records),
        "task_counts": dict(tasks),
        "source_role_counts": dict(roles),
        "coco_stratum_counts": dict(strata),
        "unique_images": len(image_paths),
        "decoded_images": decoded_images,
        "image_formats": dict(sorted(image_formats.items())),
        "null_output_records": null_output_records,
        "multi_box_records": multi_box_records,
        "max_box_groups": max_box_groups,
        "image_bytes": image_bytes,
        "max_prompt_characters": max_prompt_chars,
    }, records)


def nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise PreflightError("model config is missing " + ".".join(keys))
        value = value[key]
    return value


def audit_model(path: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise PreflightError(f"model path is not a directory: {path}")
    config_path = path / "config.json"
    tokenizer_path = path / "tokenizer.json"
    tokenizer_config = path / "tokenizer_config.json"
    for required in (config_path, tokenizer_config):
        if not required.is_file():
            raise PreflightError(f"required model metadata is missing: {required}")
    if tokenizer_path.is_file():
        tokenizer_layout = "tokenizer_json"
        tokenizer_files = [tokenizer_path]
    else:
        tokenizer_files = [path / "vocab.json", path / "merges.txt", path / "added_tokens.json"]
        missing_tokenizer_files = [item for item in tokenizer_files if not item.is_file()]
        if missing_tokenizer_files:
            raise PreflightError(
                "model path has neither tokenizer.json nor a complete BPE tokenizer: "
                + ", ".join(str(item) for item in missing_tokenizer_files)
            )
        tokenizer_layout = "bpe_vocab_merges"
    if not any((path / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        raise PreflightError("model path has no processor_config.json or preprocessor_config.json")

    model_config = read_json(config_path)
    expected_fields = {
        ("vision_config", "patch_size"): profile["patch_size"],
        ("vision_config", "merge_kernel_size"): [profile["merge_size"], profile["merge_size"]],
        ("text_config", "vocab_size"): profile["vocab_size"],
        ("text_config", "hidden_size"): profile["hidden_size"],
        ("text_config", "block_size"): 6,
        ("text_config", "text_mask_token_id"): 151676,
        ("image_token_index",): profile["image_token_id"],
        ("coord_start_token_id",): 151677,
        ("coord_end_token_id",): 152677,
    }
    for keys, expected in expected_fields.items():
        actual = nested(model_config, *keys)
        if actual != expected:
            raise PreflightError(f"model config {'.'.join(keys)}={actual!r}, expected {expected!r}")

    if tokenizer_layout == "tokenizer_json":
        tokenizer_data = read_json(tokenizer_path)
        added_tokens = tokenizer_data.get("added_tokens")
        if not isinstance(added_tokens, list):
            raise PreflightError("tokenizer.json has no added_tokens list")
        token_ids = {
            str(item.get("content")): int(item.get("id"))
            for item in added_tokens
            if isinstance(item, dict)
            and item.get("content") is not None
            and item.get("id") is not None
        }
    else:
        added_tokens = read_json(path / "added_tokens.json")
        if not isinstance(added_tokens, dict):
            raise PreflightError("added_tokens.json must map token strings to integer IDs")
        token_ids = {
            str(token): int(token_id)
            for token, token_id in added_tokens.items()
            if type(token_id) is int
        }
    bad_tokens = {
        token: {"actual": token_ids.get(token), "expected": expected}
        for token, expected in EXPECTED_TOKEN_IDS.items()
        if token_ids.get(token) != expected
    }
    if bad_tokens:
        raise PreflightError(f"tokenizer special-token mismatch: {bad_tokens}")

    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise PreflightError("model.safetensors.index.json has no weight_map")
        shards = sorted(set(map(str, weight_map.values())))
        missing_shards = [name for name in shards if not (path / name).is_file()]
        if missing_shards:
            raise PreflightError(f"model checkpoint shards are missing: {missing_shards[:3]}")
    else:
        shards = [item.name for item in path.glob("*.safetensors") if item.is_file()]
        if not shards:
            raise PreflightError("model path has no safetensors checkpoint")
    checkpoint_files = {}
    for name in shards:
        shard_path = path / name
        checkpoint_files[name] = {"bytes": shard_path.stat().st_size}
    return {
        "passed": True,
        "path": str(path),
        "config_bytes": config_path.stat().st_size,
        "tokenizer_layout": tokenizer_layout,
        "tokenizer_files": {
            item.name: item.stat().st_size for item in [*tokenizer_files, tokenizer_config]
        },
        "checkpoint_shards": len(shards),
        "checkpoint_files": checkpoint_files,
        "special_token_ids": EXPECTED_TOKEN_IDS,
    }


def audit_upstream(path: Path) -> dict[str, Any]:
    path = path.resolve()
    required = {
        "worker": path / "locateanything_worker.py",
        "processor": path / "eaglevl/utils/locany/processing_locateanything.py",
        "image_processor": path / "eaglevl/utils/locany/image_processing_locateanything.py",
    }
    missing = [str(item) for item in required.values() if not item.is_file()]
    if missing:
        raise PreflightError(f"upstream source files are missing: {missing}")
    signatures = {
        "worker": ("class LocateAnythingWorker", "AutoProcessor.from_pretrained"),
        "processor": ("class LocateAnythingProcessor", "def py_apply_chat_template"),
        "image_processor": ("class LocateAnythingImageProcessor", "patch_size: int = 14"),
    }
    for name, needles in signatures.items():
        text = required[name].read_text(encoding="utf-8")
        if any(needle not in text for needle in needles):
            raise PreflightError(
                f"upstream {name} static API does not match the Prepare adapter"
            )
    return {
        "passed": True,
        "path": str(path),
        "files": {name: item.stat().st_size for name, item in required.items()},
    }


def flatten_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise PreflightError("tokenizer returned more than one input_ids sequence")
        encoded = encoded[0]
    if not isinstance(encoded, list):
        raise PreflightError("tokenizer did not return a flat input_ids list")
    return [int(value) for value in encoded]


def audit_runtime_tokenizer(
    checkpoint_tokenizer: Any,
    records: Iterable[Mapping[str, Any]],
    runtime_tokenizer_path: Path,
    loader: Callable[[str], Any] | None = None,
    processor: Any | None = None,
    visual_tokens: int | None = None,
) -> dict[str, Any]:
    if not runtime_tokenizer_path.is_file():
        raise PreflightError(f"runtime tokenizer.json not found: {runtime_tokenizer_path}")
    if loader is None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise PreflightError("tokenizers cannot load the runtime tokenizer.json") from exc
        loader = Tokenizer.from_file
    try:
        runtime_tokenizer = loader(str(runtime_tokenizer_path.resolve()))
    except Exception as exc:
        raise PreflightError(f"cannot load runtime tokenizer.json: {exc}") from exc

    checked = 0
    expanded_prompts_checked = 0
    for record in records:
        bundle_id = str(record.get("bundle_id"))
        cases = [
            ("prompt", str(record["prompt"]), False),
            ("target_response", str(record.get("target_response") or ""), False),
        ]
        if processor is not None and visual_tokens is not None:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "static-preflight"},
                    {"type": "text", "text": str(record["prompt"])},
                ],
            }]
            rendered = processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            placeholder = "<image-1>"
            if rendered.count(placeholder) != 1:
                raise PreflightError(
                    f"processor chat template mismatch at {bundle_id}:expanded_prompt"
                )
            expansion = (
                f"<image 1>{getattr(processor, 'image_start_token', '<img>')}"
                + str(getattr(processor, "image_token", "")) * visual_tokens
                + str(getattr(processor, "image_end_token", "</img>"))
            )
            cases.append(("expanded_prompt", rendered.replace(placeholder, expansion), False))

        for field, value, add_special_tokens in cases:
            checkpoint_ids = flatten_ids(checkpoint_tokenizer(
                value,
                add_special_tokens=add_special_tokens,
                return_attention_mask=False,
            ))
            try:
                runtime_ids = [
                    int(token_id)
                    for token_id in runtime_tokenizer.encode(
                        value, add_special_tokens=add_special_tokens
                    ).ids
                ]
            except Exception as exc:
                raise PreflightError(
                    f"runtime tokenizer failed for {bundle_id}:{field}: {exc}"
                ) from exc
            if checkpoint_ids != runtime_ids:
                raise PreflightError(
                    f"checkpoint/runtime tokenizer mismatch at {bundle_id}:{field}"
                )
            checked += 1
            if field == "expanded_prompt":
                expanded_prompts_checked += 1
    return {
        "passed": True,
        "path": str(runtime_tokenizer_path.resolve()),
        "bytes": runtime_tokenizer_path.stat().st_size,
        "texts_checked": checked,
        "expanded_prompts_checked": expanded_prompts_checked,
        "regex_contract": "checkpoint_default_matches_runtime_tokenizer_json",
    }


def letterbox_for_profile(image: Any, profile: Mapping[str, Any], image_module: Any) -> Any:
    target_width = int(profile["image_width"])
    target_height = int(profile["image_height"])
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise PreflightError(f"invalid source image size: {image.size}")
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = min(target_width, max(1, int(round(source_width * scale))))
    resized_height = min(target_height, max(1, int(round(source_height * scale))))
    resized = image.resize(
        (resized_width, resized_height), image_module.Resampling.BICUBIC
    )
    fill = int(profile["letterbox_fill"])
    output = image_module.new("RGB", (target_width, target_height), (fill, fill, fill))
    output.paste(
        resized,
        ((target_width - resized_width) // 2, (target_height - resized_height) // 2),
    )
    return output


def audit_representative_images(
    processor: Any,
    records: Iterable[Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise PreflightError("Pillow is required for image preprocessing checks") from exc

    representatives: dict[str, Mapping[str, Any]] = {}
    for record in records:
        task = str(record.get("task") or "")
        image_path = record.get("_preflight_image_path")
        if task and image_path and task not in representatives:
            representatives[task] = record

    expected_tasks = set(profile.get("task_counts", {}))
    if expected_tasks and set(representatives) != expected_tasks:
        missing = sorted(expected_tasks - set(representatives))
        raise PreflightError(f"no representative image available for tasks: {missing}")
    if not representatives:
        return {}

    expected_shape = (
        int(profile["patch_count"]),
        3,
        int(profile["patch_size"]),
        int(profile["patch_size"]),
    )
    grid_side = int(profile["image_width"]) // int(profile["patch_size"])
    expected_grid = [[grid_side, grid_side]]
    checks: dict[str, Any] = {}
    for task, record in sorted(representatives.items()):
        image_path = Path(str(record["_preflight_image_path"]))
        try:
            with Image.open(image_path) as source:
                source_size = list(source.size)
                prepared = letterbox_for_profile(source.convert("RGB"), profile, Image)
        except OSError as exc:
            raise PreflightError(f"cannot decode representative image {image_path}: {exc}") from exc

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": prepared},
                {"type": "text", "text": str(record["prompt"])},
            ],
        }]
        text = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = processor.process_vision_info(messages)
        inputs = processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        )
        pixel_values = inputs.get("pixel_values")
        image_grid_hws = inputs.get("image_grid_hws")
        input_ids = inputs.get("input_ids")
        if pixel_values is None or image_grid_hws is None or input_ids is None:
            raise PreflightError(f"{task}: processor omitted a required model input")
        actual_shape = tuple(int(value) for value in pixel_values.shape)
        if actual_shape != expected_shape:
            raise PreflightError(
                f"{task}: pixel_values shape {actual_shape}, expected {expected_shape}"
            )
        grid = image_grid_hws.tolist() if hasattr(image_grid_hws, "tolist") else image_grid_hws
        if grid != expected_grid:
            raise PreflightError(f"{task}: image_grid_hws={grid}, expected {expected_grid}")
        ids = flatten_ids(input_ids)
        image_tokens = ids.count(int(profile["image_token_id"]))
        if image_tokens != int(profile["visual_tokens"]):
            raise PreflightError(
                f"{task}: processor emitted {image_tokens} visual placeholders, expected 576"
            )
        checks[task] = {
            "bundle_id": str(record.get("bundle_id")),
            "source_size": source_size,
            "prepared_size": [int(profile["image_width"]), int(profile["image_height"])],
            "pixel_values_shape": list(actual_shape),
            "image_grid_hws": grid,
            "visual_tokens": image_tokens,
        }
    return checks


def audit_processor(
    model_path: Path,
    records: Iterable[Mapping[str, Any]],
    profile: Mapping[str, Any],
    loader: Callable[..., Any] | None = None,
    runtime_tokenizer_path: Path | None = None,
    runtime_tokenizer_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    records = list(records)
    if loader is None:
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise PreflightError("transformers cannot load AutoProcessor") from exc
        loader = AutoProcessor.from_pretrained
    try:
        processor = loader(
            str(model_path.resolve()),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
            fix_mistral_regex=False,
        )
    except Exception as exc:
        raise PreflightError(f"cannot load processor metadata without model weights: {exc}") from exc

    image_processor = getattr(processor, "image_processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if image_processor is None or tokenizer is None:
        raise PreflightError("processor does not expose image_processor and tokenizer")
    if int(getattr(image_processor, "patch_size", -1)) != profile["patch_size"]:
        raise PreflightError("processor patch_size does not match the release profile")
    if list(getattr(image_processor, "merge_kernel_size", [])) != [
        profile["merge_size"], profile["merge_size"]
    ]:
        raise PreflightError("processor merge_kernel_size does not match the release profile")
    image_token = str(getattr(processor, "image_token", ""))
    image_token_id = int(getattr(processor, "image_token_id", -1))
    if image_token_id != profile["image_token_id"]:
        raise PreflightError("processor image_token_id does not match config")

    runtime_tokenizer = None
    if runtime_tokenizer_path is not None:
        runtime_tokenizer = audit_runtime_tokenizer(
            tokenizer,
            records,
            runtime_tokenizer_path,
            loader=runtime_tokenizer_loader,
            processor=processor,
            visual_tokens=int(profile["visual_tokens"]),
        )
    representative_images = audit_representative_images(processor, records, profile)

    max_prefill = {"tokens": 0, "bundle_id": None}
    max_target = {"tokens": 0, "bundle_id": None}
    for record in records:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": "static-preflight"},
                {"type": "text", "text": str(record["prompt"])},
            ],
        }]
        text = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        placeholder = "<image-1>"
        if text.count(placeholder) != 1:
            raise PreflightError("processor chat template did not emit one image placeholder")
        expansion = (
            f"<image 1>{getattr(processor, 'image_start_token', '<img>')}"
            + image_token * int(profile["visual_tokens"])
            + str(getattr(processor, "image_end_token", "</img>"))
        )
        expanded = text.replace(placeholder, expansion)
        prefill_ids = flatten_ids(
            tokenizer(expanded, add_special_tokens=False, return_attention_mask=False)
        )
        target_ids = flatten_ids(tokenizer(
            str(record.get("target_response") or ""),
            add_special_tokens=False,
            return_attention_mask=False,
        ))
        bundle_id = str(record.get("bundle_id"))
        if len(prefill_ids) > max_prefill["tokens"]:
            max_prefill = {"tokens": len(prefill_ids), "bundle_id": bundle_id}
        if len(target_ids) > max_target["tokens"]:
            max_target = {"tokens": len(target_ids), "bundle_id": bundle_id}
        if len(prefill_ids) > profile["prefill_limit"]:
            raise PreflightError(
                f"{bundle_id}: processor produced {len(prefill_ids)} prefill tokens, "
                f"limit is {profile['prefill_limit']}"
            )
        if len(target_ids) > profile["max_new_tokens"]:
            raise PreflightError(
                f"{bundle_id}: target has {len(target_ids)} tokens, "
                f"generation limit is {profile['max_new_tokens']}"
            )
        if prefill_ids.count(image_token_id) != profile["visual_tokens"]:
            raise PreflightError(f"{bundle_id}: processor did not emit 576 visual placeholders")
    return {
        "passed": True,
        "processor_class": type(processor).__name__,
        "image_processor_class": type(image_processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "processor_load_contract": {
            "use_fast": False,
            "fix_mistral_regex": False,
        },
        "patch_size": profile["patch_size"],
        "merge_kernel_size": [profile["merge_size"], profile["merge_size"]],
        "visual_tokens": profile["visual_tokens"],
        "max_prefill": max_prefill,
        "max_target": max_target,
        "representative_images": representative_images,
        "runtime_tokenizer": runtime_tokenizer,
        "model_weights_loaded": False,
        "gpu_inference": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = calibration_profile(load_contract(args.config.resolve()))
        dependencies = audit_dependencies()
        manifest, records = audit_manifest(args.selected_jsonl, profile)
        model = audit_model(args.model_path, profile)
        upstream = audit_upstream(args.upstream_repo)
        processor = audit_processor(
            args.model_path,
            records,
            profile,
            runtime_tokenizer_path=RUNTIME_TOKENIZER_JSON,
        )
        report = {
            "schema_version": 1,
            "phase": "prepare_preflight",
            "passed": True,
            "execution": {
                "model_weights_loaded": False,
                "gpu_inference": False,
            },
            "calibration_profile": profile,
            "dependencies": dependencies,
            "manifest": manifest,
            "model": model,
            "upstream": upstream,
            "processor": processor,
        }
        atomic_json(args.report_json.resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except PreflightError as exc:
        print(f"prepare preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
