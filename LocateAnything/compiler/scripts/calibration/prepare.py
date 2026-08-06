#!/usr/bin/env python3
"""Build and materialize a task-specific LocateAnything calibration bundle.

The ``select`` command creates a deterministic, deduplicated six-domain sample
set. The ``generate`` command runs the original LocateAnything processor and
PyTorch model, then stores model predictions and tensors that can be replayed
by the S600 compiler-side calibration pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.scripts.common.progress import track  # noqa: E402


SCHEMA_VERSION = 2
PAPER_TASK_WEIGHTS = {
    "detection": 0.669,
    "gui": 0.165,
    "referring": 0.073,
    "ocr": 0.036,
    "layout": 0.035,
    "pointing": 0.022,
}
TASK_ALIASES = {
    "detection": "detection",
    "object_detection": "detection",
    "object-detection": "detection",
    "gui": "gui",
    "gui_grounding": "gui",
    "gui-grounding": "gui",
    "referring": "referring",
    "phrase_grounding": "referring",
    "phrase-grounding": "referring",
    "ocr": "ocr",
    "text_grounding": "ocr",
    "scene_text_detection": "ocr",
    "layout": "layout",
    "document_layout": "layout",
    "document-layout": "layout",
    "pointing": "pointing",
    "point": "pointing",
}
EVALUATION_SPLITS = {"val", "validation", "test", "dev", "evaluation", "eval"}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            record["_manifest_path"] = str(path.resolve())
            record["_manifest_line"] = line_number
            records.append(record)
    return records


def normalize_task(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "_")
    task = TASK_ALIASES.get(key)
    if task is None:
        supported = ", ".join(PAPER_TASK_WEIGHTS)
        raise ValueError(f"unsupported task {value!r}; expected one of: {supported}")
    return task


def comma_join(value: Any, field: str) -> str:
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, list):
        text = ", ".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = ""
    if not text:
        raise ValueError(f"missing {field}")
    return text


def category_join(value: Any, field: str) -> str:
    if isinstance(value, str):
        parts = value.split("</c>") if "</c>" in value else [value]
    elif isinstance(value, list):
        parts = value
    else:
        parts = []
    categories = [" ".join(str(item).strip().split()) for item in parts]
    categories = [item for item in categories if item]
    if not categories:
        raise ValueError(f"missing {field}")
    if any("</c>" in item for item in categories):
        raise ValueError(f"invalid category separator inside {field}")
    return "</c>".join(categories)


def render_prompt(record: dict[str, Any], task: str) -> str:
    supplied = str(record.get("prompt") or "").strip()
    if supplied:
        return supplied

    if task == "detection":
        categories = category_join(record.get("categories"), "categories")
        return f"Locate all the instances that matches the following description: {categories}."

    if task == "referring":
        phrase = comma_join(record.get("phrase"), "phrase")
        if bool(record.get("multiple", False)):
            return f"Locate all the instances that match the following description: {phrase}."
        return f"Locate a single instance that matches the following description: {phrase}."

    if task == "gui":
        phrase = comma_join(record.get("phrase"), "phrase")
        if str(record.get("output_type", "box")).lower() == "point":
            return f"Point to: {phrase}."
        return f"Locate the region that matches the following description: {phrase}."

    if task == "ocr":
        phrase = str(record.get("phrase") or "").strip()
        if phrase:
            return f"Please locate the text referred as {phrase}."
        return "Detect all the text in box format."

    if task == "layout":
        categories = category_join(record.get("categories"), "categories")
        return (
            "Detect all the objects in the image that belong to the category set: "
            f"{categories}."
        )

    if task == "pointing":
        phrase = comma_join(record.get("phrase"), "phrase")
        return f"Point to: {phrase}."

    raise AssertionError(f"unhandled task: {task}")


def stable_rank(record: dict[str, Any], seed: int) -> tuple[str, ...]:
    return (
        str(seed),
        record["task"],
        str(record.get("source", "")),
        str(record.get("sample_id", "")),
        str(record.get("_source_image", record.get("image", ""))),
    )


def allocate_quotas(
    available: dict[str, int], total: int, temperature: float
) -> dict[str, int]:
    if total <= 0:
        raise ValueError("num_samples must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if sum(available.values()) < total:
        raise ValueError(
            f"requested {total} samples but only {sum(available.values())} are available"
        )

    weights = {
        task: math.pow(PAPER_TASK_WEIGHTS[task], temperature)
        for task in PAPER_TASK_WEIGHTS
    }
    quotas = {task: 0 for task in PAPER_TASK_WEIGHTS}
    for _ in range(total):
        candidates = [task for task in quotas if quotas[task] < available.get(task, 0)]
        if not candidates:
            raise ValueError("no task has enough remaining samples")
        chosen = max(
            candidates,
            key=lambda task: (
                weights[task] / (quotas[task] + 1),
                PAPER_TASK_WEIGHTS[task],
                task,
            ),
        )
        quotas[chosen] += 1
    return quotas


def parse_explicit_quotas(
    values: list[str] | None, available: dict[str, int], total: int
) -> dict[str, int] | None:
    if not values:
        return None
    quotas: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --quota {value!r}; expected task=number")
        task, raw_count = value.split("=", 1)
        task = normalize_task(task)
        if task in quotas:
            raise ValueError(f"duplicate --quota for task: {task}")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"quota must be non-negative: {task}={count}")
        quotas[task] = count
    missing = sorted(set(PAPER_TASK_WEIGHTS) - set(quotas))
    if missing:
        raise ValueError(f"explicit quotas must cover all domains; missing: {missing}")
    if sum(quotas.values()) != total:
        raise ValueError(
            f"explicit quotas sum to {sum(quotas.values())}, expected {total}"
        )
    insufficient = {
        task: {"requested": count, "available": available.get(task, 0)}
        for task, count in quotas.items()
        if count > available.get(task, 0)
    }
    if insufficient:
        raise ValueError(f"explicit quotas exceed available samples: {insufficient}")
    return quotas


def resolve_image(record: dict[str, Any], default_root: Path | None) -> Path:
    image_value = record.get("image")
    if not image_value:
        raise ValueError("missing image")
    image_path = Path(str(image_value)).expanduser()
    if image_path.is_absolute():
        return image_path.resolve()

    record_root = record.get("image_root")
    if record_root:
        root = Path(str(record_root)).expanduser()
        if not root.is_absolute():
            root = Path(record["_manifest_path"]).parent / root
    elif default_root is not None:
        root = default_root
    else:
        root = Path(record["_manifest_path"]).parent
    return (root / image_path).resolve()


def select_records(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    default_root = args.image_root.resolve() if args.image_root else None
    prepared = []
    rejected_splits = Counter()
    missing_license = 0
    for manifest_path in args.input_jsonl:
        manifest_path = manifest_path.resolve()
        for source_record in read_jsonl(manifest_path):
            task = normalize_task(source_record.get("task"))
            split = str(source_record.get("split") or "").strip().lower()
            if not split:
                raise ValueError(
                    f"{source_record['_manifest_path']}:{source_record['_manifest_line']}: "
                    "missing split"
                )
            if split in EVALUATION_SPLITS and not args.allow_evaluation_splits:
                rejected_splits[split] += 1
                continue

            image_path = resolve_image(source_record, default_root)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{source_record['_manifest_path']}:{source_record['_manifest_line']}: "
                    f"image not found: {image_path}"
                )
            source = str(source_record.get("source") or "").strip()
            if not source:
                raise ValueError(
                    f"{source_record['_manifest_path']}:{source_record['_manifest_line']}: "
                    "missing source"
                )

            license_name = str(source_record.get("license") or "").strip()
            if not license_name:
                license_name = "unknown"
                missing_license += 1

            record = {
                key: value
                for key, value in source_record.items()
                if not key.startswith("_") and key not in {"image_root", "prompt"}
            }
            record.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "source": source,
                    "split": split,
                    "license": license_name,
                    "prompt": render_prompt(source_record, task),
                    "_source_image": str(image_path),
                }
            )
            record.setdefault(
                "sample_id",
                f"{source}-{source_record['_manifest_line']}",
            )
            prepared.append(record)

    by_path: dict[str, dict[str, Any]] = {}
    for record in sorted(prepared, key=lambda item: stable_rank(item, args.seed)):
        by_path.setdefault(record["_source_image"], record)
    deduplicated = list(by_path.values())

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduplicated:
        grouped[record["task"]].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: stable_rank(item, args.seed))

    available = {task: len(grouped.get(task, [])) for task in PAPER_TASK_WEIGHTS}
    quotas = parse_explicit_quotas(args.quota, available, args.num_samples)
    if quotas is None:
        quotas = allocate_quotas(available, args.num_samples, args.temperature)
    selected = []
    for task in PAPER_TASK_WEIGHTS:
        selected.extend(grouped.get(task, [])[: quotas[task]])
    selected.sort(key=lambda item: (item["task"], stable_rank(item, args.seed)))

    materialized = []
    for index, record in enumerate(selected):
        source_image = Path(record.pop("_source_image"))
        suffix = source_image.suffix.lower() or ".img"
        destination = image_dir / f"{index:04d}_{record['task']}{suffix}"
        if not destination.exists() or destination.stat().st_size == 0:
            shutil.copy2(source_image, destination)
        record["bundle_id"] = f"{index:04d}-{record['task']}"
        record["image"] = destination.relative_to(output_dir).as_posix()
        materialized.append(record)

    selected_manifest = output_dir / "selected.jsonl"
    write_jsonl(selected_manifest, materialized)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "selection_seed": args.seed,
        "num_samples": len(materialized),
        "task_temperature": args.temperature,
        "quota_mode": "explicit" if args.quota else "temperature",
        "paper_query_weights": PAPER_TASK_WEIGHTS,
        "available_counts": available,
        "selected_counts": dict(Counter(record["task"] for record in materialized)),
        "rejected_evaluation_splits": dict(rejected_splits),
        "deduplicated_images": len(prepared) - len(deduplicated),
        "records_with_unknown_license": missing_license,
        "selected_manifest": selected_manifest.name,
    }
    write_json(output_dir / "selection_summary.json", summary)

    print(f"[select] wrote {len(materialized)} samples -> {selected_manifest}")
    print(f"[select] counts: {summary['selected_counts']}")
    if rejected_splits:
        print(f"[select] rejected evaluation splits: {dict(rejected_splits)}")
    if missing_license:
        print(f"[select] warning: {missing_license} source rows have license=unknown")
    return 0


def torch_dtype_from_name(torch_module: Any, name: str) -> Any:
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def deterministic_seed(base_seed: int, bundle_id: str, mode: str) -> int:
    prefix = bundle_id.split("-", 1)[0]
    sequence = int(prefix) if prefix.isdigit() else 0
    mode_offset = {"slow-select": 0, "hybrid": 1, "slow": 2}.get(mode, 3)
    return (int(base_seed) + sequence * 4 + mode_offset) % (2**32)


def extract_answer(result: Any) -> str:
    if isinstance(result, dict):
        answer = result.get("answer")
    else:
        answer = result
    if isinstance(answer, (tuple, list)):
        answer = answer[0] if answer else ""
    if not isinstance(answer, str):
        raise TypeError(f"LocateAnything prediction returned {type(answer).__name__}, expected str")
    return answer


def response_token_ids(tokenizer: Any, response: str) -> list[int]:
    encoded = tokenizer(response, add_special_tokens=False, return_attention_mask=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def build_fixed_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if args.patch_size <= 0 or args.merge_size <= 0:
        raise ValueError("patch_size and merge_size must be positive")
    if not 0 <= args.letterbox_fill <= 255:
        raise ValueError("letterbox_fill must be within [0, 255]")
    release_contract = {
        "image_width": 672,
        "image_height": 672,
        "resize_mode": "letterbox",
        "letterbox_fill": 128,
        "patch_size": 14,
        "merge_size": 2,
        "hidden_size": 2048,
        "prefill_limit": 1024,
    }
    drift = {
        name: getattr(args, name)
        for name, expected in release_contract.items()
        if getattr(args, name) != expected
    }
    if drift:
        raise ValueError(f"Prepare arguments drift from the release contract: {drift}")

    profile_multiple = args.patch_size * args.merge_size
    if args.image_width % profile_multiple or args.image_height % profile_multiple:
        raise ValueError(
            f"image dimensions must be divisible by patch_size * merge_size "
            f"({profile_multiple})"
        )

    grid_width = args.image_width // args.patch_size
    grid_height = args.image_height // args.patch_size
    patch_count = grid_width * grid_height
    merge_area = args.merge_size * args.merge_size
    if patch_count % merge_area:
        raise ValueError("patch count must be divisible by the merge area")
    visual_token_count = patch_count // merge_area
    if args.prefill_limit <= visual_token_count:
        raise ValueError(
            f"prefill_limit={args.prefill_limit} leaves no text capacity after "
            f"{visual_token_count} visual tokens"
        )

    patch_flat_dim = 3 * args.patch_size * args.patch_size
    return {
        "image_width": args.image_width,
        "image_height": args.image_height,
        "resize_mode": args.resize_mode,
        "letterbox_fill": args.letterbox_fill,
        "patch_size": args.patch_size,
        "merge_size": args.merge_size,
        "grid_hw": [grid_height, grid_width],
        "patch_count": patch_count,
        "vision_input_shape": [1, patch_count, patch_flat_dim],
        "visual_token_count": visual_token_count,
        "projected_visual_shape": [1, visual_token_count, args.hidden_size],
        "prefill_limit": args.prefill_limit,
        "remaining_prefill_tokens": args.prefill_limit - visual_token_count,
        "pbd_block_size": 6,
    }


def prepare_profile_image(
    source_image: Any,
    profile: dict[str, Any],
    image_module: Any,
) -> tuple[Any, dict[str, Any]]:
    source_width, source_height = source_image.size
    target_width = int(profile["image_width"])
    target_height = int(profile["image_height"])
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"invalid source image size: {source_image.size}")

    if profile["resize_mode"] == "stretch":
        resized_width, resized_height = target_width, target_height
        left = top = right = bottom = 0
        output_image = source_image.resize(
            (resized_width, resized_height), image_module.Resampling.BICUBIC
        )
    else:
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = min(target_width, max(1, int(round(source_width * scale))))
        resized_height = min(target_height, max(1, int(round(source_height * scale))))
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        right = target_width - resized_width - left
        bottom = target_height - resized_height - top
        resized = source_image.resize(
            (resized_width, resized_height), image_module.Resampling.BICUBIC
        )
        fill = int(profile["letterbox_fill"])
        output_image = image_module.new(
            "RGB", (target_width, target_height), (fill, fill, fill)
        )
        output_image.paste(resized, (left, top))

    transform = {
        "mode": profile["resize_mode"],
        "source_size": [source_width, source_height],
        "target_size": [target_width, target_height],
        "resized_size": [resized_width, resized_height],
        "scale_xy": [resized_width / source_width, resized_height / source_height],
        "padding_ltrb": [left, top, right, bottom],
        "letterbox_fill": int(profile["letterbox_fill"]),
    }
    return output_image, transform


COORDINATE_BLOCK_PATTERN = re.compile(r"<box>((?:<\d{1,4}>)+)</box>")
COORDINATE_TOKEN_PATTERN = re.compile(r"<(\d{1,4})>")


def transform_target_response(response: str, transform: dict[str, Any]) -> str:
    if not response:
        return response

    source_width, source_height = transform["source_size"]
    target_width, target_height = transform["target_size"]
    scale_x, scale_y = transform["scale_xy"]
    left, top, _, _ = transform["padding_ltrb"]

    def transform_coordinate(value: int, axis: int) -> int:
        if not 0 <= value <= 1000:
            raise ValueError(f"coordinate token outside [0, 1000]: <{value}>")
        if axis == 0:
            pixel = value / 1000.0 * source_width
            transformed = (pixel * scale_x + left) / target_width * 1000.0
        else:
            pixel = value / 1000.0 * source_height
            transformed = (pixel * scale_y + top) / target_height * 1000.0
        return max(0, min(1000, int(round(transformed))))

    def replace_block(match: re.Match[str]) -> str:
        coordinates = [
            int(value) for value in COORDINATE_TOKEN_PATTERN.findall(match.group(1))
        ]
        if len(coordinates) not in {2, 4}:
            raise ValueError(
                f"expected point or box coordinates, got {len(coordinates)} "
                f"values in {match.group(0)!r}"
            )
        transformed = [
            transform_coordinate(value, index % 2)
            for index, value in enumerate(coordinates)
        ]
        return "<box>" + "".join(f"<{value}>" for value in transformed) + "</box>"

    return COORDINATE_BLOCK_PATTERN.sub(replace_block, response)


def prepare_native_inputs(
    worker: Any,
    image: Any,
    prompt: str,
    torch_module: Any,
    profile: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = worker.processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = worker.processor.process_vision_info(messages)
    inputs = worker.processor(
        text=[text], images=images, videos=videos, return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"]
    if pixel_values.ndim == 4:
        pixel_values_flat = pixel_values.reshape(pixel_values.shape[0], -1)
    elif pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
        pixel_values_flat = pixel_values.squeeze(0)
    elif pixel_values.ndim == 2:
        pixel_values_flat = pixel_values
    else:
        raise ValueError(f"unexpected pixel_values shape: {tuple(pixel_values.shape)}")

    expected_vision_shape = tuple(profile["vision_input_shape"][1:])
    if tuple(pixel_values_flat.shape) != expected_vision_shape:
        raise ValueError(
            f"compiled profile requires pixel_values={expected_vision_shape}, "
            f"got {tuple(pixel_values_flat.shape)}"
        )

    image_grid_hws = inputs.get("image_grid_hws")
    if image_grid_hws is None:
        raise ValueError("processor did not return image_grid_hws")
    if not torch_module.is_tensor(image_grid_hws):
        image_grid_hws = torch_module.as_tensor(image_grid_hws, dtype=torch_module.int32)
    actual_grid = image_grid_hws.detach().cpu().to(torch_module.int32).tolist()
    expected_grid = [profile["grid_hw"]]
    if actual_grid != expected_grid:
        raise ValueError(f"image_grid_hws={actual_grid}, expected {expected_grid}")

    device = torch_module.device(worker.device)
    model_dtype = worker.dtype
    with torch_module.no_grad():
        model_pixel_values = pixel_values.to(device=device, dtype=model_dtype)
        model_grid = image_grid_hws.to(device=device, dtype=torch_module.int32)
        vision_features = worker.model.extract_feature(model_pixel_values, model_grid)
        if isinstance(vision_features, (tuple, list)):
            vision_features = torch_module.cat(list(vision_features), dim=0)
        projected = worker.model.mlp1(vision_features)
    projected = projected.reshape(1, -1, projected.shape[-1])
    expected_projected_shape = tuple(profile["projected_visual_shape"])
    if tuple(projected.shape) != expected_projected_shape:
        raise ValueError(
            f"compiled language profile requires projected features "
            f"{expected_projected_shape}, got {tuple(projected.shape)}"
        )

    prompt_ids = inputs["input_ids"]
    if prompt_ids.shape[1] > profile["prefill_limit"]:
        raise ValueError(
            f"prompt has {prompt_ids.shape[1]} tokens; compiled prefill limit is "
            f"{profile['prefill_limit']}"
        )

    return {
        "prompt_text": text,
        "prompt_input_ids": prompt_ids.detach().cpu().to(torch_module.int64),
        "prompt_attention_mask": inputs["attention_mask"].detach().cpu().to(torch_module.int64),
        "vision_input": pixel_values_flat.unsqueeze(0).detach().cpu().to(torch_module.float16),
        "image_grid_hws": image_grid_hws.detach().cpu().to(torch_module.int32),
        "projected_visual_features": projected.detach().cpu().to(torch_module.float16),
    }


def load_completed_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        record["bundle_id"]: {
            key: value for key, value in record.items() if not key.startswith("_")
        }
        for record in read_jsonl(path)
        if record.get("status") == "complete"
    }


def append_progress(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_tensor_artifact(
    path: Path,
    output_format: str,
    payload: dict[str, Any],
    torch_module: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    if output_format == "pt":
        torch_module.save(payload, temporary)
    elif output_format == "npy":
        vision_input = payload["vision_input"]
        if vision_input.dtype != torch_module.float16:
            raise TypeError(
                f"vision_input dtype is {vision_input.dtype}, expected torch.float16"
            )
        value = vision_input.detach().cpu().numpy()
        if tuple(value.shape) != (1, 2304, 588):
            raise ValueError(f"vision_input shape is {value.shape}, expected (1, 2304, 588)")
        if not np.isfinite(value).all():
            raise ValueError("vision_input contains NaN or Inf")
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
    else:
        raise ValueError(f"unsupported output format: {output_format}")
    os.replace(temporary, path)


def generate_bundle(args: argparse.Namespace) -> int:
    if args.dtype != "bfloat16":
        raise RuntimeError("release Prepare requires bfloat16 tensors")

    import torch
    from PIL import Image

    model_path = args.model_path.resolve()
    source_dir = (
        args.source_dir.resolve() if args.source_dir else model_path.parent
    )
    worker_path = source_dir / "locateanything_worker.py"
    if not worker_path.is_file():
        raise FileNotFoundError(
            f"locateanything_worker.py not found in {source_dir}; "
            "set --source-dir to the official LocateAnything implementation directory"
        )
    sys.path.insert(0, str(source_dir))
    try:
        from locateanything_worker import LocateAnythingWorker
    except ImportError as exc:
        raise ImportError(
            f"could not import locateanything_worker.py from {source_dir}"
        ) from exc

    selected_path = args.selected_jsonl.resolve()
    bundle_root = selected_path.parent
    output_dir = args.output_dir.resolve()
    tensor_dir = output_dir / "tensors"
    progress_path = output_dir / "generation_progress.jsonl"

    records = read_jsonl(selected_path)
    if not records:
        raise ValueError(f"selected manifest is empty: {selected_path}")
    profile = build_fixed_profile(args)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": None if args.top_k <= 0 else args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": True,
        "use_cache": True,
    }

    slow_order = sorted(
        records,
        key=lambda record: deterministic_seed(args.seed, record["bundle_id"], "slow-select"),
    )
    slow_ids = {record["bundle_id"] for record in slow_order[: args.slow_samples]}
    if not args.resume and progress_path.exists():
        raise RuntimeError(
            "prepare output already contains run state; use --resume or a separate output directory"
        )
    tensor_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_progress(progress_path) if args.resume else {}

    dtype = torch_dtype_from_name(torch, args.dtype)
    worker = LocateAnythingWorker(
        model_path=str(model_path),
        device=args.device,
        dtype=dtype,
        use_batch_runtime=False,
    )

    generated_records: dict[str, dict[str, Any]] = dict(completed)
    special_tokens = [
        "<ref>",
        "</ref>",
        "<box>",
        "</box>",
        "<null>",
        "<text_mask>",
        "<|im_end|>",
    ]
    special_token_ids = {
        token: int(worker.tokenizer.convert_tokens_to_ids(token)) for token in special_tokens
    }

    progress = track(records, "Generate calibration", unit="sample")
    for index, record in enumerate(progress, start=1):
        bundle_id = record["bundle_id"]
        existing = generated_records.get(bundle_id)
        if existing:
            if existing.get("fixed_profile") != profile:
                raise RuntimeError(
                    f"resume profile mismatch for {bundle_id}; "
                    "use a separate output directory"
                )
            existing_format = existing.get("tensor_format") or Path(
                existing["tensor_file"]
            ).suffix.lstrip(".")
            if existing_format != args.output_format:
                raise RuntimeError(
                    f"resume format mismatch for {bundle_id}: {existing_format} != "
                    f"{args.output_format}; use a separate output directory"
                )
            tensor_path = output_dir / existing["tensor_file"]
            if tensor_path.is_file() and tensor_path.stat().st_size > 0:
                progress.set_postfix(status="resume", sample=bundle_id, refresh=False)
                continue

        image_path = (bundle_root / record["image"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"selected image missing: {image_path}")
        with Image.open(image_path) as source_image:
            image, spatial_transform = prepare_profile_image(
                source_image.convert("RGB"), profile, Image
            )

        native_inputs = prepare_native_inputs(
            worker, image, record["prompt"], torch, profile
        )
        predictions = {}
        modes = ["hybrid"]
        if bundle_id in slow_ids:
            modes.append("slow")
        for mode in modes:
            seed = deterministic_seed(args.seed, bundle_id, mode)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            result = worker.predict(
                image=image,
                question=record["prompt"],
                generation_mode=mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                verbose=False,
            )
            answer = extract_answer(result)
            token_ids = response_token_ids(worker.tokenizer, answer)
            predictions[mode] = {
                "answer": answer,
                "token_ids": token_ids,
                "seed": seed,
            }

        target_response = str(record.get("target_response") or "")
        profile_target_response = transform_target_response(
            target_response, spatial_transform
        )
        target_token_ids = (
            response_token_ids(worker.tokenizer, profile_target_response)
            if profile_target_response
            else []
        )

        tensor_payload = {
            "schema_version": SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "task": record["task"],
            "source": record["source"],
            "prompt_text": native_inputs["prompt_text"],
            "prompt_input_ids": native_inputs["prompt_input_ids"],
            "prompt_attention_mask": native_inputs["prompt_attention_mask"],
            "vision_input": native_inputs["vision_input"],
            "image_grid_hws": native_inputs["image_grid_hws"],
            "projected_visual_features": native_inputs["projected_visual_features"],
            "fixed_profile": profile,
            "spatial_transform": spatial_transform,
            "source_target_response": target_response,
            "profile_target_response": profile_target_response,
            "prediction_token_ids": {
                mode: torch.tensor(value["token_ids"], dtype=torch.int64)
                for mode, value in predictions.items()
            },
            "prediction_seeds": {
                mode: int(value["seed"]) for mode, value in predictions.items()
            },
            "generation_config": generation_config,
            "target_token_ids": torch.tensor(target_token_ids, dtype=torch.int64),
            "special_token_ids": special_token_ids,
        }
        tensor_path = tensor_dir / f"{bundle_id}.{args.output_format}"
        save_tensor_artifact(tensor_path, args.output_format, tensor_payload, torch)

        generated = {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        generated.update(
            {
                "status": "complete",
                "prediction": predictions,
                "fixed_profile": profile,
                "spatial_transform": spatial_transform,
                "profile_target_response": profile_target_response,
                "tensor_format": args.output_format,
                "tensor_file": tensor_path.relative_to(output_dir).as_posix(),
            }
        )
        append_progress(progress_path, generated)
        generated_records[bundle_id] = generated
        progress.set_postfix(
            sample=bundle_id,
            modes=",".join(modes),
            refresh=False,
        )

    ordered = [generated_records[record["bundle_id"]] for record in records]
    generated_manifest = output_dir / "generated.jsonl"
    write_jsonl(generated_manifest, ordered)

    token_coverage = Counter()
    for record in ordered:
        for prediction in record["prediction"].values():
            token_coverage.update(prediction["token_ids"])
        target_response = str(record.get("target_response") or "")
        if target_response:
            token_coverage.update(response_token_ids(worker.tokenizer, target_response))
    coverage_by_token = {
        token: token_coverage.get(token_id, 0)
        for token, token_id in special_token_ids.items()
    }
    task_counts = dict(Counter(record["task"] for record in ordered))
    mode_counts = dict(
        Counter(mode for record in ordered for mode in record["prediction"])
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_path": str(model_path),
        "locateanything_source": str(source_dir),
        "selected_manifest": str(selected_path),
        "generated_manifest": generated_manifest.name,
        "sample_count": len(ordered),
        "tensor_format": args.output_format,
        "task_counts": task_counts,
        "generation_mode_counts": mode_counts,
        "generation_config": generation_config,
        "base_seed": args.seed,
        "special_token_ids": special_token_ids,
        "special_token_occurrences": coverage_by_token,
        "fixed_profile": profile,
    }
    write_json(output_dir / "generation_summary.json", summary)

    if coverage_by_token["<box>"] == 0 or coverage_by_token["</box>"] == 0:
        raise RuntimeError("generated and target responses contain no complete <box> blocks")

    print(f"[generate] wrote {len(ordered)} samples -> {generated_manifest}")
    print(f"[generate] task counts: {task_counts}")
    print(f"[generate] special token occurrences: {coverage_by_token}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select", help="select and materialize a deterministic six-domain data mix"
    )
    select_parser.add_argument(
        "--input-jsonl", type=Path, action="append", required=True,
        help="canonical source JSONL; repeat for multiple datasets",
    )
    select_parser.add_argument("--image-root", type=Path)
    select_parser.add_argument("--output-dir", type=Path, required=True)
    select_parser.add_argument("--num-samples", type=int, default=1200)
    select_parser.add_argument(
        "--quota",
        action="append",
        help="explicit task quota, repeat for all six domains (for example detection=620)",
    )
    select_parser.add_argument("--seed", type=int, default=20260729)
    select_parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="paper-weight exponent; 1.0 mirrors query ratios, 0.5 improves minority coverage",
    )
    select_parser.add_argument("--allow-evaluation-splits", action="store_true")
    select_parser.set_defaults(func=select_records)

    generate_parser = subparsers.add_parser(
        "generate", help="run the original LA PyTorch model and save calibration tensors"
    )
    generate_parser.add_argument("--selected-jsonl", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument(
        "--source-dir", type=Path,
        help="directory containing locateanything_worker.py; defaults to --model-path parent",
    )
    generate_parser.add_argument("--model-path", type=Path, required=True)
    generate_parser.add_argument(
        "--output-format", choices=["pt", "npy"], default="pt",
        help="pt saves the full calibration bundle; npy saves portable vision_input only",
    )
    generate_parser.add_argument("--device", default="cuda:0")
    generate_parser.add_argument(
        "--dtype", choices=["bfloat16"], default="bfloat16"
    )
    generate_parser.add_argument("--image-width", type=int, default=672)
    generate_parser.add_argument("--image-height", type=int, default=672)
    generate_parser.add_argument(
        "--resize-mode", choices=["letterbox", "stretch"], default="letterbox"
    )
    generate_parser.add_argument(
        "--letterbox-fill", type=int, default=128,
        help="RGB fill value for letterbox padding; 128 is near zero after LA normalization",
    )
    generate_parser.add_argument("--patch-size", type=int, default=14)
    generate_parser.add_argument("--merge-size", type=int, default=2)
    generate_parser.add_argument("--hidden-size", type=int, default=2048)
    generate_parser.add_argument("--prefill-limit", type=int, default=1024)
    generate_parser.add_argument("--max-new-tokens", type=int, default=1024)
    generate_parser.add_argument("--slow-samples", type=int, default=64)
    generate_parser.add_argument("--seed", type=int, default=20260729)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-p", type=float, default=0.9)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.1)
    generate_parser.add_argument("--resume", action="store_true")
    generate_parser.set_defaults(func=generate_bundle)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
