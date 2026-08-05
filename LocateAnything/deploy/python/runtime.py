#!/usr/bin/env python3
"""Run LocateAnything end to end on an S600 from one image and prompt."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from console import BOLD, CYAN, GREEN, RESET
from telemetry import (
    ResourceMonitor,
    parse_s600_resource_status,
    resource_summary_lines,
)


IMAGE_SIZE = 672
PATCH_SIZE = 14
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
IMAGE_TOKENS = (GRID_SIZE * GRID_SIZE) // 4
IMAGE_TOKEN = "<IMG_CONTEXT>"

TASK_COMMANDS = (
    "/detect <category>[,<category>...]  Object Detection",
    "/ground <phrase>                 Referring Comprehension",
    "/gui <element>                   GUI Grounding (point)",
    "/text                            Text OCR",
    "/layout <category>[,<category>...] Layout Grounding",
    "/point <target>                  Point Localization",
    "/ground_single <phrase>          Single-instance grounding",
    "/gui_box <element>               GUI Grounding (box)",
    "/ground_text <text>              Text grounding",
)

RUNTIME_VERSION = "0.6.0"
DEFAULT_NMS_IOU = 0.90


@dataclass(frozen=True)
class RuntimeConfig:
    source: Path
    layout_root: Path
    model_type: str
    vision_model: Path
    language_model: Path
    embeddings: Path
    tokenizer_dir: Path
    vision_runner: Path
    language_runner: Path
    image_width: int
    image_height: int
    patch_size: int
    visual_tokens: int
    vocab_size: int
    embed_dim: int
    prefill_chunk: int
    cache_len: int
    pbd_query_len: int
    ar_query_len: int
    language_graph_set: str
    default_generation_mode: str
    default_max_new_tokens: int
    default_nms_iou: float
    l2m_sizes: str
    telemetry_interval_seconds: float
    runner_startup_timeout_seconds: float
    bpu_cores: tuple[int, ...]

    def specification(self) -> dict[str, object]:
        return {
            "image_size": [self.image_width, self.image_height],
            "patch_size": self.patch_size,
            "visual_tokens": self.visual_tokens,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "prefill_chunk": self.prefill_chunk,
            "cache_len": self.cache_len,
            "pbd_query_len": self.pbd_query_len,
            "ar_query_len": self.ar_query_len,
            "language_graph_set": self.language_graph_set,
            "bpu_cores": list(self.bpu_cores),
            "l2m_sizes": self.l2m_sizes,
            "runner_startup_timeout_seconds": self.runner_startup_timeout_seconds,
        }


@dataclass(frozen=True)
class PredictionPaths:
    root: Path
    prediction: Path
    annotated_image: Path
    timings: Path
    runtime_log: Path


def _safe_name(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or fallback


def create_prediction_paths(
    layout_root: Path,
    image_path: Path,
    output_dir: Path | None = None,
    *,
    now: datetime | None = None,
    unique_id: str | None = None,
) -> PredictionPaths:
    """Create one self-contained output directory for a prediction request."""

    if output_dir is None:
        timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        image_name = _safe_name(image_path.stem, "image")
        suffix = _safe_name(unique_id or uuid.uuid4().hex[:8], "run")
        root = layout_root / "artifacts" / "runs" / "predict"
        output_dir = root / f"{timestamp}_{image_name}_{suffix}"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return PredictionPaths(
        root=output_dir,
        prediction=output_dir / "prediction.json",
        annotated_image=output_dir / "annotated.png",
        timings=output_dir / "timings.json",
        runtime_log=log_dir / "runtime.log",
    )


def print_stage(index: int, name: str, elapsed_seconds: float | None = None) -> None:
    if elapsed_seconds is None:
        print(f"[{index}/3] {name}...", flush=True)
    else:
        print(f"[{index}/3] {name}: {elapsed_seconds * 1000.0:.1f} ms", flush=True)


def _configured_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _environment_path(name: str, default: Path, layout_root: Path) -> Path:
    value = os.environ.get(name)
    return _configured_path(layout_root, value) if value else default.resolve()


def _deployment_root(runtime_dir: Path) -> Path:
    if runtime_dir.name == "python" and runtime_dir.parent.name == "deploy":
        return runtime_dir.parent
    return runtime_dir


def _runtime_layout_root(runtime_dir: Path, source: Path) -> Path:
    if source.parent.name == "config":
        candidate = source.parent.parent
        if candidate.name == "deploy":
            return candidate.parent.resolve()
        if (candidate / "deploy").is_dir():
            return candidate.resolve()
    return _deployment_root(runtime_dir).parent.resolve()


def _reject_non_finite_json(value: str):
    raise ValueError(f"non-finite JSON value: {value}")


def discover_runtime_config(
    runtime_dir: Path,
    explicit: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    environment = os.environ.get("LA_RUNTIME_CONFIG")
    if environment:
        path = Path(environment).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    deploy_root = _deployment_root(runtime_dir)
    candidates = [
        deploy_root.parent / "config" / "locateanything_3b_config.json",
        deploy_root / "config" / "runtime.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"runtime config not found; searched: {searched}")


def load_runtime_config(
    config_path: Path | None = None,
    runtime_dir: Path | None = None,
) -> RuntimeConfig:
    runtime_dir = (runtime_dir or Path(__file__).resolve().parent).resolve()
    source = discover_runtime_config(runtime_dir, config_path)
    layout_root = _runtime_layout_root(runtime_dir, source)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_json,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid runtime config {source}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a JSON object: {source}")

    required_keys = {
        "model_type", "model_dir", "vit_model_file", "llm_model_file",
        "embed_weight_file_path", "vocabulary_path", "image_width",
        "image_height", "patch_size", "visual_tokens", "vocab_size",
        "embed_dim", "prefill_chunk", "cache_len", "pbd_query_len",
        "ar_query_len", "language_graph_set", "default_generation_mode", "default_max_new_tokens",
        "default_nms_iou", "l2m_sizes", "telemetry_interval_ms",
        "runner_startup_timeout_seconds",
        "vit_bpu_core", "prefill_bpu_core", "decode_bpu_core",
    }
    missing_keys = sorted(required_keys.difference(raw))
    if missing_keys:
        raise ValueError(
            f"runtime config is missing required fields: {', '.join(missing_keys)}"
        )

    fixed_specification = {
        "model_type": "LocateAnything-3B",
        "image_width": IMAGE_SIZE,
        "image_height": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "visual_tokens": IMAGE_TOKENS,
        "vocab_size": 152681,
        "embed_dim": 2048,
        "prefill_chunk": 1024,
        "cache_len": 4096,
        "pbd_query_len": 6,
        "ar_query_len": 1,
        "default_generation_mode": "hybrid",
        "default_max_new_tokens": 4096,
        "l2m_sizes": "6:6:6:6",
    }
    for key, expected in fixed_specification.items():
        value = raw[key]
        if value != expected:
            raise ValueError(f"runtime config {key}={value!r}; expected {expected!r}")

    generation_mode = str(raw["default_generation_mode"])
    language_graph_set = str(raw["language_graph_set"])
    max_new_tokens = int(raw["default_max_new_tokens"])
    nms_iou = float(raw["default_nms_iou"])
    telemetry_ms = int(raw["telemetry_interval_ms"])
    startup_timeout = float(raw["runner_startup_timeout_seconds"])
    if max_new_tokens <= 0:
        raise ValueError("default_max_new_tokens must be positive")
    if language_graph_set not in {"standard", "fused_decode"}:
        raise ValueError("language_graph_set must be 'standard' or 'fused_decode'")
    if not 0.0 <= nms_iou <= 1.0:
        raise ValueError("default_nms_iou must be between 0 and 1")
    if telemetry_ms < 250:
        raise ValueError("telemetry_interval_ms must be at least 250")
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("runner_startup_timeout_seconds must be finite and positive")

    model_dir = _configured_path(layout_root, str(raw["model_dir"]))
    release_root = os.environ.get("LA_RELEASE_ROOT")
    if release_root:
        model_dir = _configured_path(layout_root, release_root)
    vision_model = _environment_path(
        "LA_VISION_MODEL", model_dir / str(raw["vit_model_file"]), layout_root
    )
    language_model = _environment_path(
        "LA_LANGUAGE_MODEL", model_dir / str(raw["llm_model_file"]), layout_root
    )
    embeddings = _environment_path(
        "LA_EMBEDDINGS", model_dir / str(raw["embed_weight_file_path"]), layout_root
    )
    tokenizer_dir = _environment_path(
        "LA_TOKENIZER_DIR",
        _configured_path(layout_root, str(raw["vocabulary_path"])),
        layout_root,
    )
    release_deploy_dir = layout_root / "deploy"
    vision_runner = _environment_path(
        "LA_VISION_RUNNER", release_deploy_dir / "build" / "vision_hbm_runner", layout_root
    )
    language_runner = _environment_path(
        "LA_LANGUAGE_RUNNER", release_deploy_dir / "build" / "language_hbm_runner", layout_root
    )
    expected_cores = (0, 1, 2, 3)
    for field in ("vit_bpu_core", "prefill_bpu_core", "decode_bpu_core"):
        cores = tuple(int(value) for value in raw[field])
        if cores != expected_cores:
            raise ValueError(f"runtime requires {field} [0,1,2,3], got {cores}")
    bpu_cores = expected_cores

    return RuntimeConfig(
        source=source,
        layout_root=layout_root,
        model_type=str(raw["model_type"]),
        vision_model=vision_model,
        language_model=language_model,
        embeddings=embeddings,
        tokenizer_dir=tokenizer_dir,
        vision_runner=vision_runner,
        language_runner=language_runner,
        image_width=IMAGE_SIZE,
        image_height=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        visual_tokens=IMAGE_TOKENS,
        vocab_size=152681,
        embed_dim=2048,
        prefill_chunk=1024,
        cache_len=4096,
        pbd_query_len=6,
        ar_query_len=1,
        language_graph_set=language_graph_set,
        default_generation_mode=generation_mode,
        default_max_new_tokens=max_new_tokens,
        default_nms_iou=nms_iou,
        l2m_sizes=str(raw["l2m_sizes"]),
        telemetry_interval_seconds=telemetry_ms / 1000.0,
        runner_startup_timeout_seconds=startup_timeout,
        bpu_cores=bpu_cores,
    )


def build_runtime_environment(runtime: RuntimeConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["HB_DNN_USER_DEFINED_L2M_SIZES"] = runtime.l2m_sizes
    return env


def language_runner_command(runtime: RuntimeConfig) -> list[str]:
    return [
        str(runtime.language_runner),
        "--model",
        str(runtime.language_model),
        "--embed",
        str(runtime.embeddings),
        "--graph-set",
        runtime.language_graph_set,
    ]


def load_runtime_config_from_args(
    argv: list[str] | None = None,
    runtime_dir: Path | None = None,
) -> RuntimeConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-c", "--config", type=Path)
    known, _ = pre_parser.parse_known_args(argv)
    return load_runtime_config(known.config, runtime_dir)


def require_runtime_paths(runtime: RuntimeConfig) -> None:
    required = {
        "Vision runner": (runtime.vision_runner, False),
        "Language runner": (runtime.language_runner, False),
        "Tokenizer directory": (runtime.tokenizer_dir, True),
        "Tokenizer JSON": (runtime.tokenizer_dir / "tokenizer.json", False),
        "Vision HBM": (runtime.vision_model, False),
        "Language HBM": (runtime.language_model, False),
        "Embedding table": (runtime.embeddings, False),
    }
    missing = [
        f"{label}: {path}"
        for label, (path, directory) in required.items()
        if not (path.is_dir() if directory else path.is_file())
    ]
    if missing:
        raise FileNotFoundError("runtime payload is incomplete: " + "; ".join(missing))
    if os.name == "posix":
        not_executable = [
            str(path)
            for path in (runtime.vision_runner, runtime.language_runner)
            if not os.access(path, os.X_OK)
        ]
        if not_executable:
            raise PermissionError(
                "runtime runner is not executable: " + "; ".join(not_executable)
            )


def prepare_image(image_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with Image.open(image_path) as handle:
        source = handle.convert("RGB")
    source_width, source_height = source.size
    scale = min(IMAGE_SIZE / source_width, IMAGE_SIZE / source_height)
    resized_width = min(IMAGE_SIZE, max(1, int(round(source_width * scale))))
    resized_height = min(IMAGE_SIZE, max(1, int(round(source_height * scale))))
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    right = IMAGE_SIZE - resized_width - left
    bottom = IMAGE_SIZE - resized_height - top

    resized = source.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    profile = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
    profile.paste(resized, (left, top))

    values = np.asarray(profile, dtype=np.float32) / 255.0
    values = (values - 0.5) / 0.5
    chw = values.transpose(2, 0, 1)
    patches = chw.reshape(3, GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE)
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(1, GRID_SIZE * GRID_SIZE, -1)
    vision_input = np.ascontiguousarray(patches, dtype=np.float16)

    transform = {
        "source_size": [source_width, source_height],
        "target_size": [IMAGE_SIZE, IMAGE_SIZE],
        "resized_size": [resized_width, resized_height],
        "scale_xy": [resized_width / source_width, resized_height / source_height],
        "padding_ltrb": [left, top, right, bottom],
    }
    return vision_input, transform


def _require_task_argument(raw: str, command: str) -> str:
    value = raw[len(command):].strip()
    if not value:
        raise ValueError(f"{command} requires an argument")
    return value.rstrip(".").strip()


def normalize_prompt(prompt: str) -> tuple[str, str]:
    """Map task commands to the prompt templates used by LocateAnything."""
    raw = prompt.strip()
    if not raw:
        raise ValueError("prompt must not be empty")

    if raw == "/text":
        return "Detect all the text in box format.", "text_ocr"
    if raw == "/detect" or raw.startswith("/detect "):
        categories = _require_task_argument(raw, "/detect")
        categories = "</c>".join(
            part.strip() for part in categories.split(",") if part.strip()
        )
        if not categories:
            raise ValueError("/detect requires at least one category")
        return (
            "Locate all the instances that matches the following description: "
            f"{categories}.",
            "object_detection",
        )
    if raw == "/layout" or raw.startswith("/layout "):
        categories = _require_task_argument(raw, "/layout")
        categories = "</c>".join(
            part.strip() for part in categories.split(",") if part.strip()
        )
        if not categories:
            raise ValueError("/layout requires at least one category")
        return (
            "Detect all the objects in the image that belong to the category set: "
            f"{categories}.",
            "layout_grounding",
        )
    if raw == "/ground" or raw.startswith("/ground "):
        phrase = _require_task_argument(raw, "/ground")
        return (
            f"Locate all the instances that match the following description: {phrase}.",
            "referring_comprehension",
        )
    if raw == "/ground_single" or raw.startswith("/ground_single "):
        phrase = _require_task_argument(raw, "/ground_single")
        return (
            f"Locate a single instance that matches the following description: {phrase}.",
            "referring_comprehension_single",
        )
    if raw == "/ground_text" or raw.startswith("/ground_text "):
        phrase = _require_task_argument(raw, "/ground_text")
        return f"Please locate the text referred as {phrase}.", "text_ocr_grounding"
    if raw == "/gui" or raw.startswith("/gui "):
        phrase = _require_task_argument(raw, "/gui")
        return f"Point to: {phrase}.", "gui_grounding"
    if raw == "/gui_box" or raw.startswith("/gui_box "):
        phrase = _require_task_argument(raw, "/gui_box")
        return (
            f"Locate the region that matches the following description: {phrase}.",
            "gui_grounding_box",
        )
    if raw == "/point" or raw.startswith("/point "):
        phrase = _require_task_argument(raw, "/point")
        return f"Point to: {phrase}.", "point_localization"
    commands = "; ".join(TASK_COMMANDS)
    if raw.startswith("/"):
        raise ValueError(f"unknown task command {raw.split()[0]!r}; available: {commands}")
    raise ValueError(f"use a task command; available: {commands}")


def build_prompt(prompt: str) -> str:
    normalized_prompt, _ = normalize_prompt(prompt)
    template = (
        "<|im_start|>system\n"
        "You are a helpful assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<image-1>"
        f"{normalized_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    image_placeholder = (
        "<image 1><img>" + IMAGE_TOKEN * IMAGE_TOKENS + "</img>"
    )
    return template.replace("<image-1>", image_placeholder)


def load_tokenizer(tokenizer_dir: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))


def tokenize_prompt(tokenizer_dir: Path, prompt: str, tokenizer=None) -> np.ndarray:
    tokenizer = tokenizer or load_tokenizer(tokenizer_dir)
    token_ids = tokenizer.encode(build_prompt(prompt), add_special_tokens=True).ids
    values = np.asarray(token_ids, dtype=np.int32)
    image_token_id = int(tokenizer.token_to_id(IMAGE_TOKEN))
    if values.size > 1024:
        raise ValueError(f"prompt has {values.size} tokens; compiled prefill limit is 1024")
    if int(np.count_nonzero(values == image_token_id)) != IMAGE_TOKENS:
        raise ValueError("tokenized prompt does not contain exactly 576 image tokens")
    return np.ascontiguousarray(values)


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[str, float]:
    started = time.monotonic()
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    elapsed = time.monotonic() - started
    log_path.write_text(process.stdout + f"\nelapsed_seconds={elapsed:.3f}\n", encoding="utf-8")
    if process.returncode != 0:
        sys.stderr.write(process.stdout)
        raise RuntimeError(
            f"command failed with exit code {process.returncode}; log: {log_path}"
        )
    return process.stdout, elapsed


def read_generation(path: Path) -> tuple[str, list[int]]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = value
    if "token_ids" not in fields or "stop_reason" not in fields:
        raise ValueError(f"invalid Language output: {path}")
    token_ids = [int(value) for value in fields["token_ids"].split(",") if value]
    return fields["stop_reason"], token_ids


def decode_tokens(tokenizer_dir: Path, token_ids: list[int], tokenizer=None) -> str:
    tokenizer = tokenizer or load_tokenizer(tokenizer_dir)
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def invert_coordinate(value: int, axis: int, transform: dict[str, object]) -> float:
    source_size = transform["source_size"]
    scale_xy = transform["scale_xy"]
    padding = transform["padding_ltrb"]
    source_limit = float(source_size[axis])
    target_pixel = value / 1000.0 * IMAGE_SIZE
    source_pixel = (target_pixel - float(padding[axis])) / float(scale_xy[axis])
    return min(source_limit, max(0.0, source_pixel))


def parse_detections(text: str, transform: dict[str, object]) -> list[dict[str, object]]:
    pattern = re.compile(r"<ref>(.*?)</ref>|<box>((?:<\d{1,4}>)+)</box>")
    coordinate_pattern = re.compile(r"<(\d{1,4})>")
    label = ""
    detections: list[dict[str, object]] = []
    for match in pattern.finditer(text):
        if match.group(1) is not None:
            label = match.group(1)
            continue
        coordinates = [int(value) for value in coordinate_pattern.findall(match.group(2))]
        if (
            len(coordinates) != 4
            or any(value > 1000 for value in coordinates)
            or coordinates[0] >= coordinates[2]
            or coordinates[1] >= coordinates[3]
        ):
            continue
        xyxy = [
            invert_coordinate(value, index % 2, transform)
            for index, value in enumerate(coordinates)
        ]
        if xyxy[0] >= xyxy[2] or xyxy[1] >= xyxy[3]:
            continue
        detections.append(
            {
                "label": label,
                "bbox_profile_1000": coordinates,
                "bbox_xyxy": [round(value, 2) for value in xyxy],
            }
        )
    return detections


def box_iou_xyxy(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def class_aware_nms(
    detections: list[dict[str, object]],
    iou_threshold: float = DEFAULT_NMS_IOU,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Remove near-identical boxes within a label while preserving generation order."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be between 0 and 1")

    kept: list[dict[str, object]] = []
    suppressed: list[dict[str, object]] = []
    for detection in detections:
        label = " ".join(str(detection.get("label") or "").split()).casefold()
        box = detection.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            kept.append(detection)
            continue

        duplicate: tuple[int, float] | None = None
        for kept_index, candidate in enumerate(kept):
            candidate_label = " ".join(
                str(candidate.get("label") or "").split()
            ).casefold()
            candidate_box = candidate.get("bbox_xyxy")
            if candidate_label != label or not isinstance(candidate_box, list):
                continue
            overlap = box_iou_xyxy(box, candidate_box)
            if overlap >= iou_threshold:
                duplicate = (kept_index, overlap)
                break

        if duplicate is None:
            kept.append(detection)
            continue
        rejected = dict(detection)
        rejected["nms_iou"] = round(duplicate[1], 6)
        rejected["suppressed_by"] = duplicate[0] + 1
        suppressed.append(rejected)
    return kept, suppressed


def postprocess_detections(
    detections: list[dict[str, object]],
    task: str,
    iou_threshold: float = DEFAULT_NMS_IOU,
    enabled: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not enabled or task != "object_detection":
        return list(detections), []
    return class_aware_nms(detections, iou_threshold)


def parse_points(text: str, transform: dict[str, object]) -> list[dict[str, object]]:
    pattern = re.compile(r"<ref>(.*?)</ref>|<box>((?:<\d{1,4}>)+)</box>")
    coordinate_pattern = re.compile(r"<(\d{1,4})>")
    label = ""
    points: list[dict[str, object]] = []
    for match in pattern.finditer(text):
        if match.group(1) is not None:
            label = match.group(1)
            continue
        coordinates = [int(value) for value in coordinate_pattern.findall(match.group(2))]
        if len(coordinates) != 2 or any(value > 1000 for value in coordinates):
            continue
        xy = [
            invert_coordinate(value, index, transform)
            for index, value in enumerate(coordinates)
        ]
        points.append(
            {
                "label": label,
                "point_profile_1000": coordinates,
                "point_xy": [round(value, 2) for value in xy],
            }
        )
    return points


def annotated_output_path(image_path: Path, task: str, output_dir: Path | None = None) -> Path:
    """Return a unique image path for compatibility with older callers."""

    directory = (output_dir or (Path.cwd() / "output")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    image_name = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem).strip("._") or "image"
    task_name = re.sub(r"[^A-Za-z0-9._-]+", "_", task).strip("._") or "localization"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp += f"_{uuid.uuid4().hex[:12]}"
    return directory / f"{image_name}_{task_name}_{timestamp}.png"


def save_annotated_image(
    image_path: Path,
    detections: list[dict[str, object]],
    points: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Draw parsed predictions in original-image coordinates and save a PNG."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, int(round(min(width, height) * 0.004)))
    point_radius = max(5, line_width * 2)
    palette = (
        (0, 220, 120),
        (0, 145, 255),
        (255, 90, 75),
        (255, 190, 0),
        (175, 95, 255),
        (0, 205, 205),
    )

    def caption(index: int, label: object) -> str:
        value = str(label or "").strip()
        return f"{index}: {value}" if value else str(index)

    def draw_caption(position: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
        try:
            measured = draw.textbbox((0, 0), text)
        except UnicodeEncodeError:
            text = text.encode("ascii", "replace").decode("ascii")
            measured = draw.textbbox((0, 0), text)
        padding = max(2, line_width // 2)
        text_width = max(1, measured[2] - measured[0])
        text_height = max(1, measured[3] - measured[1])
        x = max(0, min(int(position[0]), max(0, width - text_width - padding)))
        y = max(0, min(int(position[1]), max(0, height - text_height - padding)))
        bounds = draw.textbbox((x, y), text)
        background = (
            max(0, bounds[0] - padding),
            max(0, bounds[1] - padding),
            min(width - 1, bounds[2] + padding),
            min(height - 1, bounds[3] + padding),
        )
        if background[0] <= background[2] and background[1] <= background[3]:
            draw.rectangle(background, fill=color)
        draw.text((x, y), text, fill=(0, 0, 0))

    for index, item in enumerate(detections, 1):
        raw = item.get("bbox_xyxy")
        if not isinstance(raw, list) or len(raw) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in raw)
        left = max(0, min(width - 1, int(round(min(x1, x2)))))
        top = max(0, min(height - 1, int(round(min(y1, y2)))))
        right = max(0, min(width - 1, int(round(max(x1, x2)))))
        bottom = max(0, min(height - 1, int(round(max(y1, y2)))))
        color = palette[(index - 1) % len(palette)]
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
        draw_caption((left + line_width, top + line_width), caption(index, item.get("label")), color)

    point_offset = len(detections)
    for point_index, item in enumerate(points, 1):
        raw = item.get("point_xy")
        if not isinstance(raw, list) or len(raw) != 2:
            continue
        x = max(0, min(width - 1, int(round(float(raw[0])))))
        y = max(0, min(height - 1, int(round(float(raw[1])))))
        display_index = point_offset + point_index
        color = palette[(display_index - 1) % len(palette)]
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            outline=color,
            width=line_width,
        )
        draw.line((x - point_radius, y, x + point_radius, y), fill=color, width=line_width)
        draw.line((x, y - point_radius, x, y + point_radius), fill=color, width=line_width)
        label_x = min(width - 1, x + point_radius + line_width)
        label_y = max(0, y - point_radius)
        draw_caption((label_x, label_y), caption(display_index, item.get("label")), color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    runtime = load_runtime_config_from_args(argv)
    parser = argparse.ArgumentParser(
        description="Run an image and prompt through LocateAnything Vision and Language HBM.",
        epilog="Task commands:\n  " + "\n  ".join(TASK_COMMANDS),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image_positional", type=Path, nargs="?")
    parser.add_argument("prompt_positional", nargs="?")
    parser.add_argument(
        "-c", "--config", type=Path, default=runtime.source,
        help=f"runtime config (default: {runtime.source})",
    )
    parser.add_argument("-i", "--image", type=Path)
    parser.add_argument("-p", "--prompt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for prediction.json, annotated.png, timings.json, and logs/",
    )
    parser.add_argument("-o", "--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-new-tokens", type=int, default=runtime.default_max_new_tokens,
    )
    parser.add_argument(
        "--generation-mode",
        choices=("hybrid", "slow"),
        default=runtime.default_generation_mode,
        help=("hybrid=q6 PBD with q1 fallback; slow=q1 AR "
              f"(default: {runtime.default_generation_mode})"),
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=runtime.default_nms_iou,
        help=f"same-label Detection NMS threshold (default: {runtime.default_nms_iou:.2f})",
    )
    parser.add_argument(
        "--no-nms",
        action="store_true",
        help="disable Detection NMS while retaining raw model boxes",
    )
    args = parser.parse_args(argv)
    args.runtime = runtime
    args.image = args.image or args.image_positional
    args.prompt = args.prompt or args.prompt_positional
    if args.output is not None and args.output_dir is not None:
        parser.error("use --output-dir or the deprecated --output option, not both")
    if args.image is None or args.prompt is None:
        parser.error("provide IMAGE and PROMPT, either positionally or with --image/--prompt")
    return args


def main() -> int:
    # Keep the primary entry point compatible with an older wrapper copied to
    # a board.  The interactive frontend owns its own argument parser.
    if "--interactive" in sys.argv[1:]:
        sys.argv.remove("--interactive")
        from cli import main as interactive_main

        return interactive_main()
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be between 0 and 1")

    runtime: RuntimeConfig = args.runtime
    require_runtime_paths(runtime)
    tokenizer_dir = runtime.tokenizer_dir
    env = build_runtime_environment(runtime)
    task_prompt = args.prompt
    normalized_prompt, task = normalize_prompt(task_prompt)
    started = time.monotonic()
    compatibility_output = args.output.resolve() if args.output else None
    paths = create_prediction_paths(
        runtime.layout_root,
        args.image,
        output_dir=(compatibility_output.parent if compatibility_output else args.output_dir),
    )
    output_path = compatibility_output or paths.prediction
    runtime_log = paths.runtime_log
    monitor = ResourceMonitor(interval_seconds=runtime.telemetry_interval_seconds)
    monitor.start()
    monitor.begin_request()
    try:
        vision_stage_started = time.monotonic()
        vision_input, transform = prepare_image(args.image)
        with tempfile.TemporaryDirectory(prefix="locateanything-") as temporary:
            work_dir = Path(temporary)
            vision_input_path = work_dir / "vision_input.f16.bin"
            visual_features_path = work_dir / "visual_features.f16.bin"
            prompt_tokens_path = work_dir / "prompt_tokens.i32.bin"
            generation_path = work_dir / "generation.txt"
            vision_input.tofile(vision_input_path)

            _, vision_elapsed = run_command(
                [
                    str(runtime.vision_runner),
                    "--model",
                    str(runtime.vision_model),
                    "--input",
                    str(vision_input_path),
                    "--output",
                    str(visual_features_path),
                ],
                runtime_log,
                env,
            )
            vision_log = runtime_log.read_text(encoding="utf-8")
            vision_stage_seconds = time.monotonic() - vision_stage_started

            language_stage_started = time.monotonic()
            prompt_tokens = tokenize_prompt(tokenizer_dir, task_prompt)
            prompt_tokens.tofile(prompt_tokens_path)
            _, language_elapsed = run_command(
                language_runner_command(runtime) + [
                    "--mode",
                    "all",
                    "--tokens",
                    str(prompt_tokens_path),
                    "--visual",
                    str(visual_features_path),
                    "--generation-mode",
                    args.generation_mode,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--output",
                    str(generation_path),
                ],
                runtime_log,
                env,
            )
            language_log = runtime_log.read_text(encoding="utf-8")
            language_stage_seconds = time.monotonic() - language_stage_started
            runtime_log.write_text(
                "VISION\n"
                + vision_log
                + "\nLANGUAGE\n"
                + language_log,
                encoding="utf-8",
            )
            stop_reason, token_ids = read_generation(generation_path)

        postprocess_started = time.monotonic()
        text = decode_tokens(tokenizer_dir, token_ids)
        raw_detections = parse_detections(text, transform)
        detections, suppressed_detections = postprocess_detections(
            raw_detections,
            task,
            iou_threshold=args.nms_iou,
            enabled=not args.no_nms,
        )
        points = parse_points(text, transform)
        annotated_image = None
        if detections or points:
            annotated_image = paths.annotated_image
            save_annotated_image(args.image, detections, points, annotated_image)
        postprocess_seconds = time.monotonic() - postprocess_started
        total_seconds = time.monotonic() - started
        monitor.end_request()
        resource_summary = monitor.summary()
    finally:
        monitor.end_request()
        monitor.stop()

    language_tokens_per_second = (
        len(token_ids) / language_stage_seconds if language_stage_seconds > 0 else 0.0
    )

    timing_data = {
        "schema_version": 1,
        "stages_seconds": {
            "vision": round(vision_stage_seconds, 6),
            "language": round(language_stage_seconds, 6),
            "postprocess": round(postprocess_seconds, 6),
        },
        "runner_seconds": {
            "vision_hbm": round(vision_elapsed, 6),
            "language_hbm": round(language_elapsed, 6),
        },
        "total_seconds": round(total_seconds, 6),
        "resources": resource_summary.as_dict(),
    }
    paths.timings.write_text(
        json.dumps(timing_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "runtime": {
            "version": RUNTIME_VERSION,
            "config": str(runtime.source),
            "model_type": runtime.model_type,
            "runtime_specification": runtime.specification(),
        },
        "image": str(args.image.resolve()),
        "prompt": args.prompt,
        "normalized_prompt": normalized_prompt,
        "task": task,
        "text": text,
        "raw_detections": raw_detections,
        "detections": detections,
        "suppressed_detections": suppressed_detections,
        "points": points,
        "annotated_image": str(annotated_image) if annotated_image else None,
        "generation": {
            "mode": args.generation_mode,
            "stop_reason": stop_reason,
            "complete": stop_reason == "im_end",
            "token_count": len(token_ids),
            "max_new_tokens": args.max_new_tokens,
        },
        "image_transform": transform,
        "postprocess": {
            "method": "class_aware_nms"
            if task == "object_detection" and not args.no_nms
            else "none",
            "iou_threshold": args.nms_iou,
            "raw_detection_count": len(raw_detections),
            "kept_detection_count": len(detections),
            "suppressed_detection_count": len(suppressed_detections),
        },
        "timings": timing_data,
        "resources": resource_summary.as_dict(),
        "elapsed_seconds": round(total_seconds, 3),
        "runtime_log": str(runtime_log),
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{BOLD}{CYAN}LocateAnything result{RESET}")
    print(f"  Status       {'complete' if stop_reason == 'im_end' else 'truncated'}")
    print(f"  Image        {args.image}")
    print(f"  Task         {task}")
    print(f"  Response     {text}")
    print(f"{BOLD}{CYAN}Performance{RESET}")
    print(f"  Vision       {vision_stage_seconds * 1000.0:.3f} ms")
    print(
        f"  Language     {language_stage_seconds * 1000.0:.3f} ms  "
        f"{len(token_ids)} tokens  {language_tokens_per_second:.3f} tokens/s"
    )
    print(f"  Postprocess  {postprocess_seconds * 1000.0:.3f} ms")
    print(f"  End-to-end   {total_seconds * 1000.0:.3f} ms")
    print(f"{BOLD}{CYAN}Resources{RESET}")
    for line in resource_summary_lines(resource_summary):
        print(line)
    print(f"{BOLD}{CYAN}Predictions{RESET}")
    print(f"  Boxes        {len(detections)}")
    if suppressed_detections:
        print(
            f"  NMS removed  {len(suppressed_detections)} same-label boxes "
            f"at IoU >= {args.nms_iou:.2f}"
        )
    for index, detection in enumerate(detections[:20], 1):
        print(
            f"    {index}. {detection['label']!r}  pixels={detection['bbox_xyxy']}"
        )
    if len(detections) > 20:
        print(f"    ... {len(detections) - 20} more")
    print(f"  Points       {len(points)}")
    for index, point in enumerate(points[:20], 1):
        print(f"    {index}. {point['label']!r}  pixels={point['point_xy']}")
    if len(points) > 20:
        print(f"    ... {len(points) - 20} more")
    print(f"{BOLD}{CYAN}Saved{RESET}")
    if annotated_image:
        print(f"  Annotated image  {annotated_image}")
    print(f"  Prediction       {output_path}")
    print(f"  Timings          {paths.timings}")
    print(f"  Run directory    {paths.root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
