#!/usr/bin/env python3
"""Orchestrate LocateAnything Prepare, Calibrate, Build, and Verify stages.

Source selection produces the frozen dataset index consumed by this CLI. Numerical
calibration, BC export, HBDK compilation, and validation algorithms remain in
``compiler/scripts`` and ``compiler/leap_llm``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - compiler environment owns PyYAML
    raise SystemExit(
        "PyYAML is required; install the compiler package with "
        "`python -m pip install -e compiler`"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPILER_ROOT = PROJECT_ROOT / "compiler"
SCRIPTS_ROOT = COMPILER_ROOT / "scripts"
CALIBRATION_SCRIPTS = SCRIPTS_ROOT / "calibration"
BUILD_SCRIPTS = SCRIPTS_ROOT / "build"
VALIDATE_SCRIPTS = SCRIPTS_ROOT / "validate"
DEFAULT_CONFIG = COMPILER_ROOT / "config.yaml"
COMPONENTS = ("vision", "language", "all")
BUILD_TARGETS = ("bc", "hbm")
PROGRESS_MODES = ("auto", "bar", "log", "off")
VERIFY_STAGES = ("specification", "pipeline", "task", "all")
if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

from leap_llm.language_graphs import (  # noqa: E402
    LANGUAGE_GRAPH_SET_NAMES,
    language_graph_set,
)
EXPECTED_CALIBRATION_TASK_COUNTS = {
    "detection": 660,
    "gui": 150,
    "referring": 120,
    "ocr": 120,
    "layout": 90,
    "pointing": 60,
}
EXPECTED_CALIBRATION_SOURCE_ROLE_COUNTS = {
    "coco_detection": 240,
    "openimages_v6": 90,
    "v3det": 60,
    "paco": 50,
    "bdd100k": 50,
    "egoobjects": 40,
    "humanparts": 40,
    "mot17det": 45,
    "mot20det": 45,
    "groundcua": 150,
    "refcocog": 120,
    "hiertext": 120,
    "doclaynet": 90,
    "pixmo_points": 60,
}
EXPECTED_COCO_STRATUM_COUNTS = {"single": 80, "double": 100, "multi": 60}
EXPECTED_DATASET_INDEX_SHA256 = (
    "521c9203579b165b619934684ca0dd44f9a33dc9c68e0bb6abb17f481d17850b"
)
EXPECTED_CHECKPOINT_SHA256 = {
    "model-00001-of-00002.safetensors": (
        "923cfc10fed19808067da6df85a9a4220ddc1f9eb91ceee94c0fecd05d0f2d58"
    ),
    "model-00002-of-00002.safetensors": (
        "3459ba101f40594f3f62d3312014f1f8378b4ba3da3b1d562480045938fc7d47"
    ),
}
EXPECTED_CHECKPOINT_INDEX_SHA256 = (
    "2ecc63fee5f958ffc8142fa29ff7b704a58e80349e9c9ca155a9710d97700271"
)
PATH_ENV_OVERRIDES = {
    "model": "LA_MODEL_PATH",
    "upstream_source": "LA_UPSTREAM_SOURCE",
}
PATH_ROOT_OVERRIDES = {
    "model": ("LA_MODEL_ROOT", Path("artifacts/models")),
    "selected_jsonl": ("LA_CALIBRATION_ROOT", Path("artifacts/calibration")),
    "generated_dir": ("LA_CALIBRATION_ROOT", Path("artifacts/calibration")),
    "generated_jsonl": ("LA_CALIBRATION_ROOT", Path("artifacts/calibration")),
    "calibration_dir": ("LA_CALIBRATION_ROOT", Path("artifacts/calibration")),
    "scale_manifest": ("LA_CALIBRATION_ROOT", Path("artifacts/calibration")),
    "coverage_json": ("LA_CALIBRATION_ROOT", Path("artifacts/calibration")),
    "build_root": ("LA_BUILD_ROOT", Path("artifacts/builds")),
    "log_root": ("LA_LOG_ROOT", Path("artifacts/logs")),
    "verification_root": ("LA_EVALUATION_ROOT", Path("artifacts/evaluation")),
}


class ConfigurationError(ValueError):
    """Raised when a build config violates the fixed LocateAnything specification."""


@dataclass(frozen=True)
class PlanStep:
    label: str
    command: tuple[str, ...]
    cwd: Path = PROJECT_ROOT
    env: Mapping[str, str] = field(default_factory=dict)
    note: str | None = None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_file(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(item) for item in (*chain, path))
        raise ConfigurationError(f"config inheritance cycle: {cycle}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    config = _mapping(raw, f"config {path}")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(str(parent)).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _merge_config(
        _load_config_file(parent_path, (*chain, path)),
        config,
    )


def load_config(path: Path) -> dict[str, Any]:
    config = _load_config_file(path)
    validate_config(config)
    return config


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ConfigurationError("schema_version must be 1")

    paths = _mapping(config.get("paths"), "paths")
    required_paths = {
        "model", "upstream_source", "selected_jsonl", "generated_dir",
        "generated_jsonl", "calibration_dir", "scale_manifest", "coverage_json",
        "build_root", "log_root", "verification_root",
    }
    missing_paths = sorted(required_paths - paths.keys())
    if missing_paths:
        raise ConfigurationError(f"paths is missing: {', '.join(missing_paths)}")

    model = _mapping(config.get("model"), "model")
    image_width = _positive_int(model.get("image_width"), "model.image_width")
    image_height = _positive_int(model.get("image_height"), "model.image_height")
    patch_size = _positive_int(model.get("patch_size"), "model.patch_size")
    spatial_merge = _positive_int(model.get("spatial_merge"), "model.spatial_merge")
    if image_width != 672 or image_height != 672:
        raise ConfigurationError("release image size is fixed at 672x672")
    if patch_size != 14 or spatial_merge != 2:
        raise ConfigurationError("Vision specification requires patch_size=14, spatial_merge=2")
    if image_width % (patch_size * spatial_merge) or image_height % (patch_size * spatial_merge):
        raise ConfigurationError("image dimensions must align to patch_size * spatial_merge")
    if model.get("resize_mode") != "letterbox":
        raise ConfigurationError("release image preprocessing requires resize_mode=letterbox")
    if model.get("letterbox_fill") != 128:
        raise ConfigurationError("release letterbox_fill must be 128")
    if model.get("hidden_size") != 2048 or model.get("vocab_size") != 152681:
        raise ConfigurationError("hidden_size=2048 and vocab_size=152681 are fixed model specifications")
    if model.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ConfigurationError("model.checkpoint_sha256 does not match the frozen checkpoint")
    if model.get("checkpoint_index_sha256") != EXPECTED_CHECKPOINT_INDEX_SHA256:
        raise ConfigurationError(
            "model.checkpoint_index_sha256 does not match the frozen checkpoint index"
        )

    calibration = _mapping(config.get("calibration"), "calibration")
    sample_count = _positive_int(calibration.get("sample_count"), "calibration.sample_count")
    checkpoint = _positive_int(
        calibration.get("checkpoint_samples"), "calibration.checkpoint_samples"
    )
    if sample_count != 1200 or checkpoint != 512:
        raise ConfigurationError(
            "release calibration requires sample_count=1200 and checkpoint_samples=512"
        )
    if calibration.get("max_new_tokens") != 1024:
        raise ConfigurationError("release calibration requires max_new_tokens=1024")
    if calibration.get("image_token_id") != 151665:
        raise ConfigurationError("release calibration requires image_token_id=151665")
    if calibration.get("prepare_dtype") != "bfloat16":
        raise ConfigurationError("release Prepare tensors require bfloat16")
    if calibration.get("calibrate_dtype") != "float16":
        raise ConfigurationError("release activation calibration requires float16")
    if calibration.get("task_counts") != EXPECTED_CALIBRATION_TASK_COUNTS:
        raise ConfigurationError("release calibration task_counts do not match the 1200-sample profile")
    if calibration.get("source_role_counts") != EXPECTED_CALIBRATION_SOURCE_ROLE_COUNTS:
        raise ConfigurationError(
            "release calibration source_role_counts do not match the 1200-sample profile"
        )
    if calibration.get("coco_stratum_counts") != EXPECTED_COCO_STRATUM_COUNTS:
        raise ConfigurationError(
            "release calibration coco_stratum_counts do not match the 500-sample COCO profile"
        )
    if calibration.get("dataset_index_sha256") != EXPECTED_DATASET_INDEX_SHA256:
        raise ConfigurationError(
            "calibration.dataset_index_sha256 does not match the frozen dataset index"
        )

    language = _mapping(config.get("language"), "language")
    chunk_size = _positive_int(language.get("chunk_size"), "language.chunk_size")
    cache_len = _positive_int(language.get("cache_len"), "language.cache_len")
    if chunk_size != 1024 or cache_len != 4096:
        raise ConfigurationError("Language specification requires chunk_size=1024, cache_len=4096")
    if language.get("pbd_query_len") != 6 or language.get("ar_query_len") != 1:
        raise ConfigurationError("LocateAnything requires PBD q=6 and AR q=1")
    if language.get("decoder_w_bits") != 8 or language.get("lm_head_w_bits") != 8:
        raise ConfigurationError("release Language and LM Head weights must use W8")
    try:
        language_graph_set(str(language.get("graph_set")))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    vision = _mapping(config.get("vision"), "vision")
    if vision.get("w_bits") != 8:
        raise ConfigurationError("release Vision weights must use W8")

    build = _mapping(config.get("build"), "build")
    if build.get("march") != "nash-p":
        raise ConfigurationError("release target march must be nash-p")
    _positive_int(build.get("jobs"), "build.jobs")
    cores = _mapping(build.get("cores"), "build.cores")
    for name in ("vision", "prefill", "pbd", "ar"):
        if cores.get(name) not in {1, 2, 4}:
            raise ConfigurationError(f"build.cores.{name} must be 1, 2, or 4")

    verification = _mapping(config.get("verification"), "verification")
    iou_threshold = verification.get("iou_threshold")
    if not isinstance(iou_threshold, (int, float)) or not 0.0 < iou_threshold <= 1.0:
        raise ConfigurationError("verification.iou_threshold must be in (0, 1]")
    for key in ("reference_jsonl", "predictions_jsonl"):
        if not verification.get(key):
            raise ConfigurationError(f"verification.{key} must be configured")


def _resolve_config_path(
    raw: Any,
    *,
    root_override: tuple[str, Path] | None = None,
) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    if expanded.is_absolute():
        return expanded.resolve()

    if root_override is not None:
        environment, anchor = root_override
        configured_root = os.environ.get(environment)
        if configured_root:
            try:
                suffix = expanded.relative_to(anchor)
            except ValueError as exc:
                raise ConfigurationError(
                    f"{environment} cannot rebase path outside {anchor}: {expanded}"
                ) from exc
            return (Path(configured_root).expanduser() / suffix).resolve()

    artifacts_root = os.environ.get("LA_ARTIFACTS_ROOT")
    if artifacts_root:
        try:
            suffix = expanded.relative_to("artifacts")
        except ValueError:
            pass
        else:
            return (Path(artifacts_root).expanduser() / suffix).resolve()
    return (PROJECT_ROOT / expanded).resolve()


def resolve_path(config: Mapping[str, Any], key: str, override: str | None = None) -> Path:
    environment = PATH_ENV_OVERRIDES.get(key)
    raw = override
    if raw is None and environment:
        raw = os.environ.get(environment)
    if raw is None:
        raw = _mapping(config["paths"], "paths")[key]
        return _resolve_config_path(raw, root_override=PATH_ROOT_OVERRIDES.get(key))
    return _resolve_config_path(raw)


def select_components(value: str) -> tuple[str, ...]:
    return ("vision", "language") if value == "all" else (value,)


def selected_graph_set(
    args: argparse.Namespace, config: Mapping[str, Any]
):
    requested = args.graph_set or config["language"]["graph_set"]
    return language_graph_set(str(requested))


def python_command(config: Mapping[str, Any]) -> str:
    return str(_mapping(config["build"], "build").get("python") or sys.executable)


def common_env(config: Mapping[str, Any], progress: str) -> dict[str, str]:
    build = _mapping(config["build"], "build")
    env = {
        "REPO_ROOT": str(PROJECT_ROOT),
        "PYTHON_BIN": python_command(config),
        "PYTHONUNBUFFERED": "1",
        "LA_PROGRESS": progress,
    }
    if progress == "bar":
        env["TQDM_DISABLE"] = "0"
    elif progress in {"log", "off"}:
        env["TQDM_DISABLE"] = "1"
    env["CONDA_ENV"] = str(build.get("conda_env", "oellm_clean"))
    return env


def prepare_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    calibration = _mapping(config["calibration"], "calibration")
    model = _mapping(config["model"], "model")
    if args.max_new_tokens is not None and args.max_new_tokens != 1024:
        raise ConfigurationError("release preparation fixes --max-new-tokens at 1024")
    selected = resolve_path(config, "selected_jsonl", args.selected_jsonl)
    output_dir = resolve_path(config, "generated_dir", args.output_dir)
    build = _mapping(config["build"], "build")
    report_json = (
        resolve_path_value(args.report_json)
        if args.report_json
        else output_dir / "prepare_preflight.json"
    )
    preflight_command = (
        python_command(config),
        str(CALIBRATION_SCRIPTS / "preflight.py"),
        "--config", str(args.config.resolve()),
        "--selected-jsonl", str(selected),
        "--model-path", str(resolve_path(config, "model", args.model_path)),
        "--upstream-repo", str(resolve_path(config, "upstream_source", args.upstream_source)),
        "--report-json", str(report_json),
    )
    preflight_step = PlanStep(
        "validate frozen Prepare inputs without loading model weights",
        preflight_command,
        note="static data, processor, and tokenizer check only; no CUDA or model inference",
    )
    if args.preflight_only:
        return [preflight_step]
    env = common_env(config, args.progress)
    env.update({
        "SELECTED_JSONL": str(selected),
        "OUTPUT_DIR": str(output_dir),
        "UPSTREAM_REPO": str(resolve_path(config, "upstream_source", args.upstream_source)),
        "MODEL_PATH": str(resolve_path(config, "model", args.model_path)),
        "DEVICE": args.device or str(calibration["device"]),
        "DTYPE": args.dtype or str(calibration["prepare_dtype"]),
        "IMAGE_WIDTH": str(model["image_width"]),
        "IMAGE_HEIGHT": str(model["image_height"]),
        "RESIZE_MODE": str(model["resize_mode"]),
        "LETTERBOX_FILL": str(model["letterbox_fill"]),
        "PATCH_SIZE": str(model["patch_size"]),
        "MERGE_SIZE": str(model["spatial_merge"]),
        "HIDDEN_SIZE": str(model["hidden_size"]),
        "PREFILL_LIMIT": str(config["language"]["chunk_size"]),
        "MAX_NEW_TOKENS": str(args.max_new_tokens or calibration["max_new_tokens"]),
        "SLOW_SAMPLES": str(args.slow_samples or calibration["slow_samples"]),
        "SEED": str(calibration["seed"]),
        "RESUME": "1" if args.resume else "0",
    })
    command = (str(build.get("bash", "bash")), str(CALIBRATION_SCRIPTS / "prepare.sh"))
    return [preflight_step, PlanStep("prepare calibration tensors", command, env=env)]


def calibrate_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    calibration = _mapping(config["calibration"], "calibration")
    language = _mapping(config["language"], "language")
    build = _mapping(config["build"], "build")
    graph_set = selected_graph_set(args, config)
    requested_samples = args.max_samples or calibration["sample_count"]
    requested_checkpoint = args.checkpoint_samples or calibration["checkpoint_samples"]
    if requested_samples != 1200 or requested_checkpoint != 512:
        raise ConfigurationError(
            "release calibration fixes --max-samples=1200 and "
            "--checkpoint-samples=512"
        )
    env = common_env(config, args.progress)
    env.update({
        "GENERATED_JSONL": str(resolve_path(config, "generated_jsonl", args.generated_jsonl)),
        "SELECTED_JSONL": str(resolve_path(config, "selected_jsonl", args.selected_jsonl)),
        "UPSTREAM_REPO": str(
            resolve_path(config, "upstream_source", args.upstream_source)
        ),
        "MODEL_PATH": str(resolve_path(config, "model", args.model_path)),
        "OUTPUT_DIR": str(resolve_path(config, "calibration_dir", args.output_dir)),
        "DEVICE": args.device or str(calibration["device"]),
        "DTYPE": args.dtype or str(calibration["calibrate_dtype"]),
        "CALIBRATION_COMPONENT": "all" if args.component == "all" else args.component,
        "CHUNK_SIZE": str(language["chunk_size"]),
        "CACHE_LEN": str(language["cache_len"]),
        "LM_HEAD_W_BITS": str(language["lm_head_w_bits"]),
        "MAX_SAMPLES": str(requested_samples),
        "CHECKPOINT_SAMPLES": str(requested_checkpoint),
        "IMAGE_TOKEN_ID": str(calibration["image_token_id"]),
        "REPLAY_SEED": str(calibration["seed"]),
        "LANGUAGE_GRAPH_SET": graph_set.name,
        "RESUME": "1" if args.resume else "0",
    })
    rotation = calibration.get("hidden_rotation_path")
    if rotation:
        env["HIDDEN_ROTATION_PATH"] = str(resolve_path_value(rotation))
    note = None
    if args.resume:
        note = "calibration replay is atomic; only completed statistics are reused"
    command = (str(build.get("bash", "bash")), str(CALIBRATION_SCRIPTS / "calibrate.sh"))
    return [PlanStep("collect activation statistics", command, env=env, note=note)]


def resolve_path_value(value: Any) -> Path:
    return _resolve_config_path(value)


def resolve_evaluation_path(value: Any) -> Path:
    return _resolve_config_path(
        value,
        root_override=("LA_EVALUATION_ROOT", Path("artifacts/evaluation")),
    )


def build_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    calibration = _mapping(config["calibration"], "calibration")
    model = _mapping(config["model"], "model")
    language = _mapping(config["language"], "language")
    vision = _mapping(config["vision"], "vision")
    build = _mapping(config["build"], "build")
    cores = _mapping(build["cores"], "build.cores")
    graph_set = selected_graph_set(args, config)
    build_root = resolve_path(config, "build_root", args.output_dir)
    log_root = resolve_path(config, "log_root")
    bash = str(build.get("bash", "bash"))
    steps: list[PlanStep] = []
    for component in select_components(args.component):
        output = build_root / component
        env = common_env(config, args.progress)
        env.update({
            "INPUT_MODEL_PATH": str(resolve_path(config, "model", args.model_path)),
            "OUTPUT_MODEL_PATH": str(output),
            "CALIB_JSON": str(resolve_path(config, "selected_jsonl")),
            "GENERATED_JSON": str(resolve_path(config, "generated_jsonl")),
            "CALIBRATION_SCALE_MANIFEST": str(resolve_path(config, "scale_manifest")),
            "CALIBRATION_COVERAGE_JSON": str(resolve_path(config, "coverage_json")),
            "EXPECTED_SAMPLES": str(calibration["sample_count"]),
            "EXPECTED_DATASET_INDEX_SHA256": str(
                calibration["dataset_index_sha256"]
            ),
            "DEVICE": args.device or str(build["device"]),
            "MARCH": str(build["march"]),
            "JOBS": str(build["jobs"]),
            "CHUNK_SIZE": str(language["chunk_size"]),
            "CACHE_LEN": str(language["cache_len"]),
            "DECODE_SEQ_LEN": str(language["pbd_query_len"]),
            "LM_HEAD_W_BITS": str(language["lm_head_w_bits"]),
            "LANGUAGE_GRAPH_SET": graph_set.name,
            "EXPORT_ONLY": "1" if args.target == "bc" else "0",
            "RESUME": "1" if args.resume else "0",
            "BUILD_TARGET": args.target,
            "WAIT": "1",
            "DETACH": "0",
            "LOG_DIR": str(log_root),
            "LOG_FILE": str(log_root / f"build_{component}_{args.target}.log"),
        })
        rotation = calibration.get("hidden_rotation_path")
        if rotation:
            env["HIDDEN_ROTATION_PATH"] = str(resolve_path_value(rotation))
        if component == "vision":
            env.update({
                "W_BITS": str(vision["w_bits"]),
                "VIT_CORE_NUM": str(cores["vision"]),
                "IMAGE_WIDTH": str(model["image_width"]),
                "IMAGE_HEIGHT": str(model["image_height"]),
            })
            script = BUILD_SCRIPTS / "vision.sh"
        else:
            env.update({
                "W_BITS": str(language["decoder_w_bits"]),
                "PREFILL_CORE_NUM": str(cores["prefill"]),
                "DECODE_CORE_NUM": str(cores["pbd"]),
                "AR_CORE_NUM": str(cores["ar"]),
            })
            script = BUILD_SCRIPTS / "language.sh"
        steps.append(PlanStep(f"build {component} through {args.target}", (bash, str(script)), env=env))
    return steps


def verify_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    calibration = _mapping(config["calibration"], "calibration")
    language = _mapping(config["language"], "language")
    model = _mapping(config["model"], "model")
    component = "full" if args.component == "all" else args.component
    graph_set = selected_graph_set(args, config)
    specification_command = [
        python_command(config),
        str(VALIDATE_SCRIPTS / "deployment.py"),
        "--component", component,
        "--selected-jsonl", str(resolve_path(config, "selected_jsonl")),
        "--generated-jsonl", str(resolve_path(config, "generated_jsonl")),
        "--scale-manifest", str(resolve_path(config, "scale_manifest")),
        "--coverage-json", str(resolve_path(config, "coverage_json")),
        "--model-path", str(resolve_path(config, "model")),
        "--expected-samples", str(calibration["sample_count"]),
        "--expected-dataset-index-sha256", str(calibration["dataset_index_sha256"]),
        "--image-width", str(model["image_width"]),
        "--image-height", str(model["image_height"]),
        "--chunk-size", str(language["chunk_size"]),
        "--cache-len", str(language["cache_len"]),
        "--decode-seq-len", str(language["pbd_query_len"]),
        "--lm-head-w-bits", str(language["lm_head_w_bits"]),
        "--graph-set", graph_set.name,
    ]
    rotation = calibration.get("hidden_rotation_path")
    if rotation:
        specification_command.extend(("--hidden-rotation-path", str(resolve_path_value(rotation))))

    verification = _mapping(config["verification"], "verification")
    verification_root = resolve_path(config, "verification_root")
    pipeline_command = (
        python_command(config),
        str(VALIDATE_SCRIPTS / "compare_pipeline.py"),
        "--mode", "analysis",
        "--output_dir", str(verification_root / "pipeline"),
        "--scale_manifest", str(resolve_path(config, "scale_manifest")),
        "--graph-set", graph_set.name,
    )
    predictions = (
        resolve_path_value(args.predictions_jsonl)
        if args.predictions_jsonl
        else resolve_evaluation_path(verification["predictions_jsonl"])
    )
    reference = (
        resolve_path_value(args.reference_jsonl)
        if args.reference_jsonl
        else resolve_evaluation_path(verification["reference_jsonl"])
    )
    task_command = (
        python_command(config),
        str(VALIDATE_SCRIPTS / "evaluate_grounding.py"),
        "--predictions-jsonl", str(predictions),
        "--reference-jsonl", str(reference),
        "--output-json", str(verification_root / "task_metrics.json"),
        "--details-jsonl", str(verification_root / "task_details.jsonl"),
        "--iou-threshold", str(verification["iou_threshold"]),
    )
    available = {
        "specification": PlanStep(
            "verify calibration and build specification",
            tuple(specification_command),
        ),
        "pipeline": PlanStep("summarize Float/BC/HBM comparisons", pipeline_command),
        "task": PlanStep("evaluate held-out grounding predictions", task_command),
    }
    stages = (
        ("specification", "pipeline", "task")
        if args.stage == "all"
        else (args.stage,)
    )
    return [available[stage] for stage in stages]


def quote_command(command: Iterable[str]) -> str:
    return shlex.join(str(part) for part in command)


def print_build_summary(config: Mapping[str, Any]) -> None:
    language = config["language"]
    graph_set = language_graph_set(str(language["graph_set"]))
    payload = {
        "image": f"{config['model']['image_width']}x{config['model']['image_height']}",
        "vision_w_bits": config["vision"]["w_bits"],
        "chunk_size": language["chunk_size"],
        "cache_len": language["cache_len"],
        "language_w_bits": language["decoder_w_bits"],
        "lm_head_w_bits": language["lm_head_w_bits"],
        "language_graph_set": graph_set.name,
        "language_graph_count": len(graph_set.graphs),
    }
    print("[build] " + json.dumps(payload, sort_keys=True))


def run_plan(steps: list[PlanStep], args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    if args.graph_set:
        config["language"]["graph_set"] = args.graph_set
    print_build_summary(config)
    if args.dry_run:
        for index, step in enumerate(steps, 1):
            print(f"[plan {index}/{len(steps)}] {step.label}")
            print(f"  cwd: {step.cwd}")
            if step.env:
                visible = " ".join(
                    f"{key}={value}" for key, value in sorted(step.env.items())
                )
                print(f"  env: {visible}")
            print(f"  command: {quote_command(step.command)}")
            if step.note:
                print(f"  note: {step.note}")
        print("[dry-run] no command executed")
        return 0

    run_started = time.monotonic()
    for index, step in enumerate(steps, 1):
        stage_started = time.monotonic()
        print(f"[stage {index}/{len(steps)}] START {step.label}", flush=True)
        if step.note:
            print(f"  {step.note}", flush=True)
        executable = step.command[0]
        if not Path(executable).is_file() and shutil.which(executable) is None:
            raise RuntimeError(f"required executable not found: {executable}")
        env = os.environ.copy()
        env.update(step.env)
        completed = subprocess.run(step.command, cwd=step.cwd, env=env, check=False)
        if completed.returncode:
            elapsed = time.monotonic() - stage_started
            print(
                f"[stage {index}/{len(steps)}] FAILED {step.label} "
                f"after {elapsed:.1f}s (exit={completed.returncode})",
                flush=True,
            )
            return int(completed.returncode)
        elapsed = time.monotonic() - stage_started
        print(
            f"[stage {index}/{len(steps)}] DONE {step.label} ({elapsed:.1f}s)",
            flush=True,
        )
    print(f"[done] completed in {time.monotonic() - run_started:.1f}s", flush=True)
    return 0


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph-set",
        dest="graph_set",
        choices=LANGUAGE_GRAPH_SET_NAMES,
        help="Language graph set; defaults to language.graph_set in config.yaml",
    )
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="auto")
    parser.add_argument("--resume", action="store_true", help="reuse complete compatible outputs")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LocateAnything prepare -> calibrate -> build -> verify orchestrator; "
            "the four commands consume a frozen calibration index"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="materialize Float calibration tensors")
    add_common_options(prepare)
    prepare.add_argument("--selected-jsonl")
    prepare.add_argument("--output-dir")
    prepare.add_argument("--upstream-source")
    prepare.add_argument("--model-path")
    prepare.add_argument("--device")
    prepare.add_argument("--dtype", choices=("bfloat16",))
    prepare.add_argument("--slow-samples", type=int)
    prepare.add_argument("--max-new-tokens", type=int)
    prepare.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate frozen Prepare inputs without loading model weights or using CUDA",
    )
    prepare.add_argument(
        "--report-json",
        help="preflight report path; defaults to OUTPUT_DIR/prepare_preflight.json",
    )

    calibrate = subparsers.add_parser("calibrate", help="collect and freeze activation scales")
    add_common_options(calibrate)
    calibrate.add_argument("--component", choices=COMPONENTS, default="all")
    calibrate.add_argument("--generated-jsonl")
    calibrate.add_argument("--selected-jsonl")
    calibrate.add_argument("--upstream-source")
    calibrate.add_argument("--output-dir")
    calibrate.add_argument("--model-path")
    calibrate.add_argument("--device")
    calibrate.add_argument("--dtype", choices=("float16",))
    calibrate.add_argument("--max-samples", type=int)
    calibrate.add_argument("--checkpoint-samples", type=int)

    build = subparsers.add_parser("build", help="export BC or build through HBO/HBM")
    add_common_options(build)
    build.add_argument("--component", choices=COMPONENTS, default="all")
    build.add_argument("--target", choices=BUILD_TARGETS, default="hbm")
    build.add_argument("--output-dir")
    build.add_argument("--model-path")
    build.add_argument("--device")

    verify = subparsers.add_parser(
        "verify", help="validate data, scales, and build specification"
    )
    add_common_options(verify)
    verify.add_argument("--component", choices=COMPONENTS, default="all")
    verify.add_argument(
        "--stage",
        dest="stage",
        choices=VERIFY_STAGES,
        default="specification",
        help="verification stage",
    )
    verify.add_argument(
        "--level",
        dest="stage",
        choices=VERIFY_STAGES,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    verify.add_argument("--predictions-jsonl")
    verify.add_argument("--reference-jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.resolve())
        if args.command == "prepare":
            steps = prepare_plan(args, config)
        elif args.command == "calibrate":
            steps = calibrate_plan(args, config)
        elif args.command == "build":
            steps = build_plan(args, config)
        else:
            steps = verify_plan(args, config)
        return run_plan(steps, args, config)
    except (ConfigurationError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
