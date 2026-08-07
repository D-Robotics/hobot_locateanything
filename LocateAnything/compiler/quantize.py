#!/usr/bin/env python3
"""Orchestrate LocateAnything preparation, calibration, and build stages.

The numerical calibration, BC export, HBDK compilation, and model algorithms
live in ``compiler/pipeline`` and ``compiler/model``.
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPILER_ROOT = PROJECT_ROOT / "compiler"
PIPELINE_ROOT = COMPILER_ROOT / "pipeline"
DEFAULT_CONFIG = COMPILER_ROOT / "config" / "quantization.yaml"
CONFIG_DIR_KEY = "__config_dir__"
COMPONENTS = ("vision", "language", "all")
BUILD_TARGETS = ("bc", "hbm")
PROGRESS_MODES = ("auto", "bar", "log", "off")
if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

from configuration import ConfigurationFileError, load_config_file  # noqa: E402
from model.graphs import language_graph_set  # noqa: E402
from model.contract import (  # noqa: E402
    HIDDEN_SIZE,
    IMAGE_HEIGHT,
    IMAGE_TOKEN_ID,
    IMAGE_WIDTH,
    LETTERBOX_FILL,
    PATCH_SIZE,
    PBD_QUERY_LEN,
    RESIZE_MODE,
    SPATIAL_MERGE,
)
CONVERGENCE_CHECKPOINTS = (64, 128, 256, 512)


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


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        config = load_config_file(path)
    except ConfigurationFileError as exc:
        raise ConfigurationError(str(exc)) from exc
    validate_config(config)
    config[CONFIG_DIR_KEY] = path.parent
    return config


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    required_sections = {"paths", "calibration", "language", "quantization", "build"}
    missing_sections = sorted(required_sections - config.keys())
    extra_sections = sorted(set(config) - required_sections)
    if missing_sections:
        raise ConfigurationError(f"config is missing: {', '.join(missing_sections)}")
    if extra_sections:
        raise ConfigurationError(f"config contains unknown sections: {', '.join(extra_sections)}")

    paths = _mapping(config.get("paths"), "paths")
    required_paths = {
        "checkpoint", "locateanything_source", "calibration_data", "output_dir"
    }
    missing_paths = sorted(required_paths - set(paths))
    if missing_paths:
        raise ConfigurationError(f"paths is missing: {', '.join(missing_paths)}")
    extra_paths = sorted(set(paths) - required_paths)
    if extra_paths:
        raise ConfigurationError(f"paths contains unknown fields: {', '.join(extra_paths)}")

    calibration = _mapping(config.get("calibration"), "calibration")
    required_calibration = {
        "slow_samples", "max_new_tokens", "seed", "prepare_dtype", "statistics_dtype",
        "detailed_statistics",
    }
    if set(calibration) != required_calibration:
        missing = sorted(required_calibration - set(calibration))
        extra = sorted(set(calibration) - required_calibration)
        details = [*(f"missing {name}" for name in missing), *(f"unknown {name}" for name in extra)]
        raise ConfigurationError("invalid calibration fields: " + ", ".join(details))
    _positive_int(calibration.get("slow_samples"), "calibration.slow_samples")
    _positive_int(calibration.get("max_new_tokens"), "calibration.max_new_tokens")
    if type(calibration.get("seed")) is not int:
        raise ConfigurationError("calibration.seed must be an integer")
    if calibration.get("prepare_dtype") not in {"bfloat16", "float16"}:
        raise ConfigurationError("calibration.prepare_dtype must be bfloat16 or float16")
    if calibration.get("statistics_dtype") not in {"float16", "bfloat16"}:
        raise ConfigurationError("calibration.statistics_dtype must be float16 or bfloat16")
    if type(calibration.get("detailed_statistics")) is not bool:
        raise ConfigurationError("calibration.detailed_statistics must be true or false")

    language = _mapping(config.get("language"), "language")
    required_language = {"chunk_size", "cache_len"}
    allowed_language = required_language | {
        "sampling_backend", "sampling_temperature", "sampling_top_p",
        "sampling_repetition_penalty",
    }
    if not required_language.issubset(language) or not set(language).issubset(allowed_language):
        missing = sorted(required_language - set(language))
        extra = sorted(set(language) - allowed_language)
        details = [*(f"missing {name}" for name in missing), *(f"unknown {name}" for name in extra)]
        raise ConfigurationError("invalid language fields: " + ", ".join(details))
    language.setdefault("sampling_backend", "bpu")
    language.setdefault("sampling_temperature", 0.7)
    language.setdefault("sampling_top_p", 0.9)
    language.setdefault("sampling_repetition_penalty", 1.1)
    chunk_size = _positive_int(language.get("chunk_size"), "language.chunk_size")
    cache_len = _positive_int(language.get("cache_len"), "language.cache_len")
    if not 128 <= chunk_size <= 2048 or not 256 <= cache_len <= 4096:
        raise ConfigurationError(
            "language.chunk_size must be in [128, 2048] and cache_len in [256, 4096]"
        )
    if chunk_size % 64 or cache_len % 64 or cache_len <= chunk_size:
        raise ConfigurationError(
            "language.chunk_size and language.cache_len must be multiples of 64, "
            "with cache_len greater than chunk_size"
        )
    if language.get("sampling_backend") not in {"host", "bpu"}:
        raise ConfigurationError("language.sampling_backend must be host or bpu")
    for name in ("sampling_temperature", "sampling_top_p", "sampling_repetition_penalty"):
        value = language.get(name)
        if not isinstance(value, (int, float)) or not value > 0:
            raise ConfigurationError(f"language.{name} must be positive")
    if language["sampling_top_p"] > 1:
        raise ConfigurationError("language.sampling_top_p must be <= 1")

    quantization = _mapping(config.get("quantization"), "quantization")
    required_quantization = {
        "vision_weight_bits", "language_weight_bits", "lm_head_weight_bits"
    }
    if set(quantization) != required_quantization:
        missing = sorted(required_quantization - set(quantization))
        extra = sorted(set(quantization) - required_quantization)
        details = [*(f"missing {name}" for name in missing), *(f"unknown {name}" for name in extra)]
        raise ConfigurationError("invalid quantization fields: " + ", ".join(details))
    if quantization.get("vision_weight_bits") not in {8}:
        raise ConfigurationError("quantization.vision_weight_bits currently supports 8")
    for name in ("language_weight_bits", "lm_head_weight_bits"):
        if quantization.get(name) not in {4, 8}:
            raise ConfigurationError(f"quantization.{name} must be 4 or 8")

    build = _mapping(config.get("build"), "build")
    required_build = {"march", "device", "jobs", "cores"}
    if set(build) != required_build:
        missing = sorted(required_build - set(build))
        extra = sorted(set(build) - required_build)
        details = [*(f"missing {name}" for name in missing), *(f"unknown {name}" for name in extra)]
        raise ConfigurationError("invalid build fields: " + ", ".join(details))
    if not str(build.get("device") or "").strip():
        raise ConfigurationError("build.device must not be empty")
    if not str(build.get("march") or "").strip():
        raise ConfigurationError("build.march must not be empty")
    _positive_int(build.get("jobs"), "build.jobs")
    cores = _mapping(build.get("cores"), "build.cores")
    for name in ("vision", "prefill", "pbd", "ar"):
        if cores.get(name) not in {1, 2, 4}:
            raise ConfigurationError(f"build.cores.{name} must be 1, 2, or 4")

def _resolve_config_path(config_dir: Path, raw: Any) -> Path:
    expanded = Path(os.path.expanduser(str(raw)))
    if expanded.is_absolute():
        return expanded.resolve()
    return (config_dir / expanded).resolve()


def resolve_path(config: Mapping[str, Any], key: str) -> Path:
    config_dir = Path(config[CONFIG_DIR_KEY])
    paths = _mapping(config["paths"], "paths")
    checkpoint = _resolve_config_path(config_dir, paths["checkpoint"])
    source_dir = _resolve_config_path(config_dir, paths["locateanything_source"])
    calibration_data = _resolve_config_path(config_dir, paths["calibration_data"])
    output_dir = _resolve_config_path(config_dir, paths["output_dir"])
    resolved = {
        "model": checkpoint,
        "source_dir": source_dir,
        "selected_jsonl": calibration_data / "selected.jsonl",
        "generated_dir": output_dir / "calibration" / "generated",
        "generated_jsonl": output_dir / "calibration" / "generated" / "generated.jsonl",
        "calibration_dir": output_dir / "calibration" / "statistics",
        "scale_manifest": output_dir / "calibration" / "statistics" / "calibration_scale_manifest.json",
        "coverage_json": output_dir / "calibration" / "statistics" / "calibration_graph_coverage.json",
        "build_root": output_dir / "build",
        "log_root": output_dir / "logs",
    }
    return resolved[key].resolve()


def jsonl_record_count(path: Path) -> int:
    try:
        count = sum(
            bool(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except OSError as exc:
        raise ConfigurationError(f"cannot read calibration data {path}: {exc}") from exc
    if count == 0:
        raise ConfigurationError(f"calibration data is empty: {path}")
    return count


def calibration_sample_count(
    config: Mapping[str, Any], manifest: Path, override: int | None = None
) -> int:
    if override is not None:
        return _positive_int(override, "--max-samples")
    del config
    return jsonl_record_count(manifest)


def calibration_checkpoint(
    config: Mapping[str, Any], sample_count: int, override: int | None = None
) -> int:
    del config
    configured = _positive_int(override, "--checkpoint-samples") if override is not None else None
    if sample_count < 2:
        raise ConfigurationError("activation calibration requires at least two samples")
    if configured is None:
        eligible = [value for value in CONVERGENCE_CHECKPOINTS if value < sample_count]
        configured = max(eligible, default=max(1, sample_count // 2))
    if configured >= sample_count:
        raise ConfigurationError("checkpoint samples must be smaller than sample count")
    return configured


def select_components(value: str) -> tuple[str, ...]:
    return ("vision", "language") if value == "all" else (value,)


def python_command(config: Mapping[str, Any]) -> str:
    del config
    return sys.executable


def bash_command() -> str:
    return shutil.which("bash") or "bash"


def common_env(config: Mapping[str, Any], progress: str) -> dict[str, str]:
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
    return env


def prepare_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    calibration = _mapping(config["calibration"], "calibration")
    language = _mapping(config["language"], "language")
    build = _mapping(config["build"], "build")
    selected = resolve_path(config, "selected_jsonl")
    output_dir = resolve_path(config, "generated_dir")
    env = common_env(config, args.progress)
    env.update({
        "SELECTED_JSONL": str(selected),
        "OUTPUT_DIR": str(output_dir),
        "LOCATEANYTHING_SOURCE": str(resolve_path(config, "source_dir")),
        "MODEL_PATH": str(resolve_path(config, "model")),
        "DEVICE": str(build["device"]),
        "DTYPE": str(calibration["prepare_dtype"]),
        "IMAGE_WIDTH": str(IMAGE_WIDTH),
        "IMAGE_HEIGHT": str(IMAGE_HEIGHT),
        "RESIZE_MODE": RESIZE_MODE,
        "LETTERBOX_FILL": str(LETTERBOX_FILL),
        "PATCH_SIZE": str(PATCH_SIZE),
        "MERGE_SIZE": str(SPATIAL_MERGE),
        "HIDDEN_SIZE": str(HIDDEN_SIZE),
        "PREFILL_LIMIT": str(language["chunk_size"]),
        "MAX_NEW_TOKENS": str(calibration["max_new_tokens"]),
        "SLOW_SAMPLES": str(calibration["slow_samples"]),
        "SEED": str(calibration["seed"]),
        "RESUME": "1" if args.resume else "0",
    })
    command = (bash_command(), str(PIPELINE_ROOT / "run_prepare.sh"))
    return [PlanStep("prepare calibration tensors", command, env=env)]


def calibrate_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    calibration = _mapping(config["calibration"], "calibration")
    language = _mapping(config["language"], "language")
    quantization = _mapping(config["quantization"], "quantization")
    build = _mapping(config["build"], "build")
    generated_jsonl = resolve_path(config, "generated_jsonl")
    requested_samples = calibration_sample_count(config, generated_jsonl, args.max_samples)
    requested_checkpoint = calibration_checkpoint(
        config, requested_samples, args.checkpoint_samples
    )
    env = common_env(config, args.progress)
    env.update({
        "GENERATED_JSONL": str(generated_jsonl),
        "SELECTED_JSONL": str(resolve_path(config, "selected_jsonl")),
        "LOCATEANYTHING_SOURCE": str(resolve_path(config, "source_dir")),
        "MODEL_PATH": str(resolve_path(config, "model")),
        "OUTPUT_DIR": str(resolve_path(config, "calibration_dir")),
        "DEVICE": str(build["device"]),
        "DTYPE": str(calibration["statistics_dtype"]),
        "CALIBRATION_COMPONENT": "all" if args.component == "all" else args.component,
        "CHUNK_SIZE": str(language["chunk_size"]),
        "CACHE_LEN": str(language["cache_len"]),
        "SAMPLING_BACKEND": str(language["sampling_backend"]),
        "SAMPLING_TEMPERATURE": str(language["sampling_temperature"]),
        "SAMPLING_TOP_P": str(language["sampling_top_p"]),
        "SAMPLING_REPETITION_PENALTY": str(language["sampling_repetition_penalty"]),
        "VISION_W_BITS": str(quantization["vision_weight_bits"]),
        "LANGUAGE_W_BITS": str(quantization["language_weight_bits"]),
        "LM_HEAD_W_BITS": str(quantization["lm_head_weight_bits"]),
        "DETAILED_STATISTICS": "1" if calibration["detailed_statistics"] else "0",
        "MAX_SAMPLES": str(requested_samples),
        "CHECKPOINT_SAMPLES": str(requested_checkpoint),
        "IMAGE_TOKEN_ID": str(IMAGE_TOKEN_ID),
        "REPLAY_SEED": str(calibration["seed"]),
        "RESUME": "1" if args.resume else "0",
    })
    note = None
    if args.resume:
        note = "calibration replay is atomic; only completed statistics are reused"
    command = (bash_command(), str(PIPELINE_ROOT / "run_calibrate.sh"))
    return [PlanStep("collect activation statistics", command, env=env, note=note)]


def build_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    build = _mapping(config["build"], "build")
    language = _mapping(config["language"], "language")
    quantization = _mapping(config["quantization"], "quantization")
    cores = _mapping(build["cores"], "build.cores")
    build_root = resolve_path(config, "build_root")
    log_root = resolve_path(config, "log_root")
    bash = bash_command()
    steps: list[PlanStep] = []
    for component in select_components(args.component):
        output = build_root / component
        env = common_env(config, args.progress)
        env.update({
            "INPUT_MODEL_PATH": str(resolve_path(config, "model")),
            "OUTPUT_MODEL_PATH": str(output),
            "CALIBRATION_SCALE_MANIFEST": str(resolve_path(config, "scale_manifest")),
            "DEVICE": str(build["device"]),
            "MARCH": str(build["march"]),
            "JOBS": str(build["jobs"]),
            "CHUNK_SIZE": str(language["chunk_size"]),
            "CACHE_LEN": str(language["cache_len"]),
            "SAMPLING_BACKEND": str(language["sampling_backend"]),
            "SAMPLING_TEMPERATURE": str(language["sampling_temperature"]),
            "SAMPLING_TOP_P": str(language["sampling_top_p"]),
            "SAMPLING_REPETITION_PENALTY": str(language["sampling_repetition_penalty"]),
            "DECODE_SEQ_LEN": str(PBD_QUERY_LEN),
            "LM_HEAD_W_BITS": str(quantization["lm_head_weight_bits"]),
            "EXPORT_ONLY": "1" if args.target == "bc" else "0",
            "RESUME": "1" if args.resume else "0",
            "BUILD_TARGET": args.target,
            "WAIT": "1",
            "DETACH": "0",
            "LOG_DIR": str(log_root),
            "LOG_FILE": str(log_root / f"build_{component}_{args.target}.log"),
        })
        if component == "vision":
            env.update({
                "W_BITS": str(quantization["vision_weight_bits"]),
                "VIT_CORE_NUM": str(cores["vision"]),
                "IMAGE_WIDTH": str(IMAGE_WIDTH),
                "IMAGE_HEIGHT": str(IMAGE_HEIGHT),
            })
            script = PIPELINE_ROOT / "build_vision.sh"
        else:
            env.update({
                "W_BITS": str(quantization["language_weight_bits"]),
                "PREFILL_CORE_NUM": str(cores["prefill"]),
                "DECODE_CORE_NUM": str(cores["pbd"]),
                "AR_CORE_NUM": str(cores["ar"]),
            })
            script = PIPELINE_ROOT / "build_language.sh"
        steps.append(PlanStep(f"build {component} through {args.target}", (bash, str(script)), env=env))
    return steps


def quote_command(command: Iterable[str]) -> str:
    return shlex.join(str(part) for part in command)


def print_build_summary(config: Mapping[str, Any]) -> None:
    language = _mapping(config["language"], "language")
    quantization = _mapping(config["quantization"], "quantization")
    graph_set = language_graph_set()
    payload = {
        "image": f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        "vision_w_bits": quantization["vision_weight_bits"],
        "chunk_size": language["chunk_size"],
        "cache_len": language["cache_len"],
        "language_w_bits": quantization["language_weight_bits"],
        "lm_head_w_bits": quantization["lm_head_weight_bits"],
        "language_graph_set": graph_set.name,
        "language_graph_count": len(graph_set.graphs),
        "sampling_backend": language["sampling_backend"],
    }
    print("[build] " + json.dumps(payload, sort_keys=True))


def run_plan(steps: list[PlanStep], args: argparse.Namespace, config: Mapping[str, Any]) -> int:
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
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="auto")
    parser.add_argument("--resume", action="store_true", help="reuse complete compatible outputs")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LocateAnything prepare -> calibrate -> build orchestrator"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="materialize Float calibration tensors")
    add_common_options(prepare)
    calibrate = subparsers.add_parser("calibrate", help="collect activation scales")
    add_common_options(calibrate)
    calibrate.add_argument("--component", choices=COMPONENTS, default="all")
    calibrate.add_argument("--max-samples", type=int)
    calibrate.add_argument("--checkpoint-samples", type=int)

    build = subparsers.add_parser("build", help="export BC or build through HBO/HBM")
    add_common_options(build)
    build.add_argument("--component", choices=COMPONENTS, default="all")
    build.add_argument("--target", choices=BUILD_TARGETS, default="hbm")

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
            parser.error(f"unknown command: {args.command}")
        return run_plan(steps, args, config)
    except (ConfigurationError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
