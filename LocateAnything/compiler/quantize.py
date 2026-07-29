#!/usr/bin/env python3
"""Unified LocateAnything calibration, quantization, build, and verification CLI.

This file is deliberately an orchestration layer. The numerical calibration,
BC export, HBDK compilation, and validation algorithms remain in
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
VERIFY_LEVELS = ("contract", "pipeline", "task", "all")
EXPECTED_LANGUAGE_GRAPHS = (
    "prefill", "decode", "decode_ar",
    *(f"decode_pbd_q{q_len}" for q_len in range(7, 13)),
    *(f"decode_ar_q{q_len}" for q_len in range(2, 6)),
)
PATH_ENV_OVERRIDES = {
    "model": "LA_MODEL_PATH",
    "upstream_source": "LA_UPSTREAM_SOURCE",
}


class ConfigurationError(ValueError):
    """Raised when config.yaml violates the fixed LocateAnything contract."""


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
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    config = _mapping(raw, "config")
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
        "artifact_root", "log_root", "verification_root",
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
    if image_width % (patch_size * spatial_merge) or image_height % (patch_size * spatial_merge):
        raise ConfigurationError("image dimensions must align to patch_size * spatial_merge")
    if model.get("hidden_size") != 2048 or model.get("vocab_size") != 152681:
        raise ConfigurationError("hidden_size=2048 and vocab_size=152681 are fixed model contracts")

    calibration = _mapping(config.get("calibration"), "calibration")
    sample_count = _positive_int(calibration.get("sample_count"), "calibration.sample_count")
    checkpoint = _positive_int(
        calibration.get("checkpoint_samples"), "calibration.checkpoint_samples"
    )
    if checkpoint >= sample_count:
        raise ConfigurationError("checkpoint_samples must be smaller than sample_count")

    language = _mapping(config.get("language"), "language")
    chunk_size = _positive_int(language.get("chunk_size"), "language.chunk_size")
    cache_len = _positive_int(language.get("cache_len"), "language.cache_len")
    if chunk_size != 1024 or cache_len != 4096:
        raise ConfigurationError("release Language contract requires chunk_size=1024, cache_len=4096")
    if language.get("pbd_query_len") != 6 or language.get("ar_query_len") != 1:
        raise ConfigurationError("LocateAnything requires PBD q=6 and AR q=1")
    if language.get("decoder_w_bits") != 8 or language.get("lm_head_w_bits") != 8:
        raise ConfigurationError("release Language and LM Head weights must use W8")
    if language.get("fused_pbd") is not True:
        raise ConfigurationError("release profile requires fused_pbd=true")
    graphs = language.get("graphs")
    if not isinstance(graphs, list) or tuple(graphs) != EXPECTED_LANGUAGE_GRAPHS:
        raise ConfigurationError("fused Language profile must declare the canonical 13 graphs")

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


def resolve_path(config: Mapping[str, Any], key: str, override: str | None = None) -> Path:
    environment = PATH_ENV_OVERRIDES.get(key)
    raw = override
    if raw is None and environment:
        raw = os.environ.get(environment)
    if raw is None:
        raw = _mapping(config["paths"], "paths")[key]
    expanded = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def select_components(value: str) -> tuple[str, ...]:
    return ("vision", "language") if value == "all" else (value,)


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
    selected = resolve_path(config, "selected_jsonl", args.selected_jsonl)
    output_dir = resolve_path(config, "generated_dir", args.output_dir)
    build = _mapping(config["build"], "build")
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
    return [PlanStep("prepare calibration tensors", command, env=env)]


def calibrate_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    calibration = _mapping(config["calibration"], "calibration")
    language = _mapping(config["language"], "language")
    build = _mapping(config["build"], "build")
    env = common_env(config, args.progress)
    env.update({
        "GENERATED_JSONL": str(resolve_path(config, "generated_jsonl", args.generated_jsonl)),
        "MODEL_PATH": str(resolve_path(config, "model", args.model_path)),
        "OUTPUT_DIR": str(resolve_path(config, "calibration_dir", args.output_dir)),
        "DEVICE": args.device or str(calibration["device"]),
        "DTYPE": args.dtype or str(calibration["calibrate_dtype"]),
        "STAGE": "all" if args.component == "all" else args.component,
        "CHUNK_SIZE": str(language["chunk_size"]),
        "CACHE_LEN": str(language["cache_len"]),
        "LM_HEAD_W_BITS": str(language["lm_head_w_bits"]),
        "MAX_SAMPLES": str(args.max_samples or calibration["sample_count"]),
        "CHECKPOINT_SAMPLES": str(args.checkpoint_samples or calibration["checkpoint_samples"]),
        "IMAGE_TOKEN_ID": str(calibration["image_token_id"]),
        "REPLAY_SEED": str(calibration["seed"]),
        "RESUME": "1" if args.resume else "0",
    })
    rotation = calibration.get("hidden_rotation_path")
    if rotation:
        env["HIDDEN_ROTATION_PATH"] = str(resolve_path_value(rotation))
    note = None
    if args.resume:
        note = "calibration replay has no partial-resume contract; completed manifests are reused"
    command = (str(build.get("bash", "bash")), str(CALIBRATION_SCRIPTS / "calibrate.sh"))
    return [PlanStep("collect activation statistics", command, env=env, note=note)]


def resolve_path_value(value: Any) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> list[PlanStep]:
    calibration = _mapping(config["calibration"], "calibration")
    language = _mapping(config["language"], "language")
    vision = _mapping(config["vision"], "vision")
    build = _mapping(config["build"], "build")
    cores = _mapping(build["cores"], "build.cores")
    artifact_root = resolve_path(config, "artifact_root", args.output_dir)
    log_root = resolve_path(config, "log_root")
    bash = str(build.get("bash", "bash"))
    steps: list[PlanStep] = []
    for component in select_components(args.component):
        output = artifact_root / component
        env = common_env(config, args.progress)
        env.update({
            "INPUT_MODEL_PATH": str(resolve_path(config, "model", args.model_path)),
            "OUTPUT_MODEL_PATH": str(output),
            "CALIB_JSON": str(resolve_path(config, "selected_jsonl")),
            "GENERATED_JSON": str(resolve_path(config, "generated_jsonl")),
            "CALIBRATION_SCALE_MANIFEST": str(resolve_path(config, "scale_manifest")),
            "CALIBRATION_COVERAGE_JSON": str(resolve_path(config, "coverage_json")),
            "EXPECTED_SAMPLES": str(calibration["sample_count"]),
            "DEVICE": args.device or str(build["device"]),
            "MARCH": str(build["march"]),
            "JOBS": str(build["jobs"]),
            "CHUNK_SIZE": str(language["chunk_size"]),
            "CACHE_LEN": str(language["cache_len"]),
            "DECODE_SEQ_LEN": str(language["pbd_query_len"]),
            "LM_HEAD_W_BITS": str(language["lm_head_w_bits"]),
            "FUSED_PBD_PROFILES": "1" if language["fused_pbd"] else "0",
            "EXPORT_ONLY": "1" if args.target == "bc" else "0",
            "RESUME": "1" if args.resume else "0",
            "BUILD_TARGET": args.target,
            "WAIT": "1",
            "DETACH": "0",
            "LOG_DIR": str(log_root),
            "LOG_FILE": str(log_root / f"build_{component}_{args.target}.log"),
        })
        if component == "vision":
            env.update({"W_BITS": str(vision["w_bits"]), "VIT_CORE_NUM": str(cores["vision"])})
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
    contract_command = [
        python_command(config),
        str(VALIDATE_SCRIPTS / "deployment.py"),
        "--component", component,
        "--selected-jsonl", str(resolve_path(config, "selected_jsonl")),
        "--generated-jsonl", str(resolve_path(config, "generated_jsonl")),
        "--scale-manifest", str(resolve_path(config, "scale_manifest")),
        "--coverage-json", str(resolve_path(config, "coverage_json")),
        "--expected-samples", str(calibration["sample_count"]),
        "--image-width", str(model["image_width"]),
        "--image-height", str(model["image_height"]),
        "--chunk-size", str(language["chunk_size"]),
        "--cache-len", str(language["cache_len"]),
        "--decode-seq-len", str(language["pbd_query_len"]),
        "--lm-head-w-bits", str(language["lm_head_w_bits"]),
    ]
    rotation = calibration.get("hidden_rotation_path")
    if rotation:
        contract_command.extend(("--hidden-rotation-path", str(resolve_path_value(rotation))))

    verification = _mapping(config["verification"], "verification")
    verification_root = resolve_path(config, "verification_root")
    pipeline_command = (
        python_command(config),
        str(VALIDATE_SCRIPTS / "compare_pipeline.py"),
        "--mode", "analysis",
        "--output_dir", str(verification_root / "pipeline"),
        "--scale_manifest", str(resolve_path(config, "scale_manifest")),
    )
    predictions = args.predictions_jsonl or str(verification["predictions_jsonl"])
    reference = args.reference_jsonl or str(verification["reference_jsonl"])
    task_command = (
        python_command(config),
        str(VALIDATE_SCRIPTS / "evaluate_grounding.py"),
        "--predictions-jsonl", str(resolve_path_value(predictions)),
        "--reference-jsonl", str(resolve_path_value(reference)),
        "--output-json", str(verification_root / "task_metrics.json"),
        "--details-jsonl", str(verification_root / "task_details.jsonl"),
        "--iou-threshold", str(verification["iou_threshold"]),
    )
    available = {
        "contract": PlanStep("verify calibration and compile contract", tuple(contract_command)),
        "pipeline": PlanStep("summarize Float/BC/HBM comparisons", pipeline_command),
        "task": PlanStep("evaluate held-out grounding predictions", task_command),
    }
    levels = ("contract", "pipeline", "task") if args.level == "all" else (args.level,)
    return [available[level] for level in levels]


def quote_command(command: Iterable[str]) -> str:
    return shlex.join(str(part) for part in command)


def print_contract(config: Mapping[str, Any]) -> None:
    language = config["language"]
    payload = {
        "image": f"{config['model']['image_width']}x{config['model']['image_height']}",
        "vision_w_bits": config["vision"]["w_bits"],
        "chunk_size": language["chunk_size"],
        "cache_len": language["cache_len"],
        "language_w_bits": language["decoder_w_bits"],
        "lm_head_w_bits": language["lm_head_w_bits"],
        "fused_pbd": language["fused_pbd"],
        "language_graphs": len(language["graphs"]),
    }
    print("[contract] " + json.dumps(payload, sort_keys=True))


def reusable_calibration_exists(config: Mapping[str, Any], output_override: str | None) -> bool:
    if output_override:
        output = resolve_path(config, "calibration_dir", output_override)
        return all(
            (output / name).is_file()
            for name in ("calibration_scale_manifest.json", "calibration_graph_coverage.json")
        )
    return all(resolve_path(config, key).is_file() for key in ("scale_manifest", "coverage_json"))


def run_plan(steps: list[PlanStep], args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    print_contract(config)
    for index, step in enumerate(steps, 1):
        print(f"[plan {index}/{len(steps)}] {step.label}")
        print(f"  cwd: {step.cwd}")
        if step.env:
            visible = " ".join(f"{key}={value}" for key, value in sorted(step.env.items()))
            print(f"  env: {visible}")
        print(f"  command: {quote_command(step.command)}")
        if step.note:
            print(f"  note: {step.note}")

    if args.dry_run:
        print("[dry-run] no command executed")
        return 0

    if (
        args.command == "calibrate"
        and args.resume
        and reusable_calibration_exists(config, args.output_dir)
    ):
        print("[resume] calibration manifests already exist; replay skipped")
        return 0

    for step in steps:
        executable = step.command[0]
        if not Path(executable).is_file() and shutil.which(executable) is None:
            raise RuntimeError(f"required executable not found: {executable}")
        env = os.environ.copy()
        env.update(step.env)
        completed = subprocess.run(step.command, cwd=step.cwd, env=env, check=False)
        if completed.returncode:
            return int(completed.returncode)
    return 0


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="auto")
    parser.add_argument("--resume", action="store_true", help="reuse complete compatible outputs")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LocateAnything calibration and S600 build orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="materialize original Float replay tensors")
    add_common_options(prepare)
    prepare.add_argument("--selected-jsonl")
    prepare.add_argument("--output-dir")
    prepare.add_argument("--upstream-source")
    prepare.add_argument("--model-path")
    prepare.add_argument("--device")
    prepare.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    prepare.add_argument("--slow-samples", type=int)
    prepare.add_argument("--max-new-tokens", type=int)

    calibrate = subparsers.add_parser("calibrate", help="collect and freeze activation scales")
    add_common_options(calibrate)
    calibrate.add_argument("--component", choices=COMPONENTS, default="all")
    calibrate.add_argument("--generated-jsonl")
    calibrate.add_argument("--output-dir")
    calibrate.add_argument("--model-path")
    calibrate.add_argument("--device")
    calibrate.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    calibrate.add_argument("--max-samples", type=int)
    calibrate.add_argument("--checkpoint-samples", type=int)

    build = subparsers.add_parser("build", help="export BC or build through HBO/HBM")
    add_common_options(build)
    build.add_argument("--component", choices=COMPONENTS, default="all")
    build.add_argument("--target", choices=BUILD_TARGETS, default="hbm")
    build.add_argument("--output-dir")
    build.add_argument("--model-path")
    build.add_argument("--device")

    verify = subparsers.add_parser("verify", help="validate data, scales, and build contract")
    add_common_options(verify)
    verify.add_argument("--component", choices=COMPONENTS, default="all")
    verify.add_argument("--level", choices=VERIFY_LEVELS, default="contract")
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
