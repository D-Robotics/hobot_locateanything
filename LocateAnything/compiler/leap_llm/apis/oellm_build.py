"""Compile the model targets maintained by this repository."""

import argparse
import os
from pathlib import Path

import torch

from leap_llm.apis.model.model_factory import (
    create_model_api,
    get_marches_with_model,
    get_supported_marches,
    get_supported_models,
)


DEFAULT_COMPILE_KWARGS = {
    "march": "nash-p",
    "jobs": 16,
    "progress_bar": True,
    "max_time_per_fc": 0.0,
    "opt": 2,
    "debug": False,
    "advice": 0.0,
    "balance": 100,
    "input_no_padding": False,
    "output_no_padding": False,
}


def validated_path(check_exists=True):
    def validator(path_string):
        if not path_string:
            raise argparse.ArgumentTypeError("Path cannot be empty")
        path = Path(os.path.expanduser(os.path.expandvars(path_string)))
        if check_exists and not path.exists():
            raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
        return str(path.resolve())

    return validator


def validate_device(value: str) -> list[str]:
    raw_devices = [item for item in value.replace(",", " ").split() if item]
    if not raw_devices:
        raise argparse.ArgumentTypeError("Device cannot be empty")
    if "cpu" in {item.lower() for item in raw_devices}:
        if len(raw_devices) != 1:
            raise argparse.ArgumentTypeError("CPU cannot be combined with CUDA")
        return ["cpu"]

    devices = []
    for raw in raw_devices:
        value_lower = raw.lower()
        if value_lower == "cuda":
            value_lower = "cuda:0"
        if not value_lower.startswith("cuda:"):
            raise argparse.ArgumentTypeError(f"Unsupported device: {raw}")
        try:
            index = int(value_lower.split(":", 1)[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid CUDA device: {raw}") from exc
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise argparse.ArgumentTypeError(f"CUDA device is unavailable: {raw}")
        devices.append(f"cuda:{index}")
    return list(dict.fromkeys(devices))


def parse_core_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Core values must be integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one core value is required")
    return values


def validate_image_size(height: int, width: int) -> None:
    patch_merge = 28
    if height % patch_merge or width % patch_merge:
        raise ValueError(f"Qwen2.5-VL image dimensions must be multiples of {patch_merge}")
    pixels = height * width
    if not 200704 <= pixels <= 399840:
        raise ValueError(
            f"Qwen2.5-VL image pixel count must be in [200704, 399840], got {pixels}"
        )


def build_parser() -> argparse.ArgumentParser:
    model_help = ", ".join(
        f"{name} ({'/'.join(get_marches_with_model(name))})"
        for name in get_supported_models()
    )
    parser = argparse.ArgumentParser(
        description=f"Compile an S600 model. Targets: {model_help}"
    )
    parser.add_argument("--model_name", required=True, choices=get_supported_models())
    parser.add_argument("--march", required=True, choices=get_supported_marches())
    parser.add_argument(
        "--input_model_path", required=True, type=validated_path(check_exists=True)
    )
    parser.add_argument(
        "--output_model_path", required=True, type=validated_path(check_exists=False)
    )
    parser.add_argument("--cache_len", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument(
        "--decode_seq_len",
        type=int,
        default=1,
        help="Decode query length; use 6 for LocateAnything PBD and 1 for AR.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_width", type=int, default=448)
    parser.add_argument("--image_height", type=int, default=448)
    parser.add_argument("--device", type=validate_device, default=["cpu"])
    parser.add_argument("--calib_json_path", type=validated_path(), default=None)
    parser.add_argument("--calib_tsv_path", type=validated_path(), default=None)
    parser.add_argument("--w_bits", type=int, choices=[4, 8], default=8)
    parser.add_argument("--lm_head_w_bits", type=int, choices=[4, 8], default=8)
    parser.add_argument(
        "--input_model_format",
        choices=["hf", "llmc", "github"],
        default="hf",
    )
    parser.add_argument("--vit_core_num", type=parse_core_list, default=[1])
    parser.add_argument("--prefill_core_num", type=parse_core_list, default=[1])
    parser.add_argument("--decode_core_num", type=parse_core_list, default=[1])
    parser.add_argument("--ar_core_num", type=parse_core_list, default=None)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--cache_path", type=validated_path(False), default=None)
    parser.add_argument(
        "--calibration_scale_manifest", type=validated_path(), default=None
    )
    parser.add_argument("--hidden_rotation_path", type=validated_path(), default=None)
    parser.add_argument(
        "--fused_pbd_profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the complete LocateAnything fused-PBD graph family (default: enabled).",
    )
    parser.add_argument("--disable_hidden_rotation", action="store_true")
    parser.add_argument("--export_only", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if not 256 <= args.cache_len <= 4096:
        parser.error("--cache_len must be in [256, 4096]")
    if not 128 <= args.chunk_size <= 2048:
        parser.error("--chunk_size must be in [128, 2048]")
    if args.cache_len <= args.chunk_size:
        parser.error("--cache_len must be greater than --chunk_size")
    if args.cache_len % 64 or args.chunk_size % 64:
        parser.error("--cache_len and --chunk_size must be multiples of 64")
    if not 1 <= args.decode_seq_len <= args.cache_len:
        parser.error("--decode_seq_len must be in [1, cache_len]")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.batch_size != 1 and args.model_name != "qwen2_5-vl-3b":
        parser.error("batch_size > 1 is only supported by qwen2_5-vl-3b")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    def require_single_core(name):
        values = getattr(args, name)
        if len(values) != 1 or values[0] not in {1, 2, 4}:
            parser.error(f"--{name} must be one of 1, 2, or 4")

    if args.model_name == "qwen2_5-vl-3b":
        try:
            validate_image_size(args.image_height, args.image_width)
        except ValueError as exc:
            parser.error(str(exc))
        require_single_core("vit_core_num")
        require_single_core("prefill_core_num")
        require_single_core("decode_core_num")
        if args.prefill_core_num != args.decode_core_num:
            parser.error("Qwen2.5-VL prefill and decode core counts must match")
    elif args.model_name == "locateanything-vit-3b":
        require_single_core("vit_core_num")
        if args.calibration_scale_manifest is None:
            parser.error(
                "LocateAnything Vision release requires --calibration_scale_manifest"
            )
        if (args.image_width, args.image_height) != (672, 672):
            parser.error(
                "LocateAnything Vision release requires "
                "--image_width 672 --image_height 672"
            )
        if args.w_bits != 8:
            parser.error("LocateAnything Vision release requires --w_bits 8")
    elif args.model_name == "locateanything-lm-3b":
        require_single_core("prefill_core_num")
        require_single_core("decode_core_num")
        if args.calibration_scale_manifest is None:
            parser.error(
                "LocateAnything Language release requires --calibration_scale_manifest"
            )
        if args.ar_core_num is None:
            args.ar_core_num = list(args.decode_core_num)
        require_single_core("ar_core_num")
        release = (args.chunk_size, args.cache_len, args.decode_seq_len)
        if release != (1024, 4096, 6):
            parser.error(
                "LocateAnything Language release requires "
                "--chunk_size 1024 --cache_len 4096 --decode_seq_len 6"
            )
        if args.w_bits != 8 or args.lm_head_w_bits != 8:
            parser.error(
                "LocateAnything Language release requires "
                "--w_bits 8 --lm_head_w_bits 8"
            )
        if not args.fused_pbd_profiles:
            parser.error(
                "LocateAnything Language release requires --fused_pbd_profiles"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    compile_kwargs = dict(DEFAULT_COMPILE_KWARGS)
    compile_kwargs["march"] = args.march
    compile_kwargs["jobs"] = args.jobs
    if args.cache_path:
        Path(args.cache_path).mkdir(parents=True, exist_ok=True)
        compile_kwargs.update(cache_mode="enable", cache_path=args.cache_path)
    if args.model_name == "qwen2_5-vl-3b":
        compile_kwargs.update(input_no_padding=True, output_no_padding=True)

    model = create_model_api(args.model_name, args)
    if model is None:
        parser.error(f"Unsupported model: {args.model_name}")

    vit_kwargs = dict(compile_kwargs)
    llm_kwargs = dict(compile_kwargs, enable_hpc=True)
    model.compile(vit_kwargs=vit_kwargs, llm_kwargs=llm_kwargs)


if __name__ == "__main__":
    main()
