import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append("../../")

os.environ["DEV_B30_TRITON_VPU"] = "1"
os.environ["DEV_B30_ENABLE_VPU_EXTRA_OP"] = "1"
os.environ["DEV_B30_ENABLE_VPU_TRIAL_OP"] = "1"

from leap_llm.apis.model.model_factory import (  # noqa: E402
    create_model_api,
    get_marches_with_model,
    get_supported_marches,
    get_supported_models,
)

DEFAULT_COMPILE_KWARGS = {
    "march": "nash-m",
    "jobs": 32,
    "progress_bar": True,
    "max_time_per_fc": 0.0,
    "opt": 2,
    "debug": False,
    "advice": 0.0,
    "balance": 100,
    "input_no_padding": False,
    "output_no_padding": False,
}

LLM_COMPILE_KWARGS = {
    "enable_hpc": True,
}

VIT_COMPILE_KWARGS = {}


def validated_path(check_exists=True):
    def validator(path_string):
        if not path_string:
            raise argparse.ArgumentTypeError("Path cannot be empty")

        path = Path(os.path.expanduser(os.path.expandvars(path_string)))

        if check_exists and not path.exists():
            raise argparse.ArgumentTypeError(f"Path does not exist: {path}")

        return str(path.resolve())

    return validator


def validate_device(s: str) -> list[str]:
    """
    Validate device string, support single or multiple devices.
    Examples:
        - "cpu" -> ["cpu"]
        - "cuda:0" -> ["cuda:0"]
        - "cuda:1 cuda:2" -> ["cuda:1", "cuda:2"]
        - "cuda:1,cuda:2" -> ["cuda:1", "cuda:2"]
    """
    devices = []
    # Split by comma or space
    device_strs = [d.strip() for d in s.replace(",", " ").split() if d.strip()]

    if not device_strs:
        raise argparse.ArgumentTypeError("Device string cannot be empty")

    for device_str in device_strs:
        device_str_lower = device_str.lower()
        if device_str_lower == "cpu":
            if len(device_strs) > 1:
                raise argparse.ArgumentTypeError(
                    "CPU cannot be used with other devices"
                )
            return ["cpu"]
        elif device_str_lower == "cuda":
            devices.append("cuda:0")
        elif device_str_lower.startswith("cuda:"):
            if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
                raise argparse.ArgumentTypeError("CUDA is not available")
            parts = device_str_lower.split(":", 1)
            if len(parts) != 2 or parts[1] == "":
                raise argparse.ArgumentTypeError(
                    f"Invalid CUDA device format: {device_str}"
                )
            try:
                idx = int(parts[1])
            except Exception as err:
                raise argparse.ArgumentTypeError(
                    f"Invalid CUDA device index: {parts[1]}"
                ) from err
            if idx < 0 or idx >= torch.cuda.device_count():
                raise argparse.ArgumentTypeError(
                    f"CUDA device index out of range: {idx}"
                )
            devices.append(f"cuda:{idx}")
        else:
            raise argparse.ArgumentTypeError(f"Unsupported device: {device_str}")

    # Remove duplicates while preserving order
    seen = set()
    unique_devices = []
    for d in devices:
        if d not in seen:
            seen.add(d)
            unique_devices.append(d)

    return unique_devices


def validate_image_size(image_height: int, image_width: int):
    """
    Validate image dimensions for Qwen2.5-VL.

    Rules:
    1. image_height and image_width must both be multiples of patch_size (14).
    2. After patchify, H and W must be divisible by merge_size (2),
       i.e. image_height and image_width must be multiples of 28.
    3. The total pixel count (height * width) must be within the range:
         - Minimum: 200704
         - Maximum: 399840

    Raises:
        ValueError: If any constraint is violated.
    """
    patch_size = 14
    merge_size = 2
    merge_patch = patch_size * merge_size  # 28

    # Rule 1: must be divisible by patch size
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError(
            f"Invalid image size: height ({image_height}) and width ({image_width}) "
            f"must be multiples of patch_size={patch_size}."
        )

    # Rule 2: must be divisible by merge size after patchify
    if image_height % merge_patch != 0 or image_width % merge_patch != 0:
        raise ValueError(
            f"Invalid image size for Qwen2.5-VL merge: "
            f"height ({image_height}) and width ({image_width}) must be multiples "
            f"of patch_size * merge_size = {merge_patch}. "
            f"(Got H%{merge_patch}={image_height % merge_patch}, "
            f"W%{merge_patch}={image_width % merge_patch})"
        )

    # Rule 3: pixel count range check
    min_pixels = 1024 * (patch_size * patch_size)
    max_pixels = 2040 * (patch_size * patch_size)
    pixels = image_height * image_width

    if not (min_pixels <= pixels <= max_pixels):
        raise ValueError(
            f"Invalid image pixel count: height * width = {pixels}, "
            f"expected range [{min_pixels}, {max_pixels}]."
        )


def main():
    # set model_name help string
    model_help_parts = [
        f"    - {model}: {', '.join(get_marches_with_model(model))}"
        for model in get_supported_models()
    ]
    model_help = "Model name. Supported models and their marches:\n" + "\n".join(
        model_help_parts
    )

    parser = argparse.ArgumentParser(
        description="Compile a Large Language Model for deployment on hardware.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=model_help,
    )
    parser.add_argument(
        "--march",
        type=str,
        required=True,
        choices=get_supported_marches(),
        help="Target hardware architecture for compilation. (Required)",
    )
    parser.add_argument(
        "--input_model_path",
        type=validated_path(check_exists=True),
        required=True,
        help="Path to the source model directory. (Required)",
    )
    parser.add_argument(
        "--output_model_path",
        type=validated_path(check_exists=False),
        required=True,
        help="Path to save the compiled model. (Required)",
    )
    parser.add_argument(
        "--cache_len",
        type=int,
        default=4096,
        help="Maximum sequence length for the KV-cache. (default: 4096). "
        "Note: cache_len must be greater than chunk_size and a multiple of 64.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Number of tokens per prefill chunk. (default: 256)",
    )
    parser.add_argument(
        "--decode_seq_len",
        type=int,
        default=1,
        help=(
            "Number of tokens per decode step during compilation. "
            "Only applies to qwen3. Ignored when --speculative_algorithm=EAGLE3 "
            "(uses --speculative_num_draft_tokens instead). (default: 1)"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Compile-time batch size for supported text graphs. "
            "Currently only qwen2_5-vl models support values > 1. "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--vision_tokens_num",
        type=int,
        default=144,
        help=(
            "Number of vision tokens for pi0/pi05 model. "
            "Set this value if visual token reduction is applied. (default: 144)"
        ),
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=448,
        help="Input image width. (default: 448)",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=448,
        help="Input image height. (default: 448)",
    )
    parser.add_argument(
        "--max_time_per_fc",
        type=float,
        default=0.0,
        help="Set maximum time constraint (unit:us) for per funccall. (default: 0.0). ",
    )
    parser.add_argument(
        "--device",
        type=validate_device,
        default=["cpu"],
        help=(
            "Device(s) for torch model during calibration: 'cpu' (default), "
            "'cuda:0', 'cuda:1 cuda:2', or 'cuda:1,cuda:2' for multi-GPU. "
            "Multi-GPU is only supported for qwen3 llm models and qwen2_5-vl models. "
            "Using CUDA can accelerate calibration."
        ),
    )
    parser.add_argument(
        "--config_path",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the config file for spirit_v1_5 model. (Optional)",
    )

    parser.add_argument(
        "--calib_image_path",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the calibration dataset for vision models. (Optional)",
    )
    parser.add_argument(
        "--calib_audio_path",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the calibration dataset for asr models. (Optional)",
    )
    parser.add_argument(
        "--calib_text_path",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the calibration JSON file or directory of JSON files. (Optional)",
    )
    parser.add_argument(
        "--calib_action_data_path",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the npy file or directory for VLA action expert. (Optional)",
    )
    parser.add_argument(
        "--calib_json_path",
        type=validated_path(check_exists=True),
        default=None,
        help=(
            "Path to the calibration JSON file used for model calibration. (Optional)"
        ),
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=50,
        help="Number of action tokens in action expert model. (default: 50)",
    )
    parser.add_argument(
        "--calib_tsv_path",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the calibration TSV file or directory. (Optional)",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        default=8,
        help=("Weight quantization bits, 4 or 8. (Optional)"),
    )
    parser.add_argument(
        "--input_model_format",
        type=str,
        choices=["hf", "llmc", "github"],
        default="hf",
        help=(
            "Input model format. "
            "'hf' for HuggingFace format, "
            "'llmc' for LLMC internal format, "
            "'github' for GitHub format. "
            "(default: hf). (Optional)"
        ),
    )
    parser.add_argument(
        "--weight_scales_file",
        type=validated_path(check_exists=True),
        default=None,
        help="Path to the weight scales file. (Optional)",
    )
    parser.add_argument(
        "-v",
        "--verifier",
        action="store_true",
        help=(
            "Run consistency verification after compilation using the built-in "
            "verifier tool. (Optional)"
        ),
    )
    parser.add_argument(
        "--remote_ip",
        type=str,
        default=None,
        help="Remote IP address for HBM model. (Optional)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=22,
        help="Port for remote HBM connection",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="root",
        help="Username for remote HBM connection",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Password for remote HBM connection",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        default="/userdata/data/hbm_infer",
        help="Remote path for HBM inference",
    )
    parser.add_argument(
        "--vit_core_num",
        type=(lambda v: [int(x.strip()) for x in v.split(",") if x.strip() != ""]),
        default=[1],
        help=(
            "VIT model multi bpu core num list, only for j6p & vlm model. "
            "Allowed values: 1,2,4. Example: --vit_core_num 1,4"
        ),
    )
    parser.add_argument(
        "--prefill_core_num",
        type=(lambda v: [int(x.strip()) for x in v.split(",") if x.strip() != ""]),
        default=[1],
        help=(
            "LLM prefill model multi bpu core num list, only for j6p. "
            "Allowed values: 1,2,4. Example: --prefill_core_num 1,2"
        ),
    )
    parser.add_argument(
        "--decode_core_num",
        type=(lambda v: [int(x.strip()) for x in v.split(",") if x.strip() != ""]),
        default=[1],
        help=(
            "LLM decode model multi bpu core num list, only for j6p. "
            "Allowed values: 1,2,4. Example: --decode_core_num 2"
        ),
    )
    parser.add_argument(
        "--kept_tokens_file",
        type=str,
        default=None,
        help="compressed token file",
    )

    parser.add_argument(
        "--cache_path",
        type=validated_path(check_exists=False),
        default=None,
        help=(
            "Build cache path , if not exists , create."
            "Cache path can speed up compiling next. (Optional)"
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=32,
        help="Number of threads to use during compilation.",
    )
    parser.add_argument(
        "--hidden_rotation_path",
        type=validated_path(check_exists=True),
        default=None,
        help=(
            "Optional .pt hidden rotation for LocateAnything. When omitted, "
            "the validated 2048-d S600 reference Hadamard is used."
        ),
    )
    parser.add_argument(
        "--disable_hidden_rotation",
        action="store_true",
        help="Disable LocateAnything hidden-domain folding for controlled RCA only.",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Export LocateAnything BC graphs and stop before convert/HBO compile.",
    )

    # ============ EAGLE3 Speculative Decoding Parameters ============
    parser.add_argument(
        "--speculative_algorithm",
        type=str,
        default=None,
        choices=["EAGLE3"],
        help=(
            "Speculative decoding algorithm. Currently only 'EAGLE3' is supported. "
            "(Optional, only for Qwen3 series)"
        ),
    )
    parser.add_argument(
        "--speculative_draft_model_path",
        type=validated_path(check_exists=True),
        default=None,
        help=(
            "Path to the speculative draft model (EAGLE layer weights). "
            "Required when speculative_algorithm is set. (Optional)"
        ),
    )
    parser.add_argument(
        "--speculative_num_steps",
        type=int,
        default=7,
        help=(
            "Number of tree expansion steps for EAGLE3 draft model. "
            "Controls the depth of the draft tree. (default: 7)"
        ),
    )
    parser.add_argument(
        "--speculative_eagle_topk",
        type=int,
        default=8,
        help=(
            "Top-k candidates per tree level for EAGLE3. "
            "Controls the branching factor of the draft tree. (default: 8)"
        ),
    )
    parser.add_argument(
        "--speculative_num_draft_tokens",
        type=int,
        default=32,
        help=(
            "Total number of draft tokens in the draft pool. "
            "Typical values: 32, 40, 48, 50, 56, 60. (default: 32)"
        ),
    )

    args = parser.parse_args()

    if not 256 <= args.cache_len <= 4096:
        parser.error(
            f"--cache_len ({args.cache_len}) must be within the range [256, 4096]!"
        )

    if not 128 <= args.chunk_size <= 2048:
        parser.error(
            f"--chunk_size ({args.chunk_size}) must be within the range [128, 2048]!"
        )

    if args.batch_size < 1:
        parser.error(f"--batch_size ({args.batch_size}) must be at least 1.")

    if args.model_name == "qwen3":
        if args.decode_seq_len < 1:
            parser.error(
                f"--decode_seq_len ({args.decode_seq_len}) must be at least 1."
            )
        if args.decode_seq_len > args.cache_len:
            parser.error(
                f"--decode_seq_len ({args.decode_seq_len}) must not exceed "
                f"--cache_len ({args.cache_len})."
            )

    if (
        args.batch_size != 1
        and args.model_name not in ["qwen2_5-vl-3b", "qwen2_5-vl-7b"]
    ):
        parser.error(
            f"--batch_size ({args.batch_size}) is only supported for "
            "qwen2_5-vl-3b and qwen2_5-vl-7b."
        )

    _qwen2_5_vl_models = ("qwen2_5-vl-3b", "qwen2_5-vl-7b")
    if args.model_name in _qwen2_5_vl_models or "qwen3" in args.model_name:
        if args.model_name in _qwen2_5_vl_models:
            validate_image_size(args.image_height, args.image_width)

        valid_values = [1, 2, 4]

        def _validate_single_value(name, values):
            if len(values) != 1:
                parser.error(
                    f"{name} must contain exactly one value for model "
                    f"{args.model_name}. Example: --{name} 1"
                )
            v = values[0]

            if v not in valid_values:
                parser.error(
                    f"{name} must be one of {valid_values} for model "
                    f"{args.model_name}, got {v}"
                )

        if args.model_name in _qwen2_5_vl_models:
            _validate_single_value("vit_core_num", args.vit_core_num)

        _validate_single_value("prefill_core_num", args.prefill_core_num)
        _validate_single_value("decode_core_num", args.decode_core_num)

        if args.prefill_core_num[0] != args.decode_core_num[0]:
            parser.error(
                f"prefill_core_num and decode_core_num must be the same for model "
                f"{args.model_name}. Got prefill_core_num={args.prefill_core_num[0]}, "
                f"decode_core_num={args.decode_core_num[0]}"
            )

    if (
        args.cache_len <= args.chunk_size
        or args.cache_len % 64 != 0
        or args.chunk_size % 64 != 0
    ):
        parser.error(
            f"--cache_len ({args.cache_len}) must be greater than "
            f"--chunk_size ({args.chunk_size}), and both params "
            "must be a multiple of 64"
        )

    if args.w_bits != 4 and args.w_bits != 8 and "internvl" in args.model_name:
        parser.error(f"--w_bits ({args.w_bits}) must be 4 or 8.")

    # EAGLE3 parameter validation
    if args.speculative_algorithm == "EAGLE3":
        if "qwen3" not in args.model_name:
            parser.error(
                f"--speculative_algorithm=EAGLE3 is only supported for Qwen3 series "
                f"models. Got model_name={args.model_name}"
            )

        if not args.speculative_draft_model_path:
            parser.error(
                "--speculative_draft_model_path is required when "
                "--speculative_algorithm=EAGLE3"
            )

        if args.speculative_num_steps < 1:
            parser.error(
                f"--speculative_num_steps ({args.speculative_num_steps}) must be >= 1"
            )

        if args.speculative_eagle_topk < 1:
            parser.error(
                f"--speculative_eagle_topk ({args.speculative_eagle_topk}) must be >= 1"
            )

        if args.speculative_num_draft_tokens < 1:
            parser.error(
                f"--speculative_num_draft_tokens ({args.speculative_num_draft_tokens}) "
                f"must be >= 1"
            )

    if args.verifier and not (args.remote_ip and args.remote_ip.strip()):
        parser.error("--remote_ip is required when verifier is enabled")

    # if need add parser args in compile, only add parser
    compile_kwargs = DEFAULT_COMPILE_KWARGS.copy()

    if args.jobs:
        if args.jobs < 1:
            parser.error(f"--jobs ({args.jobs}) must be at least 1.")
        compile_kwargs["jobs"] = args.jobs

    if args.cache_path:
        Path(args.cache_path).mkdir(parents=True, exist_ok=True)
        compile_kwargs["cache_mode"] = "enable"
        compile_kwargs["cache_path"] = args.cache_path
        print("build speed up , cache path : ", compile_kwargs["cache_path"])

    for k, v in args.__dict__.items():
        if k not in compile_kwargs:
            continue
        compile_kwargs[k] = v

    if args.model_name in [
        "deepseek-qwen-1_5b",
        "qwen2_5-omni-3b",
        "qwen2_5-vl-3b",
        "qwen2_5-vl-7b",
        "pi0",
        "smolvla",
        "pi05",
        "whisper",
        "qwen3",
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
    ]:
        compile_kwargs["input_no_padding"] = True
        compile_kwargs["output_no_padding"] = True

    if "internvl" in args.model_name:
        ## multi bpu setting
        valid_values = [1, 2, 4]

        def _validate_core_list(name, values):
            if any(v not in valid_values for v in values):
                parser.error(f"{name} must be from [1, 2, 4] for internvl/vlm")

        _validate_core_list("vit_core_num", args.vit_core_num)
        _validate_core_list("prefill_core_num", args.prefill_core_num)
        _validate_core_list("decode_core_num", args.decode_core_num)

    model = create_model_api(args.model_name, args)
    if not model:
        print(
            f"Error: model '{args.model_name}' is not supported. "
            f"Supported models: {', '.join(get_supported_models())}"
        )
        sys.exit(1)

    vit_compile_kwargs = compile_kwargs.copy()
    vit_compile_kwargs.update(VIT_COMPILE_KWARGS)

    llm_compile_kwargs = compile_kwargs.copy()
    llm_compile_kwargs.update(LLM_COMPILE_KWARGS)

    model.compile(vit_kwargs=vit_compile_kwargs, llm_kwargs=llm_compile_kwargs)

    if not args.verifier:
        return

    from leap_llm.apis.verifier.types import VerifierArgs
    from leap_llm.apis.verifier_cli import verify_model

    # Determine quantized BC file paths from API helper when available.
    hbm_llm_model_path: str | None = None
    hbm_vlm_model_path: str | None = None

    if hasattr(model, "get_hbm_path"):
        paths = model.get_hbm_path()
        if len(paths) == 1:
            hbm_llm_model_path = paths[0]
        elif len(paths) == 2:
            hbm_llm_model_path, hbm_vlm_model_path = paths

    # Convert device list to single device string for verifier (use first device)
    device_str = args.device[0] if isinstance(args.device, list) else args.device

    verifier_args = VerifierArgs(
        model_name=args.model_name,
        model_dir=args.input_model_path,
        compare_mode="hbm",
        input_text_path=args.calib_text_path,
        input_image_path=args.calib_image_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device_str,
        hbm_llm_model_path=hbm_llm_model_path,
        hbm_vlm_model_path=hbm_vlm_model_path,
        remote_ip=args.remote_ip,
        port=args.port,
        username=args.username,
        password=args.password,
        remote_path=args.remote_path,
    )

    verify_model(verifier_args)


if __name__ == "__main__":
    main()
