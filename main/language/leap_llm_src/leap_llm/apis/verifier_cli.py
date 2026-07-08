import argparse
import os
import sys
from typing import Dict, List

from hbdk4.compiler import Hbm, load
from leap_llm.apis.calibration.data_loader import (
    load_image_data,
    load_message_data,
    load_text_data,
)
from leap_llm.apis.verifier.backends import (
    Backend,
)
from leap_llm.apis.verifier.comparison_reporter import ComparisonReporter
from leap_llm.apis.verifier.types import VerifierArgs

SUPPORTED_MODELS = [
    "deepseek-qwen-1_5b",
    "deepseek-qwen-7b",
    "internvl2-2b",
    "internvl2_5-2b",
    "internvl2-1b",
    "internvl2_5-1b",
    "internvl3_5-1b",
    "qwen2_5-vl-3b",
    "qwen2_5-vl-7b",
]


def _validate_args(args: VerifierArgs) -> None:
    """Validate the verifier arguments based on the comparison mode."""
    if not os.path.isdir(args.model_dir):
        raise ValueError(f"Model directory not found: {args.model_dir}")
    if args.compare_mode not in {"bc", "hbm"}:
        raise ValueError("compare_mode must be either 'bc' or 'hbm'")

    if args.compare_mode == "bc":
        if not (args.quant_llm_model_path or args.quant_vlm_model_path):
            raise ValueError(
                "BC mode requires at least one of: quant_llm_model_path "
                "or quant_vlm_model_path"
            )
    elif args.compare_mode == "hbm":
        if not (args.hbm_llm_model_path or args.hbm_vlm_model_path or args.remote_ip):
            raise ValueError(
                "HBM mode requires at least one of: hbm_llm_model_path or "
                "hbm_vlm_model_path or remote_ip"
            )

    if args.input_json_path:
        if args.input_text_path or args.input_image_path:
            raise ValueError(
                "Cannot specify input_text_path or input_image_path "
                "when input_json_path is provided"
            )
        try:
            next(load_message_data(args.input_json_path))
        except (StopIteration, FileNotFoundError) as e:
            raise ValueError(
                f"input_json_path is provided but contains no valid data: "
                f"{args.input_json_path}"
            ) from e

    # Validate that provided dataset paths contain valid data
    if args.input_text_path:
        try:
            next(load_text_data(args.input_text_path))
        except (StopIteration, FileNotFoundError) as e:
            raise ValueError(
                f"input_text_path is provided but contains no valid data: "
                f"{args.input_text_path}"
            ) from e

    if args.input_image_path:
        try:
            next(load_image_data(args.input_image_path))
        except (StopIteration, FileNotFoundError) as e:
            raise ValueError(
                f"input_image_path is provided but contains no valid data: "
                f"{args.input_image_path}"
            ) from e


def verify_model(verifier_args: VerifierArgs):
    """Verify model outputs by running inference and generating reports."""
    _validate_args(verifier_args)

    backend = Backend(verifier_args)
    reporter = ComparisonReporter(verifier_args.model_name)

    if verifier_args.compare_mode == "bc":
        run_llm = bool(verifier_args.quant_llm_model_path)
        run_vlm = bool(verifier_args.quant_vlm_model_path)
        if verifier_args.quant_llm_model_path:
            quant_model = load(verifier_args.quant_llm_model_path)
    elif verifier_args.compare_mode == "hbm":
        run_llm = bool(verifier_args.hbm_llm_model_path)
        run_vlm = bool(verifier_args.hbm_vlm_model_path)
        if verifier_args.hbm_llm_model_path:
            quant_model = Hbm(verifier_args.hbm_llm_model_path)
    else:
        run_llm = run_vlm = False

    if run_llm:
        if quant_model is not None:
            graph = quant_model.graphs[0]
            # output_name = graph.outputs[0].name
            # print(f"first output : {output_name}")
            output_shape = graph.outputs[0].type.shape
            if output_shape[-2] == 1:
                backend.calib_data_preparer.set_full_logits(False)

        for i, text_prompt in enumerate(
            backend._load_text_data(
                verifier_args.model_name,
                verifier_args.input_json_path,
                verifier_args.input_text_path,
            )
        ):
            torch_outputs, torch_layers = backend._run_torch_llm(text_prompt)
            sim_outputs, sim_layers = backend.run_llm(text_prompt)

            if verifier_args.compare_mode == "bc":
                llm_payload = [
                    [torch_outputs, sim_outputs],
                    [torch_layers, sim_layers],
                ]
            else:
                llm_payload = [
                    [torch_outputs, sim_outputs],
                ]

            llm_inference_results: Dict[str, List] = {"llm": llm_payload}
            reporter.compare_inference_results(
                llm_inference_results,
                verifier_args.compare_mode,
                prompt_id=f"prompt_{i}",
            )

    if run_vlm:
        for i, image_tensor in enumerate(
            backend._load_image_data(
                model_type=verifier_args.model_name,
                input_json_path=verifier_args.input_json_path,
                input_image_path=verifier_args.input_image_path,
                image_width=verifier_args.image_width,
                image_height=verifier_args.image_height,
            )
        ):
            torch_outputs, torch_layers = backend._run_torch_vlm(image_tensor)
            sim_outputs, sim_layers = backend.run_vlm(image_tensor)

            if verifier_args.compare_mode == "bc":
                vlm_payload = [
                    [torch_outputs, sim_outputs],
                    [torch_layers, sim_layers],
                ]
            else:
                vlm_payload = [
                    [torch_outputs, sim_outputs],
                ]

            vlm_inference_results: Dict[str, List] = {"vlm": vlm_payload}
            reporter.compare_inference_results(
                vlm_inference_results,
                verifier_args.compare_mode,
                image_id=f"image_{i}",
            )

    reporter.generate_reports()


def main():
    """Parse command-line arguments and run the verifier."""
    parser = argparse.ArgumentParser(
        description="Verify model accuracy by comparing PyTorch and Quantized model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (required)",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Original model directory (required)",
    )
    parser.add_argument(
        "--quant_llm_model_path",
        type=str,
        required=False,
        help="Path to the quantized LLM model for BC comparison (required for BC mode)",
    )
    parser.add_argument(
        "--quant_vlm_model_path",
        type=str,
        required=False,
        help="Path to the quantized VLM model for BC comparison (required for BC mode)",
    )
    parser.add_argument(
        "--hbm_llm_model_path",
        type=str,
        required=False,
        help="Path to HBM model file for LLM inference.(Required for HBM mode)",
    )
    parser.add_argument(
        "--hbm_vlm_model_path",
        type=str,
        required=False,
        help="Path to HBM model file for VLM inference.(Required for HBM mode)",
    )
    parser.add_argument(
        "--input_text_path",
        type=str,
        required=False,
        help="Path to text data for input (Optional)",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        required=False,
        help="Path to image data for input (Optional)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        default=256,
        help="Chunk size, default is 256 (optional)",
    )
    parser.add_argument(
        "--cache_len",
        type=int,
        required=False,
        default=4096,
        help="Cache length, default is 4096 (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="Device to run on, default is cpu (optional)",
    )
    parser.add_argument(
        "--remote_ip",
        type=str,
        required=False,
        help="Remote IP address for HBM inference.(Required for HBM mode)",
    )
    parser.add_argument(
        "--username",
        type=str,
        required=False,
        default="root",
        help=(
            "Username for remote HBM connection.(Required for HBM mode, "
            "default is root)"
        ),
    )
    parser.add_argument(
        "--password",
        type=str,
        required=False,
        default="",
        help=(
            "Password for remote HBM connection.(Required for HBM mode, "
            "default is empty)"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=22,
        help="Port for remote HBM connection.(Required for HBM mode, default is 22)",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        required=False,
        default="/tmp/",
        help="Remote path for HBM inference.(Required for HBM mode, default is /tmp/)",
    )
    parser.add_argument(
        "--kept_tokens_file",
        type=str,
        required=False,
        default=None,
        help="Compressed vocab token file for compressed model",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        required=False,
        default=448,
        help="Image width for vision model (default: 448)",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        required=False,
        default=448,
        help="Image height for vision model (default: 448)",
    )
    parser.add_argument(
        "--input_json_path",
        type=str,
        required=False,
        help="Path to conversation data for input (Optional)",
    )
    args = parser.parse_args()

    # Automatically determine compare_mode based on provided paths
    compare_mode = None
    if args.quant_llm_model_path or args.quant_vlm_model_path:
        compare_mode = "bc"
    elif args.hbm_llm_model_path or args.hbm_vlm_model_path:
        compare_mode = "hbm"
    else:
        raise ValueError(
            "No comparison model path provided. Please specify at least one of: "
            "quant_llm_model_path, quant_vlm_model_path, hbm_llm_model_path, "
            "or hbm_vlm_model_path"
        )

    verifier_args = VerifierArgs(
        model_name=args.model_name,
        model_dir=args.model_dir,
        compare_mode=compare_mode,
        input_text_path=args.input_text_path,
        input_image_path=args.input_image_path,
        input_json_path=args.input_json_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        quant_llm_model_path=args.quant_llm_model_path,
        quant_vlm_model_path=args.quant_vlm_model_path,
        hbm_llm_model_path=args.hbm_llm_model_path,
        hbm_vlm_model_path=args.hbm_vlm_model_path,
        remote_ip=args.remote_ip,
        username=args.username,
        password=args.password,
        port=args.port,
        remote_path=args.remote_path,
        kept_tokens_file=args.kept_tokens_file,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    try:
        verify_model(verifier_args)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
