#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import tempfile

import torch

from leap_llm.apis.model.locateanything_language import load_language_state_dict
from leap_llm.apis.model.locateanything_vision import LocateAnythingVisionApi
from leap_llm.models.locateanything.config.locateanything_3b import (
    load_config_from_json,
)
from leap_llm.models.locateanything.hidden_rotation import (
    build_reference_hidden_rotation,
    rotate_language_to_hidden_domain,
)
from leap_llm.models.locateanything.text_model_leap import LocateAnythingTextModel


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1).double()
    right = right.reshape(-1).double()
    return float(
        torch.dot(left, right) / (torch.linalg.norm(left) * torch.linalg.norm(right))
    )


def report(prefix: str, candidate: torch.Tensor, reference: torch.Tensor) -> None:
    difference = candidate.float() - reference.float()
    print(f"{prefix}_cosine={cosine(candidate, reference):.12f}")
    print(f"{prefix}_max_diff={float(difference.abs().max()):.9g}")
    print(
        f"{prefix}_rmse="
        f"{float(torch.sqrt(torch.mean(difference.square()))):.9g}"
    )


def build_language(model_path: str, rotate: bool, device: str):
    config = load_config_from_json(f"{model_path}/config.json").text_config
    config.prefill_seq_len = 1024
    config.decode_seq_len = 6
    config.cache_len = 2048
    config.batch_size = 1
    config.w_bits = 4
    config.has_scale = False

    model = LocateAnythingTextModel(config, use_plugin=False)
    state_dict = load_language_state_dict(model_path)
    state_dict.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in missing
        if key not in {"cache_cos", "cache_sin"} and not key.startswith("lm_head.")
    ]
    if missing or unexpected:
        raise RuntimeError(f"Language state mismatch: missing={missing}, unexpected={unexpected}")
    model.tie_lm_head_to_embeddings()

    rotation = build_reference_hidden_rotation(config.hidden_size)
    if rotate:
        rotate_language_to_hidden_domain(model, rotation, device=device)
    model.compile_mode(False)
    return model, rotation


def validate_language(model_path: str, device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(20260718)
    query_length = 2 if dtype == torch.float32 else 6
    inputs = torch.randn(1, query_length, 2048, device=device, dtype=dtype)
    position_ids = torch.arange(
        query_length, device=device, dtype=torch.int32
    ).view(1, 1, query_length)
    mask = torch.zeros(
        1, query_length, query_length, device=device, dtype=dtype
    )

    raw_model, rotation = build_language(model_path, False, device)
    raw_model = raw_model.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        raw_logits, raw_keys, raw_values = raw_model(inputs, position_ids, mask)
    raw_logits = raw_logits.float().cpu()
    raw_keys = [value.float().cpu() for value in raw_keys]
    raw_values = [value.float().cpu() for value in raw_values]
    del raw_model
    torch.cuda.empty_cache()
    gc.collect()

    rotated_inputs = (inputs.float().cpu() @ rotation).to(device=device, dtype=dtype)
    rotated_model, _ = build_language(model_path, True, device)
    rotated_model = rotated_model.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        rotated_logits, rotated_keys, rotated_values = rotated_model(
            rotated_inputs, position_ids, mask
        )
    rotated_logits = rotated_logits.float().cpu()
    rotated_keys = [value.float().cpu() for value in rotated_keys]
    rotated_values = [value.float().cpu() for value in rotated_values]

    report("language_logits", rotated_logits, raw_logits)
    key_difference = max(
        float((candidate - reference).abs().max())
        for candidate, reference in zip(rotated_keys, raw_keys)
    )
    value_difference = max(
        float((candidate - reference).abs().max())
        for candidate, reference in zip(rotated_values, raw_values)
    )
    print(f"language_kv_max_diff={max(key_difference, value_difference):.9g}")


def validate_vision(model_path: str, device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(20260718)
    inputs = torch.randn(1, 1024, 588, device=device, dtype=dtype)

    with tempfile.TemporaryDirectory(prefix="la_rotation_") as output_dir:
        raw_api = LocateAnythingVisionApi(
            model_path,
            output_dir,
            device=device,
            vit_core_num=[4],
            apply_hidden_rotation=False,
        )
        raw_model = raw_api.model
        raw_model.compile_mode(False)
        raw_model = raw_model.to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            raw_output = raw_model(inputs).float().cpu()
        del raw_api, raw_model
        torch.cuda.empty_cache()
        gc.collect()

        rotated_api = LocateAnythingVisionApi(
            model_path,
            output_dir,
            device=device,
            vit_core_num=[4],
            apply_hidden_rotation=True,
        )
        rotated_model = rotated_api.model
        rotated_model.compile_mode(False)
        rotated_model = rotated_model.to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            rotated_output = rotated_model(inputs).float().cpu()

    expected = raw_output @ build_reference_hidden_rotation()
    report("vision_output", rotated_output, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--component", choices=("language", "vision", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.component in {"language", "all"}:
        validate_language(args.model_path, args.device, dtype)
    if args.component in {"vision", "all"}:
        validate_vision(args.model_path, args.device, dtype)


if __name__ == "__main__":
    main()
