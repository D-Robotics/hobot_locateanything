from pathlib import Path

import numpy as np
import pytest
import torch

from compiler.scripts.calibration.prepare import build_parser, save_tensor_artifact


def test_save_tensor_artifact_supports_pt_and_npy(tmp_path):
    vision_input = torch.zeros((1, 2304, 588), dtype=torch.float16)
    vision_input[..., 0] = -1
    vision_input[..., -1] = 1
    payload = {"vision_input": vision_input, "metadata": "kept-in-pt"}

    pt_path = tmp_path / "sample.pt"
    npy_path = tmp_path / "sample.npy"
    save_tensor_artifact(pt_path, "pt", payload, torch)
    save_tensor_artifact(npy_path, "npy", payload, torch)

    loaded_pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    loaded_npy = np.load(npy_path, allow_pickle=False)
    assert loaded_pt["metadata"] == "kept-in-pt"
    np.testing.assert_array_equal(loaded_npy, vision_input.numpy())


def test_save_tensor_artifact_rejects_non_fp16_npy(tmp_path):
    payload = {"vision_input": torch.zeros((1, 2304, 588), dtype=torch.float32)}
    with pytest.raises(TypeError, match="expected torch.float16"):
        save_tensor_artifact(tmp_path / "sample.npy", "npy", payload, torch)


def test_generate_parser_defaults_to_pt_and_accepts_npy():
    parser = build_parser()
    common = [
        "generate",
        "--selected-jsonl", "selected.jsonl",
        "--output-dir", "output",
        "--model-path", "model",
    ]
    assert parser.parse_args(common).output_format == "pt"
    assert parser.parse_args(common).upstream_repo is None
    assert parser.parse_args([*common, "--output-format", "npy"]).output_format == "npy"
