import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = (
    Path(__file__).parents[1]
    / "compiler"
    / "scripts"
    / "calibration/export_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("export_calibration_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_export_inputs_writes_valid_npy_and_manifest(tmp_path):
    source = tmp_path / "tensors"
    output = tmp_path / "npy"
    source.mkdir()
    for index in range(2):
        torch.save(
            {"vision_input": torch.full(MODULE.VISION_INPUT_SHAPE, index, dtype=torch.float16)},
            source / f"sample-{index}.pt",
        )

    result = MODULE.export_inputs(source, output)

    assert result["source_count"] == 2
    assert result["output_count"] == 2
    assert len(result["records"]) == 2
    value = np.load(output / "sample-1.npy", allow_pickle=False)
    assert value.shape == MODULE.VISION_INPUT_SHAPE
    assert value.dtype == np.float16
    assert np.all(value == 1)
    manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"


def test_export_inputs_rejects_invalid_shape(tmp_path):
    source = tmp_path / "tensors"
    source.mkdir()
    torch.save({"vision_input": torch.zeros((1, 2), dtype=torch.float16)}, source / "bad.pt")

    with pytest.raises(RuntimeError, match="vision_input shape"):
        MODULE.export_inputs(source, tmp_path / "npy")


def test_export_inputs_rejects_non_fp16_source(tmp_path):
    source = tmp_path / "tensors"
    source.mkdir()
    torch.save(
        {"vision_input": torch.zeros(MODULE.VISION_INPUT_SHAPE, dtype=torch.float32)},
        source / "bad.pt",
    )

    with pytest.raises(RuntimeError, match="expected torch.float16"):
        MODULE.export_inputs(source, tmp_path / "npy")


def test_parser_has_only_input_and_output_directories():
    root = MODULE.parser()
    options = {
        option
        for action in root._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert options == {"--input_dir", "--output_dir"}
