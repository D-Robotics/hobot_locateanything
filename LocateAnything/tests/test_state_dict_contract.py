from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "compiler/leap_llm/apis/model/state_dict_contract.py"
)
SPEC = importlib.util.spec_from_file_location("state_dict_contract", MODULE_PATH)
contract = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(contract)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.register_buffer("derived_cache", torch.ones(2), persistent=False)


def test_state_dict_load_accepts_complete_weights_and_nonpersistent_cache():
    model = TinyModel()

    contract.load_state_dict_fail_closed(
        model,
        {"weight": torch.tensor([2.0, 3.0])},
        component="test",
    )

    torch.testing.assert_close(model.weight, torch.tensor([2.0, 3.0]))
    assert "derived_cache" not in model.state_dict()


def test_state_dict_load_rejects_missing_parameter():
    with pytest.raises(RuntimeError, match=r"missing=\['weight'\]"):
        contract.load_state_dict_fail_closed(
            TinyModel(),
            {},
            component="test",
        )


def test_state_dict_load_rejects_unexpected_parameter():
    with pytest.raises(RuntimeError, match=r"unexpected=\['obsolete.weight'\]"):
        contract.load_state_dict_fail_closed(
            TinyModel(),
            {
                "weight": torch.ones(2),
                "obsolete.weight": torch.ones(2),
            },
            component="test",
        )


def test_locateanything_derived_rope_buffers_are_nonpersistent_and_loads_are_closed():
    root = Path(__file__).resolve().parents[1]
    language_model = (
        root / "compiler/leap_llm/models/locateanything/text_model_leap.py"
    ).read_text(encoding="utf-8")
    vision_model = (
        root / "compiler/leap_llm/models/locateanything/vision_model_leap.py"
    ).read_text(encoding="utf-8")
    language_api = (
        root / "compiler/leap_llm/apis/model/locateanything_language.py"
    ).read_text(encoding="utf-8")
    vision_api = (
        root / "compiler/leap_llm/apis/model/locateanything_vision.py"
    ).read_text(encoding="utf-8")

    assert 'register_buffer("cache_cos", cache_cos, persistent=False)' in language_model
    assert 'register_buffer("cache_sin", cache_sin, persistent=False)' in language_model
    assert 'register_buffer("rope_cos", cos.to(torch.float32), persistent=False)' in vision_model
    assert 'register_buffer("rope_sin", sin.to(torch.float32), persistent=False)' in vision_model
    assert "load_state_dict_fail_closed(" in language_api
    assert "load_state_dict_fail_closed(" in vision_api
    assert "WARN missing" not in language_api
    assert "WARN unexpected" not in language_api
    assert "WARN missing" not in vision_api
    assert "WARN unexpected" not in vision_api
