import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reference = load(ROOT / "compiler" / "scripts" / "validate/prepare_evaluation.py", "heldout_reference")
runner = load(ROOT / "deploy" / "run_s600_heldout.py", "s600_heldout")


def test_letterbox_transform_matches_672_profile():
    value = reference.letterbox_transform(1920, 1080, 672, 672)
    assert value["resized_size"] == [672, 378]
    assert value["padding_ltrb"] == [0, 147, 0, 147]
    assert value["letterbox_fill"] == 128


def test_final_response_uses_last_end_callback():
    text = "[callback] END: first\nnoise\n[callback] END: <ref>x</ref><box><1><2></box>\n"
    assert runner.final_response(text) == "<ref>x</ref><box><1><2></box>"


def test_final_response_is_none_when_callback_missing():
    assert runner.final_response("[demo] xlm_infer ret=1") is None
