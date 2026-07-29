from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = (
    Path(__file__).parents[1] / "compiler/scripts/build/language.sh",
    Path(__file__).parents[1] / "compiler/scripts/build/vision.sh",
)


def load_oellm_build():
    compiler_root = Path(__file__).parents[1] / "compiler"
    if str(compiler_root) not in sys.path:
        sys.path.insert(0, str(compiler_root))
    from leap_llm.apis import oellm_build

    return oellm_build


def parse_release_profile(tmp_path: Path, model_name: str, *overrides: str):
    build = load_oellm_build()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    parser = build.build_parser()
    base = [
        "--model_name", model_name,
        "--march", "nash-p",
        "--input_model_path", str(checkpoint),
        "--output_model_path", str(tmp_path / "output"),
    ]
    if model_name == "locateanything-vit-3b":
        base.extend(["--image_width", "672", "--image_height", "672"])
    elif model_name == "locateanything-lm-3b":
        base.extend(
            [
                "--chunk_size", "1024",
                "--cache_len", "4096",
                "--decode_seq_len", "6",
            ]
        )
    return build, parser, parser.parse_args([*base, *overrides])


def test_compile_wrappers_validate_all_four_evidence_inputs_before_build():
    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        validation, build = text.split(
            'env PYTHONUNBUFFERED=1 PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" oellm_build',
            1,
        )
        for argument in (
            "--selected-jsonl",
            "--generated-jsonl",
            "--scale-manifest",
            "--coverage-json",
            "--expected-samples",
        ):
            assert argument in validation, f"{path.name} misses validation argument {argument}"
        assert 'EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-}"' in validation
        assert "--calibration_scale_manifest" in build
        assert "--calib_json_path" not in build
        assert "--calib_image_path" not in build


def test_compile_wrappers_forward_rotation_contract_to_validation_and_build():
    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert 'VALIDATION_ARGS+=(--hidden-rotation-path "$HIDDEN_ROTATION_PATH")' in text
        assert "VALIDATION_ARGS+=(--disable-hidden-rotation)" in text
        assert 'EXTRA_ARGS+=(--hidden_rotation_path "$HIDDEN_ROTATION_PATH")' in text
        assert "EXTRA_ARGS+=(--disable_hidden_rotation)" in text


def test_language_compile_validates_only_language_observer_evidence():
    text = SCRIPTS[0].read_text(encoding="utf-8")
    validation = text.split(
        'env PYTHONUNBUFFERED=1 PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" oellm_build',
        1,
    )[0]
    assert "--component language" in validation


def test_build_wrappers_have_explicit_wait_detach_and_target_contracts():
    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert 'WAIT="${WAIT:-1}"' in text
        assert 'DETACH="${DETACH:-0}"' in text
        assert 'BUILD_TARGET must be bc or hbm' in text
        assert 'if [[ "$DETACH" == "1" || "$WAIT" == "0" ]]' in text
        assert 'status=${PIPESTATUS[0]}' in text
        assert "BUILD_TARGET must be bc, hbo" not in text


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (("--image_width", "448"), "--image_width 672 --image_height 672"),
        (("--w_bits", "4"), "--w_bits 8"),
    ],
)
def test_oellm_build_rejects_nonrelease_locateanything_vision_profiles(
    tmp_path, capsys, overrides, message
):
    build, parser, args = parse_release_profile(
        tmp_path, "locateanything-vit-3b", *overrides
    )

    with pytest.raises(SystemExit):
        build.validate_args(parser, args)

    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (("--chunk_size", "512"), "--chunk_size 1024 --cache_len 4096 --decode_seq_len 6"),
        (("--cache_len", "2048"), "--chunk_size 1024 --cache_len 4096 --decode_seq_len 6"),
        (("--decode_seq_len", "1"), "--chunk_size 1024 --cache_len 4096 --decode_seq_len 6"),
        (("--w_bits", "4"), "--w_bits 8 --lm_head_w_bits 8"),
        (("--lm_head_w_bits", "4"), "--w_bits 8 --lm_head_w_bits 8"),
        (("--no-fused_pbd_profiles",), "--fused_pbd_profiles"),
    ],
)
def test_oellm_build_rejects_nonrelease_locateanything_language_profiles(
    tmp_path, capsys, overrides, message
):
    build, parser, args = parse_release_profile(
        tmp_path, "locateanything-lm-3b", *overrides
    )

    with pytest.raises(SystemExit):
        build.validate_args(parser, args)

    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "model_name", ("locateanything-vit-3b", "locateanything-lm-3b")
)
def test_oellm_build_accepts_locateanything_release_profiles(tmp_path, model_name):
    build, parser, args = parse_release_profile(tmp_path, model_name)

    build.validate_args(parser, args)


def test_oellm_build_keeps_qwen_baseline_contract_independent(tmp_path):
    build, parser, args = parse_release_profile(tmp_path, "qwen2_5-vl-3b")

    build.validate_args(parser, args)
