import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "compiler" / "scripts" / "common/quantization.py"
SPEC = importlib.util.spec_from_file_location("locateanything_quantization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_dynamic_activation_qdq_uses_one_scale_per_last_dimension_row():
    value = torch.tensor(
        [[[1.0, -2.0, 0.5], [10.0, -4.0, 1.0]]], dtype=torch.float32
    )

    dequantized, quantized, scale = MODULE.dynamic_activation_qdq(value)

    expected = value.abs().amax(dim=-1, keepdim=True) / 127.0 + 2.0**-16
    assert scale.shape == (1, 2, 1)
    assert torch.allclose(scale, expected)
    assert int(quantized.min()) >= -127
    assert int(quantized.max()) <= 127
    assert torch.allclose(dequantized, quantized * scale)


def test_nonnegative_u8_uses_all_codes_and_preserves_more_small_probabilities():
    value = torch.tensor([[[0.003, 0.006, 0.02, 1.0]]], dtype=torch.float32)

    u8_value, u8_integer, u8_scale = MODULE.dynamic_nonnegative_u8_qdq(value)
    s8_value, s8_integer, _ = MODULE.dynamic_activation_qdq(value)

    assert int(u8_integer.min()) >= 0
    assert int(u8_integer.max()) <= 255
    assert u8_scale.shape == (1, 1, 1)
    assert int((u8_value == 0).sum()) < int((s8_value == 0).sum())


def test_nonnegative_u8_rejects_signed_values():
    with pytest.raises(ValueError, match="nonnegative U8"):
        MODULE.dynamic_nonnegative_u8_qdq(torch.tensor([[-0.01, 0.5]]))


def test_centered_value_wv_affine_identity_is_exact_without_qdq_error():
    torch.manual_seed(7)
    attention = torch.softmax(torch.randn(1, 2, 5, 5), dim=-1)
    value_transposed = torch.randn(1, 2, 3, 5)
    value_mean = value_transposed.mean(dim=-1, keepdim=True)
    centered_value = value_transposed - value_mean
    expected = attention @ value_transposed.transpose(-1, -2)
    actual = attention @ centered_value.transpose(-1, -2)
    actual += attention.sum(dim=-1, keepdim=True) * value_mean.transpose(-1, -2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_centered_wv_uses_s8_and_preserves_output_shape():
    torch.manual_seed(11)
    attention = torch.softmax(torch.randn(1, 2, 5, 5), dim=-1)
    value_transposed = torch.randn(1, 2, 3, 5)

    result = MODULE.centered_value_wv_qdq(attention, value_transposed)

    assert result["output"].shape == (1, 2, 5, 3)
    assert int(result["attention_int"].min()) >= -127
    assert int(result["attention_int"].max()) <= 127
    assert int(result["value_int"].min()) >= -127
    assert int(result["value_int"].max()) <= 127
    assert torch.isfinite(result["output"]).all()


def test_centered_value_wv_uses_quantized_centered_value_and_exact_mean_term():
    torch.manual_seed(17)
    attention = torch.softmax(torch.randn(1, 2, 256, 256) * 4.0, dim=-1)
    value_transposed = torch.randn(1, 2, 32, 256) * 0.6 + 0.03

    result = MODULE.centered_value_wv_qdq(attention, value_transposed)
    expected = result["quant_attention"] @ result["quant_value"].transpose(-1, -2)
    expected += result["attention_sum"] * result["value_mean"].transpose(-1, -2)
    reference = attention @ value_transposed.transpose(-1, -2)

    torch.testing.assert_close(result["output"], expected)
    assert MODULE.tensor_comparison(reference, result["output"])["cosine"] > 0.999


def test_fake_matmul_centered_value_mode_preserves_shape_and_high_cosine():
    class Emulator:
        capture_operators = False

        @staticmethod
        def _rescue_mode(_module, _kind):
            return "centered_value_s8"

    module = type("FakeWV", (), {})()
    module._quant_emulator = Emulator()
    module._quant_name = "layers.2.self_attn.wv_matmul"
    torch.manual_seed(19)
    attention = torch.softmax(torch.randn(1, 2, 4, 5), dim=-1)
    value = torch.randn(1, 2, 5, 3)

    candidate = MODULE.QuantizationEmulator._fake_matmul_forward(
        module, attention, value
    )
    reference = attention @ value

    assert candidate.shape == reference.shape
    assert MODULE.tensor_comparison(reference, candidate)["cosine"] > 0.999


def test_weight_qdq_uses_one_scale_per_output_channel():
    weight = torch.tensor(
        [[-1.0, 0.5, 0.25], [8.0, -4.0, 2.0]], dtype=torch.float32
    )

    dequantized, quantized, scale = MODULE.weight_qdq(weight)

    assert scale.shape == (2, 1)
    assert scale[:, 0].tolist() == pytest.approx([1.0 / 127.0, 8.0 / 127.0])
    assert int(quantized.min()) >= -128
    assert int(quantized.max()) <= 127
    assert torch.allclose(dequantized, quantized * scale)


def test_static_activation_qdq_uses_manifest_absmax():
    value = torch.tensor([-3.0, 0.0, 3.0], dtype=torch.float32)

    dequantized, quantized, scale = MODULE.static_activation_qdq(
        value, absmax=4.0, bits=8
    )

    assert float(scale) == pytest.approx(4.0 / 127.0)
    assert int(quantized.min()) >= -127
    assert int(quantized.max()) <= 127
    assert torch.allclose(dequantized, quantized * scale)


def test_attention_quantizers_use_dynamic_per_row_a8_proposal():
    value = torch.tensor(
        [[[1.0, -2.0, 0.5], [10.0, -4.0, 1.0]]], dtype=torch.float32
    )

    dequantized, quantized, scale, kind = MODULE.activation_qdq(
        value,
        "blocks.24.qk_matmul.x_fake_quant",
        absmax=32.0,
        bits=8,
    )

    assert MODULE.is_attention_quantizer("blocks.24.qk_matmul.x_fake_quant")
    assert kind == "dynamic_attention_quantizer"
    assert scale.shape == (1, 2, 1)
    assert torch.allclose(dequantized, quantized * scale)


def test_non_attention_quantizer_keeps_static_manifest_scale():
    value = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)

    _dequantized, _quantized, scale, kind = MODULE.activation_qdq(
        value,
        "unrelated.fake_quant",
        absmax=4.0,
        bits=8,
    )

    assert not MODULE.is_attention_quantizer("unrelated.fake_quant")
    assert kind == "static_quantizer"
    assert float(scale) == pytest.approx(4.0 / 127.0)


def test_float_rescue_policy_is_stage_aware_and_reports_exact_matches():
    policy = MODULE.FloatRescuePolicy(
        [
            MODULE.FloatRescueRule(
                r"layers\.2\.self_attn\.cache_v_fq",
                stages=("pbd_q6", "ar_q1"),
                kinds=("static",),
            ),
            MODULE.FloatRescueRule(
                r"lm_head",
                stages=("ar_q1",),
                kinds=("linear",),
                mode="float_weight",
            ),
        ],
        name="value-cache-and-head",
    )
    policy.bind(
        [
            ("layers.2.self_attn.cache_v_fq", "static"),
            ("layers.2.self_attn.cache_k_fq", "static"),
            ("lm_head", "linear"),
        ]
    )

    assert policy.resolve("prefill", "layers.2.self_attn.cache_v_fq", "static") == "quantized"
    assert policy.resolve("pbd_q6", "layers.2.self_attn.cache_v_fq", "static") == "float"
    assert policy.resolve("ar_q1", "lm_head", "linear") == "float_weight"
    assert policy.resolve("pbd_q6", "lm_head", "linear") == "quantized"
    description = policy.describe()
    assert description["name"] == "value-cache-and-head"
    assert description["inventory_matches"][0]["matches"] == [
        {"module": "layers.2.self_attn.cache_v_fq", "kind": "static"}
    ]
    assert description["runtime_matches"] == {
        "ar_q1/lm_head/linear/float_weight": 1,
        "pbd_q6/layers.2.self_attn.cache_v_fq/static/float": 1,
    }


def test_float_rescue_policy_rejects_zero_inventory_matches():
    policy = MODULE.FloatRescuePolicy(
        [MODULE.FloatRescueRule(r"layers\.99\..*", kinds=("linear",))],
        name="missing-layer",
    )

    with pytest.raises(ValueError, match="matched no quantized modules"):
        policy.bind([("layers.2.self_attn.v_proj", "linear")])


def test_quantization_policy_accepts_nonnegative_u8_for_static_operand():
    policy = MODULE.FloatRescuePolicy(
        [
            MODULE.FloatRescueRule(
                r"layers\.2\.self_attn\.wv_matmul\.x_fake_quant",
                stages=("pbd_q6",),
                kinds=("static",),
                mode="nonnegative_u8",
            )
        ],
        name="wv-u8",
    )
    policy.bind([("layers.2.self_attn.wv_matmul.x_fake_quant", "static")])

    assert policy.resolve(
        "pbd_q6", "layers.2.self_attn.wv_matmul.x_fake_quant", "static"
    ) == "nonnegative_u8"


def test_quantization_policy_accepts_centered_value_s8_for_fake_wv_matmul():
    policy = MODULE.FloatRescuePolicy(
        [
            MODULE.FloatRescueRule(
                r"layers\.2\.self_attn\.wv_matmul",
                stages=("pbd_q6",),
                kinds=("fake_matmul",),
                mode="centered_value_s8",
            )
        ],
        name="centered-wv",
    )
    policy.bind([("layers.2.self_attn.wv_matmul", "fake_matmul")])

    assert policy.resolve(
        "pbd_q6", "layers.2.self_attn.wv_matmul", "fake_matmul"
    ) == "centered_value_s8"


def test_float_rescue_policy_rejects_partial_linear_modes_for_matmul():
    with pytest.raises(ValueError, match="invalid for"):
        MODULE.FloatRescuePolicy(
            [
                MODULE.FloatRescueRule(
                    r"layers\.2\.self_attn\.wv_matmul",
                    kinds=("matmul",),
                    mode="float_activation",
                )
            ],
            name="invalid-matmul-mode",
        )


def test_dynamic_quantizer_patterns_are_exact_and_fail_closed():
    first = type("StaticModule", (), {})()
    second = type("StaticModule", (), {})()
    emulator = MODULE.QuantizationEmulator.__new__(MODULE.QuantizationEmulator)
    emulator.static_modules = {
        "layers.2.self_attn.wv_matmul.x_fake_quant": first,
        "layers.2.self_attn.wv_matmul.y_fake_quant": second,
    }
    emulator.dynamic_quantizer_patterns = []

    result = emulator.set_dynamic_quantizer_patterns([
        r"layers\.2\.self_attn\.wv_matmul\.x_fake_quant"
    ])

    assert first._quant_dynamic_attention is True
    assert second._quant_dynamic_attention is False
    assert result["matches"] == {
        r"layers\.2\.self_attn\.wv_matmul\.x_fake_quant": [
            "layers.2.self_attn.wv_matmul.x_fake_quant"
        ]
    }
    with pytest.raises(ValueError, match="matched no static modules"):
        emulator.set_dynamic_quantizer_patterns([r"layers\.99\..*"])
    assert first._quant_dynamic_attention is True
    assert second._quant_dynamic_attention is False
