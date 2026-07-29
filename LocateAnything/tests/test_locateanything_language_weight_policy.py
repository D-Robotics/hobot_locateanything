from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "compiler/leap_llm/models/locateanything/config/locateanything_3b.py"
)
SPEC = importlib.util.spec_from_file_location("locateanything_config", CONFIG_PATH)
config_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(config_module)


def test_language_config_has_independent_lm_head_weight_bits():
    config = config_module.Qwen2PBDTextConfig()
    assert config.w_bits == 8
    assert config.lm_head_w_bits == 8


def test_language_compile_sources_encode_decoder_w8_lmhead_w8_policy():
    attention = (
        ROOT
        / "compiler/leap_llm/models/locateanything/blocks/text_attention_leap.py"
    ).read_text(encoding="utf-8")
    mlp = (
        ROOT
        / "compiler/leap_llm/models/locateanything/blocks/text_mlp_leap.py"
    ).read_text(encoding="utf-8")
    model = (
        ROOT / "compiler/leap_llm/models/locateanything/text_model_leap.py"
    ).read_text(encoding="utf-8")
    api = (
        ROOT / "compiler/leap_llm/apis/model/locateanything_language.py"
    ).read_text(encoding="utf-8")
    compile_script = (
        ROOT / "compiler/scripts/build/language.sh"
    ).read_text(encoding="utf-8")
    replay_script = (
        ROOT / "compiler/scripts/calibration/calibrate.py"
    ).read_text(encoding="utf-8")

    assert attention.count("w_bits=config.w_bits") == 4
    assert mlp.count("w_bits=config.w_bits") == 3
    assert "w_bits=config.lm_head_w_bits" in model
    assert 'LM_HEAD_W_BITS="${LM_HEAD_W_BITS:-8}"' in compile_script
    assert 'W_BITS="${W_BITS:-8}"' in compile_script
    assert "device=args.device, w_bits=8, lm_head_w_bits=args.lm_head_w_bits" in replay_script
    assert "select_pbd_tokens(" in replay_script
    assert "self.text_cfg.num_hidden_layers * 7" in api
    assert 'weight_profile=f"decoder_w{w_bits}_lmhead_w{lm_head_w_bits}"' in api


def test_batch_one_rope_gather_preserves_query_channel_order():
    """q > 1 must not transpose gathered [query, channel] RoPE rows."""

    source = (
        ROOT / "compiler/leap_llm/models/locateanything/text_model_leap.py"
    ).read_text(encoding="utf-8")
    build = source.split("def build(", 1)[1].split("if self.use_plugin:", 1)[0]
    batch_one = build.split("else:", 1)[1]

    assert "cos = leap.gather_nd(self.cache_cos, position_ids, 0)" in batch_one
    assert "sin = leap.gather_nd(self.cache_sin, position_ids, 0)" in batch_one
    assert "leap.transpose(cos" not in batch_one
    assert "leap.transpose(sin" not in batch_one
