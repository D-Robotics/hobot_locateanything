from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from compiler.scripts.common import hybrid_generation as hybrid


def test_hybrid_generation_default_budget_matches_official_worker() -> None:
    assert hybrid.HybridGenerationConfig().max_new_tokens == 2048


BOX_START = 151668
BOX_END = 151669
COORD_START = 151677
IM_END = 151645


class FakeEmbedding:
    def __call__(self, token_ids):
        values = token_ids.to(torch.float32)
        return torch.stack((values, values + 1), dim=-1)


class FakeModel:
    def __init__(self):
        self.embed_tokens = FakeEmbedding()
        self.calls = []

    def __call__(self, embeds, positions, attention, *caches):
        q_len = int(embeds.shape[1])
        self.calls.append(
            {
                "q_len": q_len,
                "positions": positions.detach().clone(),
                "cache_shapes": [tuple(cache.shape) for cache in caches],
            }
        )
        logits = torch.zeros((1, q_len, 16), dtype=torch.float32)
        new_key = torch.full((1, q_len, 1, 1), float(q_len))
        new_value = torch.full((1, q_len, 1, 1), float(q_len + 10))
        return logits, [new_key], [new_value]


class FakeEmulator:
    def __init__(self):
        self.stage = None
        self.enabled = False
        self.history = []

    def set_stage(self, stage):
        self.stage = stage

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.history.append((self.stage, self.enabled))


def fake_official_decoding() -> hybrid.OfficialDecoding:
    module = ModuleType("fake_official_decoding")
    pbd_outputs = [
        [BOX_START, COORD_START, 7],
        [IM_END, 0, 0],
    ]
    ar_outputs = [COORD_START + 1, BOX_END]

    def sample_tokens(logits, generated, token_ids, **kwargs):
        del generated, token_ids, kwargs
        q_len = int(logits.shape[1])
        probabilities = torch.zeros_like(logits)
        confidence = torch.ones((1, q_len))
        if q_len > 1:
            selected = torch.tensor([pbd_outputs.pop(0)], dtype=torch.long)
            decoded = torch.zeros((1, 1), dtype=torch.long)
        else:
            selected = torch.tensor([[ar_outputs.pop(0)]], dtype=torch.long)
            decoded = None
        return probabilities, confidence, selected, decoded

    def handle_pattern(selected, token_ids, generation_mode):
        del token_ids
        assert generation_mode == "hybrid"
        tokens = [int(token) for token in selected.tolist()]
        if tokens[0] == IM_END:
            return {"type": "im_end", "tokens": [IM_END]}
        return {
            "type": "error_box",
            "tokens": [BOX_START, COORD_START],
        }

    module.sample_tokens = sample_tokens
    module.handle_pattern = handle_pattern
    module.get_token_ids_from_config = lambda config: config.token_ids
    return hybrid.OfficialDecoding(module, Path("fake/generate_utils.py"), "0" * 64)


def payload_without_saved_prediction():
    return {
        "prompt_input_ids": torch.tensor([[10, 11]], dtype=torch.long),
        "prompt_attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "projected_visual_features": torch.empty((1, 0, 2)),
        "special_token_ids": {
            "<box>": BOX_START,
            "</box>": BOX_END,
            "<ref>": 151672,
            "</ref>": 151673,
            "<null>": 152678,
            "<text_mask>": 151676,
            "<|im_end|>": IM_END,
        },
    }


def fake_replay():
    def build_prefill_inputs(
        model,
        payload,
        rotation,
        *,
        chunk_size,
        cache_len,
        image_token_id,
        device,
        dtype,
    ):
        del rotation, cache_len, image_token_id
        ids = payload["prompt_input_ids"].reshape(-1).to(device)
        active_len = int(payload["prompt_attention_mask"].sum().item())
        active = model.embed_tokens(ids).to(dtype=dtype)
        embeds = torch.zeros((1, chunk_size, active.shape[-1]), dtype=dtype)
        embeds[0, :active_len] = active[:active_len]
        positions = torch.arange(chunk_size, dtype=torch.int32).view(1, 1, -1)
        attention = torch.zeros((1, 1, chunk_size, chunk_size), dtype=dtype)
        return embeds, positions, attention, active_len

    def build_right_aligned_caches(
        new_keys, new_values, *, active_len, cache_len
    ):
        keys = []
        values = []
        for new_key, new_value in zip(new_keys, new_values, strict=True):
            key = torch.zeros((1, cache_len, 1, 1))
            value = torch.zeros((1, cache_len, 1, 1))
            key[:, -active_len:] = new_key[:, :active_len]
            value[:, -active_len:] = new_value[:, :active_len]
            keys.append(key)
            values.append(value)
        return keys, values

    def build_decode_inputs(
        model,
        token_ids,
        *,
        q_len,
        past_len,
        cache_len,
        is_pbd,
        pbd_prefix_len=0,
        device,
        dtype,
    ):
        del is_pbd, pbd_prefix_len
        ids = torch.tensor(token_ids[:q_len], device=device)
        embeds = model.embed_tokens(ids).to(dtype=dtype).unsqueeze(0)
        positions = torch.arange(
            past_len, past_len + q_len, dtype=torch.int32
        ).view(1, 1, -1)
        attention = torch.zeros((1, 1, q_len, cache_len), dtype=dtype)
        return embeds, positions, attention

    return SimpleNamespace(
        build_prefill_inputs=build_prefill_inputs,
        build_right_aligned_caches=build_right_aligned_caches,
        build_decode_inputs=build_decode_inputs,
    )


def test_fixed_graph_hybrid_runs_full_pbd_ar_pbd_loop_without_saved_output():
    model = FakeModel()
    emulator = FakeEmulator()
    generator = hybrid.FixedGraphHybridGenerator(
        model,
        torch.eye(2),
        torch.device("cpu"),
        torch.float32,
        [torch.zeros((1, 16, 1, 1)), torch.zeros((1, 16, 1, 1))],
        emulator,
        fake_official_decoding(),
        chunk_size=4,
        cache_len=16,
        pbd_query_len=3,
        image_token_id=151665,
        token_ids={
            "box_start_token_id": BOX_START,
            "box_end_token_id": BOX_END,
            "ref_start_token_id": 151672,
            "ref_end_token_id": 151673,
            "coord_start_token_id": COORD_START,
            "coord_end_token_id": 152677,
            "none_token_id": 4064,
            "null_token_id": 152678,
            "im_end_token_id": IM_END,
            "default_mask_token_id": 151676,
        },
        model_max_length=64,
        replay=fake_replay(),
    )

    result = generator.generate(
        payload_without_saved_prediction(),
        quantized=True,
        seed=7,
        config=hybrid.HybridGenerationConfig(max_new_tokens=12),
    )

    assert result["response_token_ids"] == [
        BOX_START,
        COORD_START,
        COORD_START + 1,
        BOX_END,
        IM_END,
    ]
    assert result["stop_reason"] == "im_end"
    assert result["pbd_calls"] == 2
    assert result["pbd_fused_prefix_calls"] == 1
    assert result["q1_commit_calls"] == 1
    assert result["ar_fallback_sample_calls"] == 2
    assert result["final_history_len"] == 6
    assert [call["q_len"] for call in model.calls] == [4, 3, 2, 1, 4]
    assert all(
        shape == (1, 16, 1, 1)
        for call in model.calls[1:]
        for shape in call["cache_shapes"]
    )
    assert emulator.enabled is False


def test_load_official_decoding_uses_checkpoint_file(tmp_path):
    source = tmp_path / "generate_utils.py"
    source.write_text(
        "def sample_tokens(*args, **kwargs):\n"
        "    return args, kwargs\n\n"
        "def handle_pattern(*args, **kwargs):\n"
        "    return args, kwargs\n\n"
        "def get_token_ids_from_config(config):\n"
        "    return config.token_ids\n",
        encoding="utf-8",
    )

    loaded = hybrid.load_official_decoding(tmp_path)

    assert loaded.source == source.resolve()
    assert len(loaded.sha256) == 64
    assert callable(loaded.sample_tokens)
    assert callable(loaded.handle_pattern)


def test_generation_config_rejects_non_hybrid_mode():
    config = hybrid.HybridGenerationConfig(generation_mode="slow")

    try:
        config.validate()
    except ValueError as error:
        assert "hybrid" in str(error)
    else:
        raise AssertionError("non-hybrid configuration was accepted")
