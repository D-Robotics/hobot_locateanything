from __future__ import annotations

import torch

from compiler.leap_llm.language_graphs import language_graph_set
from compiler.leap_llm.apis.calibration.locateanything_replay import (
    append_cache_updates,
    replay_sequential_ar_q1,
)


def test_language_graph_sets_are_exact_and_ordered():
    assert language_graph_set("standard").graphs == (
        "prefill",
        "decode",
        "decode_ar",
    )
    assert language_graph_set("fused_decode").graphs == (
        "prefill",
        "decode",
        "decode_ar",
        "decode_pbd_q7",
        "decode_pbd_q8",
        "decode_pbd_q9",
        "decode_pbd_q10",
        "decode_pbd_q11",
        "decode_pbd_q12",
        "decode_ar_q2",
        "decode_ar_q3",
        "decode_ar_q4",
        "decode_ar_q5",
    )


def test_standard_calibration_requires_six_sequential_ar_calls_per_context():
    standard = language_graph_set("standard")
    assert standard.calibration_stages == ("prefill", "pbd_q6", "ar_q1")
    assert standard.calibration_execution_counts(7) == {
        "prefill": 7,
        "pbd_q6": 7,
        "ar_q1": 42,
    }


def test_cache_updates_are_committed_between_ar_calls():
    keys = [torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]]]])]
    values = [keys[0] + 10]

    for token in (20.0, 21.0, 22.0):
        update = [torch.tensor([[[[token]]]])]
        keys, values = append_cache_updates(
            keys,
            values,
            update,
            [update[0] + 10],
            accepted=1,
        )

    torch.testing.assert_close(
        keys[0].reshape(-1), torch.tensor([3.0, 20.0, 21.0, 22.0])
    )
    torch.testing.assert_close(
        values[0].reshape(-1), torch.tensor([13.0, 30.0, 31.0, 32.0])
    )


def test_standard_ar_replay_advances_position_and_commits_each_token():
    calls = []

    def build_inputs(_model, token_ids, **kwargs):
        calls.append({"token": token_ids[0], "past_len": kwargs["past_len"]})
        return token_ids[0], kwargs["past_len"], None

    class Model:
        def __init__(self):
            self.cache_inputs = []

        def __call__(self, token, _position, _mask, key_cache, value_cache):
            self.cache_inputs.append(key_cache.clone())
            key = torch.tensor([[[[float(token)]]]])
            return None, [key], [key + 100]

    model = Model()
    keys = [torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)]
    values = [keys[0] + 100]
    keys, values = replay_sequential_ar_q1(
        model,
        range(10, 16),
        keys,
        values,
        active_len=23,
        cache_len=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        input_builder=build_inputs,
    )

    assert calls == [
        {"token": token, "past_len": 23 + offset}
        for offset, token in enumerate(range(10, 16))
    ]
    assert [cache.reshape(-1)[-1].item() for cache in model.cache_inputs] == [
        7.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
    ]
    torch.testing.assert_close(
        keys[0].reshape(-1), torch.tensor([6.0, 7.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    )
    torch.testing.assert_close(values[0], keys[0] + 100)
