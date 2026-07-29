from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "compiler"
    / "scripts"
    / "validate/probe_language_rope.py"
)
SPEC = importlib.util.spec_from_file_location("language_rope_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sentinel_cache_encodes_position_and_channel() -> None:
    cache = MODULE.sentinel_cache(positions=4, channels=3)

    assert cache.shape == (4, 3)
    assert cache.dtype == np.float32
    np.testing.assert_array_equal(
        cache,
        np.asarray(
            [
                [0, 1, 2],
                [1000, 1001, 1002],
                [2000, 2001, 2002],
                [3000, 3001, 3002],
            ],
            dtype=np.float32,
        ),
    )


def test_q1_expected_gather_has_language_rope_layout() -> None:
    cache = MODULE.sentinel_cache()
    position_ids = MODULE.probe_position_ids(1)
    expected = MODULE.expected_gather(cache, position_ids)

    assert position_ids.tolist() == [[[7]]]
    assert expected.shape == (1, 1, 1, MODULE.CACHE_CHANNELS)
    np.testing.assert_array_equal(expected[0, 0, 0], cache[7])


def test_q6_expected_gather_preserves_query_order() -> None:
    cache = MODULE.sentinel_cache()
    position_ids = MODULE.probe_position_ids(6)
    expected = MODULE.expected_gather(cache, position_ids)

    assert position_ids.reshape(-1).tolist() == [5, 1, 7, 0, 9, 3]
    assert expected.shape == (1, 1, 6, MODULE.CACHE_CHANNELS)
    for query, position in enumerate(position_ids.reshape(-1)):
        np.testing.assert_array_equal(expected[0, 0, query], cache[position])


def test_q6_comparison_detects_old_extra_transpose_layout() -> None:
    cache = MODULE.sentinel_cache()
    position_ids = MODULE.probe_position_ids(6)
    expected = MODULE.expected_gather(cache, position_ids)
    old_layout = cache[position_ids.reshape(-1)].T.reshape(expected.shape)

    comparison = MODULE.compare_output(expected, old_layout)

    assert comparison["status"] == "value_mismatch"
    assert comparison["matched"] is False
    assert comparison["mismatch_count"] > 0
    assert comparison["max_abs"] > 0


def test_parser_exposes_only_output_directory() -> None:
    parser = MODULE.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }

    assert options == {"--output_dir"}
